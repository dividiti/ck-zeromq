#!/usr/bin/env python3

import numpy as np
import os
import struct
import time
import zmq

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools


## Worker properties:
HUB_IP                  = os.getenv('CK_HUB_IP', 'localhost')
JOBS_LIMIT              = int(os.getenv('CK_WORKER_JOB_LIMIT', 0))
WORKER_ID               = os.getenv('CK_WORKER_ID') or os.getpid()


## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_DATA_TYPE         = os.getenv('ML_MODEL_DATA_TYPE', 'float32')
MODEL_SOFTMAX_LAYER     = os.getenv('CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME', os.getenv('CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME', ''))


## ZeroMQ communication setup:
#
zmq_context = zmq.Context()

from_factory = zmq_context.socket(zmq.PULL)
from_factory.connect('tcp://{}:5557'.format(HUB_IP))

to_funnel = zmq_context.socket(zmq.PUSH)
to_funnel.connect('tcp://{}:5558'.format(HUB_IP))


## CUDA/TRT model setup:
#
pycuda_context = pycuda.tools.make_default_context()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
try:
    with open(MODEL_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        serialized_engine = f.read()
        trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
        print('[TRT] successfully loaded')
except:
    pycuda_context.pop()
    raise RuntimeError('TensorRT model file {} is not found or corrupted'.format(MODEL_PATH))

max_batch_size      = trt_engine.max_batch_size

d_inputs, h_d_outputs, model_bindings = [], [], []
for interface_layer in trt_engine:
    dtype   = trt_engine.get_binding_dtype(interface_layer)
    shape   = trt_engine.get_binding_shape(interface_layer)
    size    = trt.volume(shape) * max_batch_size

    dev_mem = cuda.mem_alloc(size * dtype.itemsize)
    model_bindings.append( int(dev_mem) )

    if trt_engine.binding_is_input(interface_layer):
        interface_type = 'Input'
        d_inputs.append(dev_mem)
        model_input_shape   = shape
    else:
        interface_type = 'Output'
        host_mem    = cuda.pagelocked_empty(size, trt.nptype(dtype))
        h_d_outputs.append({ 'host_mem': host_mem, 'dev_mem': dev_mem })
        if MODEL_SOFTMAX_LAYER=='' or interface_layer == MODEL_SOFTMAX_LAYER:
            model_output_shape  = shape
            h_output            = host_mem

    print("{} layer {}: dtype={}, shape={}, elements_per_max_batch={}".format(interface_type, interface_layer, dtype, shape, size))

cuda_stream         = cuda.Stream()
model_classes       = trt.volume(model_output_shape)

if MODEL_DATA_LAYOUT == 'NHWC':
    (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS) = model_input_shape
else:
    (MODEL_IMAGE_CHANNELS, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH) = model_input_shape

print("Data layout: {}".format(MODEL_DATA_LAYOUT) )
print('Model image height: {}'.format(MODEL_IMAGE_HEIGHT))
print('Model image width: {}'.format(MODEL_IMAGE_WIDTH))
print('Model image channels: {}'.format(MODEL_IMAGE_CHANNELS))
print('Model data type: {}'.format(MODEL_DATA_TYPE))
print('Model BGR colours: {}'.format(MODEL_COLOURS_BGR))
print('Model max_batch_size: {}'.format(max_batch_size))
print("")

print("[worker {}] Ready to run inference on batches up to {} samples".format(WORKER_ID, max_batch_size))


## Main inference loop:
#
with trt_engine.create_execution_context() as trt_context:
    done_count = 0
    total_inference_time = 0
    while JOBS_LIMIT<1 or done_count < JOBS_LIMIT:
        job         = from_factory.recv_json()
        job_id      = job['job_id']
        batch_ids   = job['batch_ids']
        batch_data  = job['batch_data']
        batch_size  = int( len(batch_data)/(MODEL_IMAGE_HEIGHT*MODEL_IMAGE_WIDTH*MODEL_IMAGE_CHANNELS) )

        if batch_size>max_batch_size:   # basic protection. FIXME: could report to hub, could split and still do inference...
            print("[worker {}] unable to perform inference on {}-sample batch. Skipping it.".format(WORKER_ID, batch_size))
            continue

        bytize_start = time.time()

        #vectored_batch = np.array(batch_data, dtype=np.float32)
        vectored_batch = bytearray(struct.pack("{}f".format(len(batch_data)), *batch_data)) # almost twice as fast!

        inference_start = time.time()

        cuda.memcpy_htod_async(d_inputs[0], vectored_batch, cuda_stream)    # assuming one input layer for image classification
        trt_context.execute_async(bindings=model_bindings, batch_size=batch_size, stream_handle=cuda_stream.handle)
        for output in h_d_outputs:
            cuda.memcpy_dtoh_async(output['host_mem'], output['dev_mem'], cuda_stream)
        cuda_stream.synchronize()

        raw_batch_results = np.split(h_output, max_batch_size)

        inference_time_ms = (time.time() - inference_start)*1000

        response = {
            'job_id': job_id,
            'worker_id': WORKER_ID,
            'inference_time_ms': inference_time_ms,
            'batch_results': {},
        }
        for i in range(batch_size):
            sample_id = batch_ids[i]
            response['batch_results'][sample_id]=raw_batch_results[i].tolist()

        to_funnel.send_json(response)

        print("[worker {}] classified batch #{} in {:.2f} ms (after spending {:.2f} ms to convert to bytearray)".format(WORKER_ID, job_id, inference_time_ms, (inference_start-bytize_start)*1000))
        total_inference_time += inference_time_ms

        done_count += 1

    print("[worker {}] Total inference time: {}s".format(WORKER_ID, total_inference_time))

pycuda_context.pop()

