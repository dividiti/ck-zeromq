#!/usr/bin/env python3

#import numpy as np
import os
import struct
import time
import zmq

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools

## Transfer mode:
#
TRANSFER_MODE           = os.getenv('CK_ZMQ_TRANSFER_MODE', 'raw')
FP_MODE                 = os.getenv('CK_FP_MODE', 'NO') in ('YES', 'yes', 'ON', 'on', '1')

## ZMQ ports:
#
ZMQ_FAN_PORT            = os.getenv('CK_ZMQ_FAN_PORT', 5557)
ZMQ_FUNNEL_PORT         = os.getenv('CK_ZMQ_FUNNEL_PORT', 5558)
ZMQ_POST_WORK_TIMEOUT_S = os.getenv('CK_ZMQ_POST_WORK_TIMEOUT_S', '')   # empty string means no timeout

## Worker properties:
#
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
from_factory.connect('tcp://{}:{}'.format(HUB_IP, ZMQ_FAN_PORT))
if ZMQ_POST_WORK_TIMEOUT_S != '':
    from_factory.RCVTIMEO = int(ZMQ_POST_WORK_TIMEOUT_S)*1000   # expects milliseconds

to_funnel = zmq_context.socket(zmq.PUSH)
to_funnel.connect('tcp://{}:{}'.format(HUB_IP, ZMQ_FUNNEL_PORT))


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
model_monopixels    = trt.volume(model_input_shape)
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
print('Image transfer mode: {}'.format(TRANSFER_MODE))
print('Images transferred as floats: {}'.format(FP_MODE))
print("")

print("[worker {}] Ready to run inference on batches up to {} samples".format(WORKER_ID, max_batch_size))


## Main inference loop:
#
with trt_engine.create_execution_context() as trt_context:
    done_count = 0
    total_inference_time = 0
    while JOBS_LIMIT<1 or done_count < JOBS_LIMIT:

        try:
            if TRANSFER_MODE == 'dummy':
                job_data_raw        = from_factory.recv()
                job_id, batch_size  = struct.unpack('ii', job_data_raw )
            elif TRANSFER_MODE == 'raw':
                job_data_raw  = from_factory.recv()
                if FP_MODE:
                    job_id      = struct.unpack('i', job_data_raw[:4] )[0]
                    batch_size  = (len(job_data_raw)-4) // 4 // model_monopixels
                else:
                    batch_data  = list( struct.unpack('i{}b'.format(len(job_data_raw)-4), job_data_raw) )
                    job_id      = batch_data.pop(0)
                    batch_size  = len(batch_data) // model_monopixels
            else:
                if TRANSFER_MODE == 'json':
                    job_data_struct    = from_factory.recv_json()
                elif TRANSFER_MODE in ('pickle', 'numpy'):
                    job_data_struct    = from_factory.recv_pyobj()

                job_id      = job_data_struct['job_id']
                batch_data  = job_data_struct['batch_data']
                batch_size  = len(batch_data) // model_monopixels
        except Exception as e:
            print("Caught exception: {} , ExceptionType: {}".format(e, type(e)))
            continue

        if batch_size>max_batch_size:   # basic protection. FIXME: could report to hub, could split and still do inference...
            print("[worker {}] unable to perform inference on {}-sample batch. Skipping it.".format(WORKER_ID, batch_size))
            continue

        floatize_start = time.time()

        if TRANSFER_MODE == 'dummy':
            pass
        elif TRANSFER_MODE == 'raw' and FP_MODE:
            float_batch = job_data_raw[4:]
        elif TRANSFER_MODE == 'numpy' and FP_MODE:
            float_batch = batch_data
        else:
            #float_batch = np.array(batch_data, dtype=np.float32)
            float_batch = struct.pack("{}f".format(len(batch_data)), *batch_data) # almost twice as fast!

        inference_start = time.time()

        if TRANSFER_MODE != 'dummy':
            cuda.memcpy_htod_async(d_inputs[0], float_batch, cuda_stream)    # assuming one input layer for image classification
            trt_context.execute_async(bindings=model_bindings, batch_size=batch_size, stream_handle=cuda_stream.handle)
            for output in h_d_outputs:
                cuda.memcpy_dtoh_async(output['host_mem'], output['dev_mem'], cuda_stream)
            cuda_stream.synchronize()

        inference_time_ms   = (time.time() - inference_start)*1000
        floatize_time_ms    = (inference_start-floatize_start)*1000

        response = {
            'job_id': job_id,
            'worker_id': WORKER_ID,
            'floatize_time_ms': floatize_time_ms,
            'inference_time_ms': inference_time_ms,
            'raw_batch_results': [ 0 ]*1001*batch_size if TRANSFER_MODE == 'dummy' else h_output[:model_classes*batch_size].tolist(),
        }

        to_funnel.send_json(response)

        print("[worker {}] classified job_id={} [{}] in {:.2f} ms (after spending {:.2f} ms to convert to floats)".format(WORKER_ID, job_id, batch_size, inference_time_ms, floatize_time_ms))
        total_inference_time += inference_time_ms

        done_count += 1

    print("[worker {}] Total inference time: {}s".format(WORKER_ID, total_inference_time))

pycuda_context.pop()

