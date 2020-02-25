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


## ZMQ ports:
#
ZMQ_FAN_PORT            = os.getenv('CK_ZMQ_FAN_PORT', 5557)
ZMQ_FUNNEL_PORT         = os.getenv('CK_ZMQ_FUNNEL_PORT', 5558)

## Worker properties:
#
HUB_IP                  = os.getenv('CK_HUB_IP', 'localhost')
JOBS_LIMIT              = int(os.getenv('CK_WORKER_JOB_LIMIT', 0))
WORKER_ID               = os.getenv('CK_WORKER_ID') or os.getpid()
WORKER_OUTPUT_FORMAT    = os.getenv('CK_WORKER_OUTPUT_FORMAT', 'softmax')
WORKER_POSTWORK_TIMEOUT_S = os.getenv('CK_WORKER_POSTWORK_TIMEOUT_S', '')  # empty string means no timeout

## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_INPUT_DATA_TYPE   = os.getenv('ML_MODEL_INPUT_DATA_TYPE', 'float32')
MODEL_DATA_TYPE         = os.getenv('ML_MODEL_DATA_TYPE', '(unknown)')
MODEL_SOFTMAX_LAYER     = os.getenv('CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME', os.getenv('CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME', ''))

## Transfer mode:
#
TRANSFER_MODE           = os.getenv('CK_ZMQ_TRANSFER_MODE', 'raw')
CONVERSION_NEEDED       = (os.getenv('CK_FP_MODE', 'NO') not in ('YES', 'yes', 'ON', 'on', '1')) and (MODEL_INPUT_DATA_TYPE == 'float32')
CONVERT_TO_TYPE_CHAR    = 'f' if (MODEL_INPUT_DATA_TYPE == 'float32') else 'b'
ID_SIZE_IN_BYTES        = 4 # assuming uint32


## ZeroMQ communication setup:
#
zmq_context = zmq.Context()

from_factory = zmq_context.socket(zmq.PULL)
from_factory.connect('tcp://{}:{}'.format(HUB_IP, ZMQ_FAN_PORT))
if WORKER_POSTWORK_TIMEOUT_S != '':
    from_factory.RCVTIMEO = int(WORKER_POSTWORK_TIMEOUT_S)*1000   # expects milliseconds

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
        model_input_shape       = shape
        model_input_type_size   = dtype.itemsize
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
print('Model input data type: {}'.format(MODEL_INPUT_DATA_TYPE))
print('Model (internal) data type: {}'.format(MODEL_DATA_TYPE))
print('Model BGR colours: {}'.format(MODEL_COLOURS_BGR))
print('Model max_batch_size: {}'.format(max_batch_size))
print('Image transfer mode: {}'.format(TRANSFER_MODE))
print('Images transferred need to be converted to input data type of the model: {}'.format(CONVERSION_NEEDED))
print('Worker output format: {}'.format(WORKER_OUTPUT_FORMAT))
print("")

print("[worker {}] Ready to run inference on batches up to {} samples".format(WORKER_ID, max_batch_size))


## Main inference loop:
#
with trt_engine.create_execution_context() as trt_context:
    done_count = 0
    total_inference_time = 0
    while JOBS_LIMIT<1 or done_count < JOBS_LIMIT:

        wait_and_receive_start = time.time()

        try:
            if TRANSFER_MODE == 'dummy':
                job_data_raw    = from_factory.recv()
            elif TRANSFER_MODE == 'raw':
                job_data_raw    = from_factory.recv()
            elif TRANSFER_MODE == 'json':
                job_data_struct = from_factory.recv_json()
            elif TRANSFER_MODE in ('pickle', 'numpy'):
                job_data_struct = from_factory.recv_pyobj()
        except zmq.error.Again as e:    # ZeroMQ's timeout exception
            if done_count==0:
                print('.', end='', flush=True)
                continue
            else:
                print("Having done {} inference cycles, leaving after a timeout of {} seconds".format(
                                done_count, WORKER_POSTWORK_TIMEOUT_S))
                break

        floatize_start = time.time()

        converted_batch = None
        if TRANSFER_MODE == 'dummy':
            job_id, batch_size  = struct.unpack('ii', job_data_raw )
        else:
            if TRANSFER_MODE == 'raw':
                num_raw_bytes   = len(job_data_raw)-ID_SIZE_IN_BYTES
                if CONVERSION_NEEDED:
                    batch_data      = list( struct.unpack('i{}b'.format(num_raw_bytes), job_data_raw) )     # expensive
                    job_id          = batch_data.pop(0)
                    batch_size      = len(batch_data) // model_monopixels
                else:
                    converted_batch = job_data_raw[ID_SIZE_IN_BYTES:]
                    job_id          = struct.unpack('i', job_data_raw[:ID_SIZE_IN_BYTES] )[0]
                    batch_size      = num_raw_bytes // model_input_type_size // model_monopixels
            elif TRANSFER_MODE in ('json', 'pickle', 'numpy'):
                job_id      = job_data_struct['job_id']
                batch_data  = job_data_struct['batch_data']
                batch_size  = len(batch_data) // model_monopixels

            if not converted_batch:
                if type(batch_data)==list or CONVERSION_NEEDED:
                    converted_batch = struct.pack("{}{}".format(len(batch_data), CONVERT_TO_TYPE_CHAR), *batch_data)
                else:
                    converted_batch = batch_data

        if batch_size>max_batch_size:   # basic protection. FIXME: could report to hub, could split and still do inference...
            print("[worker {}] unable to perform inference on {}-sample batch. Skipping it.".format(WORKER_ID, batch_size))
            continue

        inference_start = time.time()

        if TRANSFER_MODE != 'dummy':
            cuda.memcpy_htod_async(d_inputs[0], converted_batch, cuda_stream)    # assuming one input layer for image classification
            trt_context.execute_async(bindings=model_bindings, batch_size=batch_size, stream_handle=cuda_stream.handle)
            for output in h_d_outputs:
                cuda.memcpy_dtoh_async(output['host_mem'], output['dev_mem'], cuda_stream)
            cuda_stream.synchronize()

        inference_time_ms           = (time.time() - inference_start)*1000
        floatize_time_ms            = (inference_start-floatize_start)*1000
        wait_and_receive_time_ms    = (floatize_start-wait_and_receive_start)*1000

        if WORKER_OUTPUT_FORMAT == 'softmax':
            if TRANSFER_MODE == 'dummy':        # no inference - fake a softmax batch
                merged_batch_predictions = [ 0 ]*1001*batch_size
            else:
                batch_results = h_output[:model_classes*batch_size].tolist()

                if model_classes == 1:          # model returns argmax - fake the softmax by padding with 1000 zeros (1001 overall)
                    merged_batch_predictions = []
                    for arg_max in batch_results:
                        merged_batch_predictions.extend( [0]*(arg_max +1) + [1] + [0]*(1000-arg_max-1) )
                else:                           # model returns softmax - just pass it on
                    merged_batch_predictions = batch_results

        elif WORKER_OUTPUT_FORMAT == 'argmax':
            if TRANSFER_MODE == 'dummy':        # no inference - fake an argmax batch
                merged_batch_predictions = [ 0 ]*batch_size
            else:
                batch_results = h_output[:model_classes*batch_size].tolist()

                if model_classes == 1:          # model returns argmax - just pass it on
                    merged_batch_predictions = batch_results
                else:                           # model returns softmax - filter it to return argmax
                    merged_batch_predictions = []

                    for j in range(batch_size): # walk through the batch and append individual argmaxen
                        one_argmax = max(zip(batch_results[j*1001:(j+1)*1001], range(1001)))[1]-1
                        merged_batch_predictions.append( one_argmax )

        response = {
            'job_id': job_id,
            'worker_id': WORKER_ID,
            'wait_and_receive_time_ms': wait_and_receive_time_ms,
            'floatize_time_ms': floatize_time_ms,
            'inference_time_ms': inference_time_ms,
            'raw_batch_results': merged_batch_predictions,
        }

        to_funnel.send_json(response)

        print("[worker {}] classified job_id={} [{}] in {:.2f} ms (after spending {:.2f} ms on waiting+receiving AND {:.2f} ms on type conversion)".format(WORKER_ID, job_id, batch_size, inference_time_ms, wait_and_receive_time_ms, floatize_time_ms))
        total_inference_time += inference_time_ms

        done_count += 1

    print("[worker {}] Total inference time: {}s".format(WORKER_ID, total_inference_time))

pycuda_context.pop()

