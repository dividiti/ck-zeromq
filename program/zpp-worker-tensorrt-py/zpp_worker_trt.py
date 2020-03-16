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
MODEL_PLUGIN_PATH       = os.getenv('ML_MODEL_TENSORRT_PLUGIN','')

if MODEL_PLUGIN_PATH:
    import ctypes
    if not os.path.isfile(MODEL_PLUGIN_PATH):
        raise IOError("{}\n{}\n".format(
            "Failed to load library ({}).".format(MODEL_PLUGIN_PATH),
            "Please build the plugin."
        ))
    ctypes.CDLL(MODEL_PLUGIN_PATH)

MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_INPUT_DATA_TYPE   = os.getenv('ML_MODEL_INPUT_DATA_TYPE', 'float32')
MODEL_DATA_TYPE         = os.getenv('ML_MODEL_DATA_TYPE', '(unknown)')
MODEL_SOFTMAX_LAYER     = os.getenv('CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME', os.getenv('CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME', ''))
MODEL_SUBTRACT_MEAN     = os.getenv('ML_MODEL_SUBTRACT_MEAN', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
if MODEL_SUBTRACT_MEAN:
    MODEL_GIVEN_CHANNEL_MEANS = os.getenv('ML_MODEL_GIVEN_CHANNEL_MEANS', '0.0 0.0 0.0')
    channel_means = np.fromstring(MODEL_GIVEN_CHANNEL_MEANS, dtype=np.float32, sep=' ')
    if MODEL_COLOURS_BGR:
        channel_means = channel_means[::-1]     # swapping Red and Blue colour channels

## Transfer mode (numpy floats by default):
#
TRANSFER_MODE           = os.getenv('CK_TRANSFER_MODE', 'numpy')
TRANSFER_FLOAT          = os.getenv('CK_TRANSFER_FLOAT', 'YES') in ('YES', 'yes', 'ON', 'on', '1')
PREPROCESS_ON_GPU       = (TRANSFER_FLOAT == False) and (TRANSFER_MODE != 'json') and os.getenv('CK_PREPROCESS_ON_GPU', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
CONVERSION_NEEDED       = (TRANSFER_FLOAT == False) and (MODEL_INPUT_DATA_TYPE == 'float32')
CONVERSION_TYPE_SYMBOL  = 'f' if (MODEL_INPUT_DATA_TYPE == 'float32') else 'b'
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


## CUDA/TensorRT model setup:
#
pycuda_context = pycuda.tools.make_default_context()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
try:
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
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
        if CONVERSION_NEEDED:
            d_preconverted_input = cuda.mem_alloc(size * 1)
    else:
        interface_type = 'Output'
        host_mem    = cuda.pagelocked_empty(size, trt.nptype(dtype))
        h_d_outputs.append({ 'host_mem': host_mem, 'dev_mem': dev_mem })
        if MODEL_SOFTMAX_LAYER=='' or interface_layer == MODEL_SOFTMAX_LAYER:
            model_output_shape  = shape
            h_output            = host_mem

    print("{} layer {}: dtype={}, shape={}, elements_per_max_batch={}".format(interface_type, interface_layer, dtype, shape, size))

cuda_stream         = cuda.Stream()
input_volume        = trt.volume(model_input_shape)     # total number of monochromatic subpixels (before batching)
output_volume       = trt.volume(model_output_shape)    # total number of elements in one image prediction (before batching)

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
print('Model input_volume: {}'.format(input_volume))
print('Model output_volume: {}'.format(output_volume))
print('Image transfer mode: {}'.format(TRANSFER_MODE))
print('Transferred images need to be converted to the input data type of the model: {}'.format(CONVERSION_NEEDED))
print('Transferred images need to be preprocessed (e.g. by subtracting means): {}'.format(PREPROCESS_ON_GPU))
print('Worker output format: {}'.format(WORKER_OUTPUT_FORMAT))

if CONVERSION_NEEDED:
    compilation_start = time.time();

    from pycuda.compiler import SourceModule
    # Define type conversion kernels and more. NB: Must be done after initializing CUDA context.
    source_module = SourceModule(source="""
    // See all type converstion (cast) built-in functionshere:
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__CAST.html
    // Convert signed 8-bit integer to 32-bit floating-point using round-to-nearest-even mode.
    __global__ void convert_int8_to_fp32(
        float * __restrict__ out, const signed char * __restrict__ in, long num_elems
    )
    {
        long idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >=  num_elems)
            return;

        out[idx] = __int2float_rn( (signed int) in[idx] );
    }

    // Convert unsigned 8-bit integer to 32-bit floating-point using round-to-nearest-even mode.
    __global__ void convert_uint8_to_fp32(
        float * __restrict__ out, const unsigned char * __restrict__ in, long num_elems
    )
    {
        long idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >=  num_elems)
            return;

        out[idx] = __int2float_rn( (unsigned int) in[idx] );
    }

    // Subtract channel means assuming NCHW layout.
    __global__ void subtract_means(float * data,
        float R_mean, float G_mean, float B_mean,
        long HW, // H*W
        long num_elems // N*C*H*W
    )
    {
        long idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_elems)
            return;

        switch ( (idx / HW) % 3 )
        {
            case 0:
                data[idx] -= R_mean;
                break;
            case 1:
                data[idx] -= G_mean;
                break;
            case 2:
                data[idx] -= B_mean;
                break;
        }
    }

    // Convert unsigned 8-bit integer to 32-bit floating-point using round-to-nearest-even mode,
    // and then subtract RGB channel means assuming NCHW layout.
    __global__ void convert_uint8_to_fp32_and_subtract_means(
        float * __restrict__ out, const unsigned char * __restrict__ in,
        float R_mean, float G_mean, float B_mean,
        long HW, // H*W
        long num_elems // N*C*H*W
    )
    {
        long idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_elems)
            return;

        // Convert.
        out[idx] = __int2float_rn( (unsigned int) in[idx] );

        // Subtract means.
        switch ( (idx / HW) % 3 )
        {
            case 0:
                out[idx] -= R_mean;
                break;
            case 1:
                out[idx] -= G_mean;
                break;
            case 2:
                out[idx] -= B_mean;
                break;
        }
    }
    """, cache_dir=False)

    if PREPROCESS_ON_GPU and MODEL_SUBTRACT_MEAN:
        conversion_kernel_name = 'convert_uint8_to_fp32_and_subtract_means' if CONVERSION_TYPE_SYMBOL == 'f' else None
        conversion_kernel = source_module.get_function(conversion_kernel_name)
#        subtract_means_kernel = source_module.get_function('subtract_means')
#        conversion_kernel_name = 'convert_uint8_to_fp32' if CONVERSION_TYPE_SYMBOL == 'f' else None
#        conversion_kernel = source_module.get_function(conversion_kernel_name)
    elif not PREPROCESS_ON_GPU:
        conversion_kernel_name = 'convert_int8_to_fp32' if CONVERSION_TYPE_SYMBOL == 'f' else None
        conversion_kernel = source_module.get_function(conversion_kernel_name)

    compilation_time_ms = (time.time() - compilation_start)*1000
    print("Compilation time of GPU kernel(s): {:.2f} ms".format(compilation_time_ms))

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
                job_data_raw    = memoryview( from_factory.recv() )
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

        # FIXME: floatize -> conversion?
        floatize_start = time.time()

        if TRANSFER_MODE == 'dummy':
            job_id, batch_size  = struct.unpack('ii', job_data_raw)
            converted_batch = None
        else:
            if TRANSFER_MODE == 'raw':
                job_id          = struct.unpack('<I', job_data_raw[:ID_SIZE_IN_BYTES])[0]
                batch_data      = job_data_raw[ID_SIZE_IN_BYTES:]
                num_raw_bytes   = len(batch_data)

                if CONVERSION_NEEDED:
                    batch_size      = num_raw_bytes // input_volume
                    converted_batch = None
                else:
                    batch_size      = num_raw_bytes // input_volume // model_input_type_size
                    converted_batch = batch_data
            elif TRANSFER_MODE in ('json', 'pickle', 'numpy'):
                job_id      = job_data_struct['job_id']
                batch_data  = job_data_struct['batch_data']
                batch_size  = len(batch_data) // input_volume
                if type(batch_data)==list: # json
                    converted_batch = struct.pack("{}{}".format(len(batch_data), CONVERSION_TYPE_SYMBOL), *batch_data)
                elif CONVERSION_NEEDED: # pickle, numpy
                    converted_batch = None
                else:
                    converted_batch = batch_data

            if converted_batch is not None:
                memcpy_htod_start = time.time()
                cuda.memcpy_htod_async(d_inputs[0], converted_batch, cuda_stream) # assuming one input layer for image classification
                memcpy_htod_time_ms = (time.time() - memcpy_htod_start)*1000
            else: # raw, pickle, numpy if CONVERSION_NEEDED
                # TODO: Read max dimensions from CUDA info. Currently taken from TX2.
                max_block_dim_x = 1024
                max_grid_dim_x = 2147483647
                max_x = max_block_dim_x * max_grid_dim_x
                # The kernel processes one element per thread, so we cannot exceed max_x elements.
                num_elems = len(batch_data)
                if num_elems >= max_x:
                    print("Error: Number of elements exceeds max dimension X: {} >= {}".format(num_elems, max_x))
                    pass
                # Copy input to the GPU.
                memcpy_htod_start = time.time()
                cuda.memcpy_htod_async(d_preconverted_input, batch_data, cuda_stream)
                memcpy_htod_time_ms = (time.time() - memcpy_htod_start )*1000
                # One thread per element. TODO: Number of threads can be tuned down e.g. halved.
                block_dim_x = int( max_block_dim_x / 1 )
                grid_dim_x = int( (num_elems + block_dim_x - 1) / block_dim_x )
                if PREPROCESS_ON_GPU:
                    if MODEL_SUBTRACT_MEAN:
                        (R_mean, G_mean, B_mean) = channel_means
                        conversion_kernel(d_inputs[0], d_preconverted_input,
                            R_mean, G_mean, B_mean, np.int64(MODEL_IMAGE_HEIGHT*MODEL_IMAGE_WIDTH), np.int64(num_elems), grid=(grid_dim_x,1,1), block=(block_dim_x,1,1))
                    # TODO: Implement other transforms e.g. normalization.
                else:
                    conversion_kernel(d_inputs[0], d_preconverted_input, np.int64(num_elems), grid=(grid_dim_x,1,1), block=(block_dim_x,1,1))


        if batch_size > max_batch_size:   # basic protection. FIXME: could report to hub, could split and still do inference...
            print("[worker {}] unable to perform inference on {}-sample batch. Skipping it.".format(WORKER_ID, batch_size))
            continue

        inference_start = time.time()

        if TRANSFER_MODE != 'dummy':
            trt_context.execute_async(bindings=model_bindings, batch_size=batch_size, stream_handle=cuda_stream.handle)
            for output in h_d_outputs:
                cuda.memcpy_dtoh_async(output['host_mem'], output['dev_mem'], cuda_stream)
            cuda_stream.synchronize()

        inference_time_ms           = (time.time() - inference_start)*1000 + memcpy_htod_time_ms
        floatize_time_ms            = (inference_start-floatize_start)*1000 - memcpy_htod_time_ms
        wait_and_receive_time_ms    = (floatize_start-wait_and_receive_start)*1000

        if TRANSFER_MODE == 'dummy':        # no inference - fake a batch
            merged_batch_predictions = [ 0 ] * output_volume * batch_size
        else:
            batch_results = h_output[:output_volume * batch_size].tolist()

            if WORKER_OUTPUT_FORMAT == 'direct_return':
                merged_batch_predictions = batch_results

            elif WORKER_OUTPUT_FORMAT == 'softmax':
                if output_volume == 1:          # model returns argmax - fake the softmax by padding with 1000 zeros (1001 overall)
                    merged_batch_predictions = []
                    for arg_max in batch_results:
                        merged_batch_predictions.extend( [0]*(arg_max +1) + [1] + [0]*(1000-arg_max-1) )
                else:                           # model returns softmax - just pass it on
                    merged_batch_predictions = batch_results

            elif WORKER_OUTPUT_FORMAT == 'argmax':
                if output_volume == 1:          # model returns argmax - just pass it on
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
