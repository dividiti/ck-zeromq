#!/usr/bin/env python3

import array
import os
import struct
import sys
import threading
import time

import numpy as np
import zmq
import mlperf_loadgen as lg

try:
    raw_input
except NameError:
    # Python 3
    raw_input = input

zmq_context = zmq.Context()

# Socket to send tasks on
to_workers = zmq_context.socket(zmq.PUSH)
to_workers.bind("tcp://*:5557")

# Socket to receive results on
from_workers = zmq_context.socket(zmq.PULL)
from_workers.bind("tcp://*:5558")
from_workers.RCVTIMEO = 2000

###########################################################################################################
## NB: if you run into "zmq.error.ZMQError: Address already in use" after a crash,
##     run "sudo netstat -ltnp | grep python" and kill the socket-hogging process.
###########################################################################################################

## Data transfer:
#
TRANSFER_MODE           = os.getenv('CK_ZMQ_TRANSFER_MODE', 'json')
FP_MODE                 = os.getenv('CK_FP_MODE', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
TRANSFER_TYPE_NP        = np.float32 if FP_MODE else np.int8
TRANSFER_TYPE_CHAR      = 'f' if FP_MODE else 'b'

## LoadGen test properties:
#
LOADGEN_SCENARIO            = os.getenv('CK_LOADGEN_SCENARIO', 'SingleStream')
LOADGEN_MODE                = os.getenv('CK_LOADGEN_MODE', 'AccuracyOnly')
LOADGEN_BUFFER_SIZE         = int(os.getenv('CK_LOADGEN_BUFFER_SIZE'))      # set to how many samples are you prepared to keep in memory at once
LOADGEN_DATASET_SIZE        = int(os.getenv('CK_LOADGEN_DATASET_SIZE'))     # set to how many total samples to choose from (0 = full set)
LOADGEN_CONF_FILE           = os.getenv('CK_LOADGEN_CONF_FILE', '')
LOADGEN_MULTISTREAMNESS     = os.getenv('CK_LOADGEN_MULTISTREAMNESS', '')   # if not set, use value from LoadGen's config file, or LoadGen code
LOADGEN_MAX_DURATION_S      = os.getenv('CK_LOADGEN_MAX_DURATION_S', '')    # if not set, use value from LoadGen's config file, or LoadGen code
LOADGEN_COUNT_OVERRIDE      = os.getenv('CK_LOADGEN_COUNT_OVERRIDE', '')
BATCH_SIZE                  = int(os.getenv('CK_BATCH_SIZE', '1'))

## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
LABELS_PATH             = os.environ['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_IMAGE_HEIGHT      = int(os.getenv('CK_ENV_ONNX_MODEL_IMAGE_HEIGHT', os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT', '')))
MODEL_IMAGE_WIDTH       = int(os.getenv('CK_ENV_ONNX_MODEL_IMAGE_WIDTH', os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH', '')))
MODEL_IMAGE_CHANNELS    = 3
MODEL_SOFTMAX_LAYER     = os.getenv('CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME', os.getenv('CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME', ''))


## Image normalization:
#
MODEL_NORMALIZE_DATA    = os.getenv('ML_MODEL_NORMALIZE_DATA') in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN           = os.getenv('ML_MODEL_SUBTRACT_MEAN', 'YES') in ('YES', 'yes', 'ON', 'on', '1')
GIVEN_CHANNEL_MEANS     = os.getenv('ML_MODEL_GIVEN_CHANNEL_MEANS', '')
if GIVEN_CHANNEL_MEANS:
    GIVEN_CHANNEL_MEANS = np.array(GIVEN_CHANNEL_MEANS.split(' '), dtype=np.float32)
    if MODEL_COLOURS_BGR:
        GIVEN_CHANNEL_MEANS = GIVEN_CHANNEL_MEANS[::-1]     # swapping Red and Blue colour channels

## Input image properties:
#
IMAGE_DIR               = os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR')
IMAGE_LIST_FILE         = os.path.join(IMAGE_DIR, os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF'))
IMAGE_DATA_TYPE         = np.dtype( os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DATA_TYPE', 'uint8') )

## Misc
#
VERBOSITY_LEVEL         = int(os.getenv('CK_VERBOSE', '0'))


# Load preprocessed image filepaths:
with open(IMAGE_LIST_FILE, 'r') as f:
    image_path_list = [ os.path.join(IMAGE_DIR, s.strip()) for s in f ]
LOADGEN_DATASET_SIZE = LOADGEN_DATASET_SIZE or len(image_path_list)


def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


def tick(letter, quantity=1):
    print(letter + (str(quantity) if quantity>1 else ''), end='')


# Currently loaded preprocessed images are stored in a dictionary:
preprocessed_image_buffer = {}


def load_query_samples(sample_indices):     # 0-based indices in our whole dataset
    global MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS
    global preprocessed_image_buffer

    print("load_query_samples({})".format(sample_indices))

    tick('B', len(sample_indices))

    for sample_index in sample_indices:
        img_filename = image_path_list[sample_index]
        img = np.fromfile(img_filename, IMAGE_DATA_TYPE)
        img = img.reshape((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS))
        if MODEL_COLOURS_BGR:
            img = img[...,::-1]     # swapping Red and Blue colour channels

        if IMAGE_DATA_TYPE != 'float32':
            img = img.astype(np.float32)

            # Normalize
            if MODEL_NORMALIZE_DATA:
                img = img/127.5 - 1.0

            # Subtract mean value
            if SUBTRACT_MEAN:
                if len(GIVEN_CHANNEL_MEANS):
                    img -= GIVEN_CHANNEL_MEANS
                else:
                    img -= np.mean(img)

        nhwc_img = img if MODEL_DATA_LAYOUT == 'NHWC' else img.transpose(2,0,1)

        preprocessed_image_buffer[sample_index] = np.array(nhwc_img).ravel().astype(TRANSFER_TYPE_NP)
        tick('l')
    print('')


def unload_query_samples(sample_indices):
    #print("unload_query_samples({})".format(sample_indices))
    preprocessed_image_buffer = {}
    tick('U')
    print('')


in_progress  = {}                   # local store of metadata about batches between issue_queries and send_responses
funnel_should_be_running = True     # a way for the fan to signal to the funnel_thread to end

def issue_queries(query_samples):

    global BATCH_SIZE

    if VERBOSITY_LEVEL:
        printable_query = [(qs.index, qs.id) for qs in query_samples]
        print("issue_queries( {} )".format(printable_query))
    tick('Q', len(query_samples))

    for j in range(0, len(query_samples), BATCH_SIZE):
        batch               = query_samples[j:j+BATCH_SIZE]   # NB: the last one may be shorter than BATCH_SIZE in length
        batch_vector_numpy  = np.ravel([ preprocessed_image_buffer[qs.index] for qs in batch ])

        job_id      = batch[0].id   # assume it is both sufficiently unique and sufficiently small to fit our needs

        in_progress[job_id] = {
            'submission_time':  time.time(),
            'batch':            batch,
        }
    
        if TRANSFER_MODE == 'numpy':
            job_data_struct = {
                'job_id': job_id,
                'batch_data': batch_vector_numpy,
            }
            to_workers.send_pyobj(job_data_struct)
        else:
            batch_vector_array  = batch_vector_numpy.tolist()
            if TRANSFER_MODE == 'raw':
                job_data_raw = struct.pack('<I{}{}'.format(len(batch_vector_array), TRANSFER_TYPE_CHAR), job_id, *batch_vector_array)
                to_workers.send(job_data_raw)
            else:
                job_data_struct = {
                    'job_id': job_id,
                    'batch_data': batch_vector_array,
                }
                if TRANSFER_MODE == 'json':
                    to_workers.send_json(job_data_struct)
                elif TRANSFER_MODE == 'pickle':
                    to_workers.send_pyobj(job_data_struct)

        print("[fan] -> job_id={} {}".format(job_id, [qs.index for qs in batch]))


def send_responses():

    global funnel_should_be_running

    funnel_start = time.time()
    inference_times_ms_by_worker_id = {}

    while funnel_should_be_running:

        try:
            done_job            = from_workers.recv_json()
        except Exception as e:
            continue    # go back and check if the funnel_should_be_running condition has been turned off by the main thread

        job_id              = done_job['job_id']
        local_metadata      = in_progress.pop(job_id)
        roundtrip_time_ms   = int((time.time()-local_metadata['submission_time'])*1000)
        batch               = local_metadata['batch']
        batch_size          = len(batch)
        worker_id           = done_job['worker_id']
        inference_time_ms   = done_job['inference_time_ms']
        raw_batch_results   = np.array(done_job['raw_batch_results'])
        batch_results       = np.split(raw_batch_results, batch_size)

        if worker_id not in inference_times_ms_by_worker_id:
            inference_times_ms_by_worker_id[worker_id] = []
        inference_times_ms_by_worker_id[worker_id].append( inference_time_ms )
        print("[funnel] <- [worker {}] job_id={}, inference={} ms, roundtrip={} ms".format(worker_id, job_id, inference_time_ms, roundtrip_time_ms))

        response = []
        response_array_refs = []    # This is needed to guarantee that the individual buffers to which we keep extra-Pythonian references, do not get garbage-collected.
        for qs, softmax_vector in zip(batch, batch_results):
            predicted_label = np.argmax( softmax_vector[-1000:] )

            response_array = array.array("B", np.array(predicted_label, np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(qs.id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)
        tick('R', len(response))
        sys.stdout.flush()
    print("[funnel] quitting")


def flush_queries():
    pass


def process_latencies(latencies_ns):
    latencies_ms = [ns/1.0e6 for ns in latencies_ns]
    print("LG called process_latencies({})".format(latencies_ms))

    latencies_size      = len(latencies_ms)
    latencies_avg       = sum(latencies_ms)/latencies_size
    latencies_sorted    = sorted(latencies_ms)
    latencies_p50       = int(latencies_size * 0.5);
    latencies_p90       = int(latencies_size * 0.9);

    print("--------------------------------------------------------------------")
    print("|                LATENCIES (in milliseconds and fps)               |")
    print("--------------------------------------------------------------------")
    print("Number of queries run:       {:9d}".format(latencies_size))
    print("Min latency:                 {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[0], 1e3/latencies_sorted[0]))
    print("Median latency:              {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[latencies_p50], 1e3/latencies_sorted[latencies_p50]))
    print("Average latency:             {:9.2f} ms   ({:.3f} fps)".format(latencies_avg, 1e3/latencies_avg))
    print("90 percentile latency:       {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[latencies_p90], 1e3/latencies_sorted[latencies_p90]))
    print("Max latency:                 {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[-1], 1e3/latencies_sorted[-1]))
    print("--------------------------------------------------------------------")


def benchmark_using_loadgen():
    "Perform the benchmark using python API for the LoadGen library"

    global funnel_should_be_running

    scenario = {
        'SingleStream':     lg.TestScenario.SingleStream,
        'MultiStream':      lg.TestScenario.MultiStream,
        'Server':           lg.TestScenario.Server,
        'Offline':          lg.TestScenario.Offline,
    }[LOADGEN_SCENARIO]

    mode = {
        'AccuracyOnly':     lg.TestMode.AccuracyOnly,
        'PerformanceOnly':  lg.TestMode.PerformanceOnly,
        'SubmissionRun':    lg.TestMode.SubmissionRun,
    }[LOADGEN_MODE]

    ts = lg.TestSettings()
    if LOADGEN_CONF_FILE:
        ts.FromConfig(LOADGEN_CONF_FILE, 'random_model_name', LOADGEN_SCENARIO)
    ts.scenario = scenario
    ts.mode     = mode

    if LOADGEN_MULTISTREAMNESS:
        ts.multi_stream_samples_per_query = int(LOADGEN_MULTISTREAMNESS)

    if LOADGEN_MAX_DURATION_S:
        ts.max_duration_ms = int(LOADGEN_MAX_DURATION_S)*1000

    if LOADGEN_COUNT_OVERRIDE:
        ts.min_query_count = int(LOADGEN_COUNT_OVERRIDE)
        ts.max_query_count = int(LOADGEN_COUNT_OVERRIDE)

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(LOADGEN_DATASET_SIZE, LOADGEN_BUFFER_SIZE, load_query_samples, unload_query_samples)

    log_settings = lg.LogSettings()
    log_settings.enable_trace = False

    funnel_thread = threading.Thread(target=send_responses, args=())
    funnel_should_be_running = True
    funnel_thread.start()

    lg.StartTestWithLogSettings(sut, qsl, ts, log_settings)

    funnel_should_be_running = False    # politely ask the funnel_thread to end
    funnel_thread.join()                # wait for it to actually end

    from_workers.close()
    to_workers.close()

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


benchmark_using_loadgen()