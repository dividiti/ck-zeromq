#!/usr/bin/env python3

import array
import json
import os
import struct
import sys
import threading
import time

import numpy as np
import zmq
import mlperf_loadgen as lg


###########################################################################################################
## NB: if you run into "zmq.error.ZMQError: Address already in use" after a crash,
##     run "sudo netstat -ltnp | grep python" and kill the socket-hogging process.
###########################################################################################################


## ZMQ ports:
#
ZMQ_FAN_PORT            = os.getenv('CK_ZMQ_FAN_PORT', 5557)
ZMQ_FUNNEL_PORT         = os.getenv('CK_ZMQ_FUNNEL_PORT', 5558)

## LoadGen test properties:
#
LOADGEN_SCENARIO            = os.getenv('CK_LOADGEN_SCENARIO', 'SingleStream')
LOADGEN_MODE                = os.getenv('CK_LOADGEN_MODE', 'AccuracyOnly')
LOADGEN_BUFFER_SIZE         = int(os.getenv('CK_LOADGEN_BUFFER_SIZE'))      # set to how many samples are you prepared to keep in memory at once
LOADGEN_DATASET_SIZE        = int(os.getenv('CK_LOADGEN_DATASET_SIZE'))     # set to how many total samples to choose from (0 = full set)
LOADGEN_CONFIG_FILE         = os.getenv('CK_ENV_LOADGEN_CONFIG_FILE', '')   # Very Important: make sure 'pass_env_to_resolve' is on
LOADGEN_MULTISTREAMNESS     = os.getenv('CK_LOADGEN_MULTISTREAMNESS', '')   # if not set, use value from LoadGen's config file, or LoadGen code
LOADGEN_MAX_DURATION_S      = os.getenv('CK_LOADGEN_MAX_DURATION_S', '')    # if not set, use value from LoadGen's config file, or LoadGen code
LOADGEN_COUNT_OVERRIDE      = os.getenv('CK_LOADGEN_COUNT_OVERRIDE', '')
LOADGEN_TARGET_QPS          = os.getenv('CK_LOADGEN_TARGET_QPS', '')        # Maps to differently named internal config options, depending on scenario - see below.
BATCH_SIZE                  = int(os.getenv('CK_BATCH_SIZE', '1'))
LOADGEN_WARM_UP_SAMPLES     = int(os.getenv('CK_LOADGEN_WARM_UP_SAMPLES', '0'))
SIDELOAD_JSON               = os.getenv('CK_LOADGEN_SIDELOAD_JSON','')

## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
LABELS_PATH             = os.environ['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_INPUT_DATA_TYPE   = os.getenv('ML_MODEL_INPUT_DATA_TYPE', 'float32')
MODEL_DATA_TYPE         = os.getenv('ML_MODEL_DATA_TYPE', '(unknown)')
MODEL_IMAGE_HEIGHT      = int(os.getenv('ML_MODEL_MODEL_IMAGE_HEIGHT',
                              os.getenv('CK_ENV_ONNX_MODEL_IMAGE_HEIGHT',
                              os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT',
                              ''))))
MODEL_IMAGE_WIDTH       = int(os.getenv('ML_MODEL_MODEL_IMAGE_WIDTH',
                              os.getenv('CK_ENV_ONNX_MODEL_IMAGE_WIDTH',
                              os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH',
                              ''))))
MODEL_IMAGE_CHANNELS    = 3
MODEL_SOFTMAX_LAYER     = os.getenv('CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME', os.getenv('CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME', ''))


## Data transfer:
#
TRANSFER_MODE           = os.getenv('CK_ZMQ_TRANSFER_MODE', 'json')
FP_MODE                 = (os.getenv('CK_FP_MODE', 'NO') in ('YES', 'yes', 'ON', 'on', '1')) and (MODEL_INPUT_DATA_TYPE == 'float32')
TRANSFER_TYPE_NP, TRANSFER_TYPE_CHAR = (np.float32, 'f') if FP_MODE else (np.int8, 'b')

## Internal processing:
#
VECTOR_DATA_TYPE        = np.float32

## Image normalization:
#
MODEL_NORMALIZE_DATA    = os.getenv('ML_MODEL_NORMALIZE_DATA') in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN           = os.getenv('ML_MODEL_SUBTRACT_MEAN', 'YES') in ('YES', 'yes', 'ON', 'on', '1')
GIVEN_CHANNEL_MEANS     = os.getenv('ML_MODEL_GIVEN_CHANNEL_MEANS', '')
if GIVEN_CHANNEL_MEANS:
    GIVEN_CHANNEL_MEANS = np.array(GIVEN_CHANNEL_MEANS.split(' '), dtype=VECTOR_DATA_TYPE)
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


## ZeroMQ communication setup:
#
zmq_context = zmq.Context()

to_workers = zmq_context.socket(zmq.PUSH)
to_workers.bind("tcp://*:{}".format(ZMQ_FAN_PORT))

from_workers = zmq_context.socket(zmq.PULL)
from_workers.bind("tcp://*:{}".format(ZMQ_FUNNEL_PORT))
from_workers.RCVTIMEO = 2000


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
            img = img.astype(VECTOR_DATA_TYPE)

            # Normalize
            if MODEL_NORMALIZE_DATA:
                img = img/127.5 - 1.0

            # Subtract mean value
            if SUBTRACT_MEAN:
                if len(GIVEN_CHANNEL_MEANS):
                    img -= GIVEN_CHANNEL_MEANS
                else:
                    img -= np.mean(img, axis=(0,1), keepdims=True)

        if MODEL_INPUT_DATA_TYPE == 'int8':
            img = np.clip(img, -128, 127)

        nhwc_img = img if MODEL_DATA_LAYOUT == 'NHWC' else img.transpose(2,0,1)

        preprocessed_image_buffer[sample_index] = np.array(nhwc_img).ravel().astype(TRANSFER_TYPE_NP)
        tick('l')
    print('')


def unload_query_samples(sample_indices):
    #print("unload_query_samples({})".format(sample_indices))
    preprocessed_image_buffer = {}
    tick('U')
    print('')


openme_data  = {}                   # side-loaded stats
in_progress  = {}                   # local store of metadata about batches between issue_queries and send_responses
funnel_should_be_running = True     # a way for the fan to signal to the funnel_thread to end
warm_up_mode             = False    # while on, QuerySampleResponses will not be sent to LoadGen

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

    global funnel_should_be_running, warm_up_mode, openme_data

    funnel_start = time.time()

    received_job_timings = openme_data['received_job_timings'] = []
    inference_times_ms_by_worker_id = {}

    while funnel_should_be_running:

        try:
            done_job            = from_workers.recv_json()
        except Exception as e:
            continue    # go back and check if the funnel_should_be_running condition has been turned off by the main thread

        job_id              = done_job['job_id']
        local_metadata      = in_progress.pop(job_id)
        received_timestamp  = time.time()
        roundtrip_time_ms   = (received_timestamp-local_metadata['submission_time'])*1000
        worker_id           = done_job['worker_id']
        inference_time_ms   = done_job['inference_time_ms']
        floatize_time_ms    = done_job['floatize_time_ms']

        print("[funnel] <- [worker {}] job_id={}, worker_type_conversion={:.2f} ms, inference={:.2f} ms, roundtrip={:.2f} ms".format(
                            worker_id, job_id, floatize_time_ms, inference_time_ms, roundtrip_time_ms))

        received_job_timings.append({
            'job_id':                   job_id,
            'worker_id':                worker_id,
            'received_timestamp':       received_timestamp,
            'worker_floatize_time_ms':  floatize_time_ms,
            'inference_time_ms':        inference_time_ms,
            'roundtrip_time_ms':        roundtrip_time_ms,
        })

        if warm_up_mode:
            continue

        if worker_id not in inference_times_ms_by_worker_id:
            inference_times_ms_by_worker_id[worker_id] = []
        inference_times_ms_by_worker_id[worker_id].append( inference_time_ms )

        batch               = local_metadata['batch']
        batch_size          = len(batch)
        raw_batch_results   = np.array(done_job['raw_batch_results'])
        batch_results       = np.split(raw_batch_results, batch_size)

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

    global openme_data

    latencies_ms = openme_data['loadgen_measured_latencies_ms'] = [ns/1.0e6 for ns in latencies_ns]
    print("LG called process_latencies({})".format(latencies_ms))

    latencies_size      = len(latencies_ms)
    latencies_avg       = sum(latencies_ms)/latencies_size
    latencies_sorted    = sorted(latencies_ms)
    latencies_p50       = int(latencies_size * 0.5);
    latencies_p90       = int(latencies_size * 0.9);
    latencies_p99       = int(latencies_size * 0.99);

    print("--------------------------------------------------------------------")
    print("|                LATENCIES (in milliseconds and fps)               |")
    print("--------------------------------------------------------------------")
    print("Number of samples run:       {:9d}".format(latencies_size))
    print("Min latency:                 {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[0], 1e3/latencies_sorted[0]))
    print("Median latency:              {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[latencies_p50], 1e3/latencies_sorted[latencies_p50]))
    print("Average latency:             {:9.2f} ms   ({:.3f} fps)".format(latencies_avg, 1e3/latencies_avg))
    print("90 percentile latency:       {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[latencies_p90], 1e3/latencies_sorted[latencies_p90]))
    print("99 percentile latency:       {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[latencies_p99], 1e3/latencies_sorted[latencies_p99]))
    print("Max latency:                 {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[-1], 1e3/latencies_sorted[-1]))
    print("--------------------------------------------------------------------")


def benchmark_using_loadgen():
    "Perform the benchmark using python API for the LoadGen library"

    global funnel_should_be_running, warm_up_mode, openme_data

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
    if LOADGEN_CONFIG_FILE:
        ts.FromConfig(LOADGEN_CONFIG_FILE, 'random_model_name', LOADGEN_SCENARIO)
    ts.scenario = scenario
    ts.mode     = mode

    if LOADGEN_MULTISTREAMNESS:
        ts.multi_stream_samples_per_query = int(LOADGEN_MULTISTREAMNESS)

    if LOADGEN_MAX_DURATION_S:
        ts.max_duration_ms = int(LOADGEN_MAX_DURATION_S)*1000

    if LOADGEN_COUNT_OVERRIDE:
        ts.min_query_count = int(LOADGEN_COUNT_OVERRIDE)
        ts.max_query_count = int(LOADGEN_COUNT_OVERRIDE)

    if LOADGEN_TARGET_QPS:
        target_qps                  = float(LOADGEN_TARGET_QPS)
        ts.multi_stream_target_qps  = target_qps
        ts.server_target_qps        = target_qps
        ts.offline_expected_qps     = target_qps

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(LOADGEN_DATASET_SIZE, LOADGEN_BUFFER_SIZE, load_query_samples, unload_query_samples)

    log_settings = lg.LogSettings()
    log_settings.enable_trace = False

    funnel_thread = threading.Thread(target=send_responses, args=())
    funnel_should_be_running = True
    funnel_thread.start()

    if LOADGEN_WARM_UP_SAMPLES:
        warm_up_id_range = list(range(LOADGEN_WARM_UP_SAMPLES))
        load_query_samples(warm_up_id_range)

        warm_up_mode = True
        print("Sending out the warm-up samples, waiting for responses...")
        issue_queries([lg.QuerySample(id,id) for id in warm_up_id_range])

        while len(in_progress)>0:       # waiting for the in_progress queue to clear up
            time.sleep(1)
        print(" Done!")

        warm_up_mode = False

    lg.StartTestWithLogSettings(sut, qsl, ts, log_settings)

    funnel_should_be_running = False    # politely ask the funnel_thread to end
    funnel_thread.join()                # wait for it to actually end

    from_workers.close()
    to_workers.close()

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    if SIDELOAD_JSON:
        with open(SIDELOAD_JSON, 'w') as sideload_fd:
            json.dump(openme_data, sideload_fd, indent=4, sort_keys=True)


benchmark_using_loadgen()
