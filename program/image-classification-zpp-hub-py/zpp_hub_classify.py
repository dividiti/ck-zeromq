#!/usr/bin/env python3

import json
import os
import shutil
import struct
import threading
import time

from imagenet_helper import (load_preprocessed_batch, image_list, class_labels,
    MODEL_DATA_LAYOUT, MODEL_COLOURS_BGR, MODEL_INPUT_DATA_TYPE, MODEL_DATA_TYPE, MODEL_USE_DLA, MODEL_MAX_BATCH_SIZE,
    IMAGE_DIR, IMAGE_LIST_FILE, MODEL_NORMALIZE_DATA, SUBTRACT_MEAN, GIVEN_CHANNEL_MEANS, BATCH_SIZE)

import numpy as np
import zmq

try:
    raw_input
except NameError:
    # Python 3
    raw_input = input


import sys
try:
    sys.getwindowsversion()
except AttributeError:
    win = False
else:
    win = True

if win:
    import win32api,win32process,win32con
    pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    # https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass
    print("Setting REALTIME_PRIORITY_CLASS on Windows ...")
    win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)


## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']


## Transfer mode (numpy floats by default):
#
TRANSFER_MODE           = os.getenv('CK_TRANSFER_MODE', 'numpy')
TRANSFER_FLOAT          = (os.getenv('CK_TRANSFER_FLOAT', 'YES') in ('YES', 'yes', 'ON', 'on', '1')) and (MODEL_INPUT_DATA_TYPE == 'float32')
TRANSFER_TYPE_NP, TRANSFER_TYPE_SYMBOL = (np.float32, 'f') if TRANSFER_FLOAT else (np.int8, 'b')

SLEEP_AFTER_SEND_MS     = int(os.getenv('CK_SLEEP_AFTER_SEND_MS', 0))

## ZMQ ports:
#
ZMQ_FAN_PORT            = os.getenv('CK_ZMQ_FAN_PORT', 5557)
ZMQ_FUNNEL_PORT         = os.getenv('CK_ZMQ_FUNNEL_PORT', 5558)


## Writing the results out:
#
RESULTS_DIR             = os.getenv('CK_RESULTS_DIR', './results')
FULL_REPORT             = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')


## Processing in batches:
#
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))


## ZeroMQ communication setup:
#
zmq_context = zmq.Context()

to_workers = zmq_context.socket(zmq.PUSH)
to_workers.bind("tcp://*:{}".format(ZMQ_FAN_PORT))

from_workers = zmq_context.socket(zmq.PULL)
from_workers.bind("tcp://*:{}".format(ZMQ_FUNNEL_PORT))


## (Shared) placeholders:
#
in_progress     = {}    # to be written to by one thread and read by another
output_dict     = {     # to be topped up by both threads
    'batch_size': BATCH_SIZE,
    'batch_count': BATCH_COUNT,
    'avg_inference_time_ms_by_worker_id': {},
    'avg_roundtrip_time_ms_by_worker_id': {},
    'min_roundtrip_time_ms_by_worker_id': {},
    'pc50_roundtrip_time_ms_by_worker_id': {},
    'pc90_roundtrip_time_ms_by_worker_id': {},
    'pc99_roundtrip_time_ms_by_worker_id': {},
    'max_roundtrip_time_ms_by_worker_id': {},
}


def fan_code():

    print("Press Enter when the workers are ready: ")
    _ = raw_input()
    print("[fan] Submitting jobs...")

    fan_start = time.time()

    image_index = 0
    for batch_index in range(BATCH_COUNT):

        batch_first_index = image_index
        batch_data, image_index = load_preprocessed_batch(image_list, image_index)

        batch_vector_numpy  = batch_data.ravel()

        batch_ids   = list(range(batch_first_index, image_index))
        job_id      = batch_index+1

        in_progress[job_id] = {
            'submission_time':  time.time(),
            'batch_ids':        batch_ids,
        }

        if TRANSFER_MODE == 'dummy':
            job_data_raw = struct.pack('<II', job_id, BATCH_SIZE)
            to_workers.send(job_data_raw)
        elif TRANSFER_MODE == 'numpy':
            job_data_struct = {
                'job_id': job_id,
                'batch_data': batch_vector_numpy,
            }
            to_workers.send_pyobj(job_data_struct)
        elif TRANSFER_MODE == 'pickle':
            job_data_struct = {
                'job_id': job_id,
                'batch_data': np.asarray(batch_vector_numpy),
            }
            to_workers.send_pyobj(job_data_struct)
        else:
            batch_vector_array  = batch_vector_numpy.tolist()
            if TRANSFER_MODE == 'raw':
                job_data_raw = struct.pack('<I{}{}'.format(len(batch_vector_array), TRANSFER_TYPE_SYMBOL), job_id, *batch_vector_array)
                to_workers.send(job_data_raw)
            elif TRANSFER_MODE == 'json':
                job_data_struct = {
                    'job_id': job_id,
                    'batch_data': batch_vector_array,
                }
                to_workers.send_json(job_data_struct)

        print("[fan] -> job_id={} {}".format(job_id, batch_ids))

        time.sleep(SLEEP_AFTER_SEND_MS/1000)  # do not overflow the ZeroMQ

    fan_time_s = time.time()-fan_start-SLEEP_AFTER_SEND_MS/1000
    print("[fan] Done submitting batches. Submission took {:.2f} s".format(fan_time_s))

    output_dict['fan_time_s']               = fan_time_s
    output_dict['avg_send_batch_time_ms']   = fan_time_s*1000/BATCH_COUNT


def funnel_code():

    # Cleanup results directory
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)

    funnel_start = time.time()
    inference_times_ms_by_worker_id = {}
    roundtrip_times_ms_by_worker_id = {}

    for _ in range(BATCH_COUNT):
        done_job = from_workers.recv_json()

        job_id              = done_job['job_id']
        local_metadata      = in_progress.pop(job_id)
        roundtrip_time_ms   = (time.time()-local_metadata['submission_time'])*1000
        worker_id           = done_job['worker_id']
        inference_time_ms   = done_job['inference_time_ms']
        floatize_time_ms    = done_job['floatize_time_ms']

        print("[funnel] <- [worker {}] job_id={}, worker_type_conversion={:.2f} ms, inference={:.2f} ms, roundtrip={:.2f} ms".format(
                            worker_id, job_id, floatize_time_ms, inference_time_ms, roundtrip_time_ms))

        batch_ids           = local_metadata['batch_ids']
        batch_size          = len(batch_ids)
        apparent_batch_size = MODEL_MAX_BATCH_SIZE if MODEL_USE_DLA else batch_size
        raw_batch_results   = np.array(done_job['raw_batch_results'])
        batch_results       = np.split(raw_batch_results, apparent_batch_size)[:batch_size]

        if worker_id not in inference_times_ms_by_worker_id:
            inference_times_ms_by_worker_id[worker_id] = []
        inference_times_ms_by_worker_id[worker_id].append( inference_time_ms )

        if worker_id not in roundtrip_times_ms_by_worker_id:
            roundtrip_times_ms_by_worker_id[worker_id] = []
        roundtrip_times_ms_by_worker_id[worker_id].append( roundtrip_time_ms )

        for sample_id, prediction_for_one_sample in zip(batch_ids, batch_results):
            if len(prediction_for_one_sample)==1:
                predicted_label = int(prediction_for_one_sample[0])
                trimmed_softmax_vector = [0]*predicted_label + [1] + [0]*(1000-predicted_label-1)
            else:
                trimmed_softmax_vector = prediction_for_one_sample[-1000:]    # skipping the background class on the left (if present)

            res_file = os.path.join(RESULTS_DIR, image_list[int(sample_id)])
            with open(res_file + '.txt', 'w') as f:
                for prob in trimmed_softmax_vector:
                    f.write('{}\n'.format(prob))

    funnel_time_s = time.time()-funnel_start
    print("[funnel] Done receiving batches. Receiving took {:.2f} s".format(funnel_time_s))

    print("")

    print("[funnel] Batch inference time (ms):")
    for worker_id in inference_times_ms_by_worker_id:
        offset = 1 if len(inference_times_ms_by_worker_id[worker_id]) > 1 else 0    # skip the potential cold startup in case there is more data
        avg_inference_time_ms_by_worker_id = np.mean(inference_times_ms_by_worker_id[worker_id][offset:])
        output_dict['avg_inference_time_ms_by_worker_id'][worker_id] = avg_inference_time_ms_by_worker_id
        print("- [worker {}] average: {:.2f}".format(worker_id, avg_inference_time_ms_by_worker_id))

    print("")

    print("[funnel] Batch roundtrip time (ms):")
    for worker_id in roundtrip_times_ms_by_worker_id:
        offset = 1 if len(roundtrip_times_ms_by_worker_id[worker_id]) > 1 else 0    # skip the potential cold startup in case there is more data

        avg_roundtrip_time_ms_by_worker_id = np.mean(roundtrip_times_ms_by_worker_id[worker_id][offset:])
        output_dict['avg_roundtrip_time_ms_by_worker_id'][worker_id] = avg_roundtrip_time_ms_by_worker_id
        print("- [worker {}] average: {:.2f}".format(worker_id, avg_roundtrip_time_ms_by_worker_id))

        min_roundtrip_time_ms_by_worker_id = np.min(roundtrip_times_ms_by_worker_id[worker_id][offset:])
        output_dict['min_roundtrip_time_ms_by_worker_id'][worker_id] = min_roundtrip_time_ms_by_worker_id
        print("- [worker {}] minimum: {:.2f}".format(worker_id, min_roundtrip_time_ms_by_worker_id))

        pc50_roundtrip_time_ms_by_worker_id = np.percentile(roundtrip_times_ms_by_worker_id[worker_id][offset:], 50)
        output_dict['pc50_roundtrip_time_ms_by_worker_id'][worker_id] = pc50_roundtrip_time_ms_by_worker_id
        print("- [worker {}] 50 percentile: {:.2f}".format(worker_id, pc50_roundtrip_time_ms_by_worker_id))

        pc90_roundtrip_time_ms_by_worker_id = np.percentile(roundtrip_times_ms_by_worker_id[worker_id][offset:], 90)
        output_dict['pc90_roundtrip_time_ms_by_worker_id'][worker_id] = pc90_roundtrip_time_ms_by_worker_id
        print("- [worker {}] 90 percentile: {:.2f}".format(worker_id, pc90_roundtrip_time_ms_by_worker_id))

        pc99_roundtrip_time_ms_by_worker_id = np.percentile(roundtrip_times_ms_by_worker_id[worker_id][offset:], 99)
        output_dict['pc99_roundtrip_time_ms_by_worker_id'][worker_id] = pc99_roundtrip_time_ms_by_worker_id
        print("- [worker {}] 99 percentile: {:.2f}".format(worker_id, pc99_roundtrip_time_ms_by_worker_id))

        max_roundtrip_time_ms_by_worker_id = np.max(roundtrip_times_ms_by_worker_id[worker_id][offset:])
        output_dict['max_roundtrip_time_ms_by_worker_id'][worker_id] = max_roundtrip_time_ms_by_worker_id
        print("- [worker {}] maximum: {:.2f}".format(worker_id, max_roundtrip_time_ms_by_worker_id))

        print("")

    output_dict['funnel_time_s']        = funnel_time_s
    output_dict['avg_rountrip_time_ms'] = funnel_time_s*1000/BATCH_COUNT


## We need one thread to feed the ZeroMQ, another (the main one) to read back from it:
#
fan_thread = threading.Thread(target=fan_code, args=())
fan_thread.start()

funnel_code()

fan_thread.join()


## Store benchmarking results:
#
with open('tmp-ck-timer.json', 'w') as out_file:
    json.dump(output_dict, out_file, indent=4, sort_keys=True)

