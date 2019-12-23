#!/usr/bin/env python3

import json
import os
import shutil
import threading
import time

import numpy as np
import zmq

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


SLEEP_AFTER_SEND_MS     = int(os.getenv('CK_SLEEP_AFTER_SEND_MS', 0))

## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_DATA_TYPE         = os.getenv('ML_MODEL_DATA_TYPE', 'float32')
MODEL_SOFTMAX_LAYER     = os.getenv('CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME', os.getenv('CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME', ''))
MODEL_IMAGE_HEIGHT      = int(os.getenv('CK_ENV_ONNX_MODEL_IMAGE_HEIGHT', os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT', '')))
MODEL_IMAGE_WIDTH       = int(os.getenv('CK_ENV_ONNX_MODEL_IMAGE_WIDTH', os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH', '')))


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
IMAGE_DATA_TYPE         = os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DATA_TYPE', 'uint8')

## Writing the results out:
#
RESULTS_DIR             = os.getenv('CK_RESULTS_DIR', './results')
FULL_REPORT             = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')

## Processing in batches:
#
BATCH_SIZE              = int(os.getenv('CK_BATCH_SIZE', 1))
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))


def load_preprocessed_batch(image_list, image_index):
    batch_data = []
    for _ in range(BATCH_SIZE):
        img_file = os.path.join(IMAGE_DIR, image_list[image_index])
        img = np.fromfile(img_file, np.dtype(IMAGE_DATA_TYPE))
        img = img.reshape((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3))
        if MODEL_COLOURS_BGR:
            img = img[...,::-1]     # swapping Red and Blue colour channels

        if IMAGE_DATA_TYPE == 'uint8':
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

        # Add img to batch
        batch_data.append( [img] )
        image_index += 1

    nhwc_data = np.concatenate(batch_data, axis=0)

    if MODEL_DATA_LAYOUT == 'NHWC':
        #print(nhwc_data.shape)
        return nhwc_data, image_index
    else:
        nchw_data = nhwc_data.transpose(0,3,1,2)
        #print(nchw_data.shape)
        return nchw_data, image_index


print("Loading preprocessed image filenames...")
with open(IMAGE_LIST_FILE, 'r') as f:
    image_list = [ s.strip() for s in f ]



## (Shared) placeholders:
#
in_progress     = {}    # to be written to by one thread and read by another
output_dict     = {     # to be topped up by both threads
    'batch_size': BATCH_SIZE,
    'batch_count': BATCH_COUNT,
    'avg_inference_time_ms_by_worker_id': {},
}


def fan_code():

    print("Press Enter when the workers are ready: ")
    _ = raw_input()
    print("[fan] Submitting jobs...")

    fan_start = time.time()

    image_index = 0
    for batch_index in range(BATCH_COUNT):
        batch_number = batch_index+1
      
        batch_first_index = image_index
        batch_data, image_index = load_preprocessed_batch(image_list, image_index)

        batch_ids = list(range(batch_first_index, image_index))
        submitted_job = {
            'job_id': batch_number,
            'batch_data': batch_data.ravel().tolist(),
            'batch_ids': batch_ids,
        }

        in_progress[batch_number] = {
            'submission_time':  time.time(),
            'batch_ids':        batch_ids,
        }
        to_workers.send_json(submitted_job)
        print("[fan] -> job number {} {}".format(batch_number, batch_ids))

        time.sleep(SLEEP_AFTER_SEND_MS/1000)  # do not overflow the ZeroMQ

    fan_time_s = time.time()-fan_start-SLEEP_AFTER_SEND_MS/1000
    print("[fan] Done submitting batches. Submission took {} s".format(fan_time_s))

    output_dict['fan_time_s']               = fan_time_s
    output_dict['avg_send_batch_time_ms']   = fan_time_s*1000/BATCH_COUNT


def funnel_code():

    # Cleanup results directory
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)

    funnel_start = time.time()
    inference_times_ms_by_worker_id = {}

    for _ in range(BATCH_COUNT):
        done_job = from_workers.recv_json()

        local_metadata      = in_progress[done_job['job_id']]
        roundtrip_time_ms   = int((time.time()-local_metadata['submission_time'])*1000)
        batch_ids           = local_metadata['batch_ids']
        batch_results       = done_job['batch_results']
        worker_id           = done_job['worker_id']
        inference_time_ms   = done_job['inference_time_ms']
        if worker_id not in inference_times_ms_by_worker_id:
            inference_times_ms_by_worker_id[worker_id] = []
        inference_times_ms_by_worker_id[worker_id].append( inference_time_ms )
        print("[funnel] <- [worker {}] {}, inference={} ms, roundtrip={} ms".format(worker_id, batch_ids, inference_time_ms, roundtrip_time_ms))

        for sample_id in batch_results:
            softmax_vector = batch_results[sample_id][-1000:]    # skipping the background class on the left (if present)
            res_file = os.path.join(RESULTS_DIR, image_list[int(sample_id)])
            with open(res_file + '.txt', 'w') as f:
                for prob in softmax_vector:
                    f.write('{}\n'.format(prob))

    funnel_time_s = time.time()-funnel_start
    print("[funnel] Done receiving batches. Receiving took {} s".format(funnel_time_s))

    for worker_id in inference_times_ms_by_worker_id:
        offset = 1 if len(inference_times_ms_by_worker_id[worker_id]) > 1 else 0    # skip the potential cold startup in case there is more data
        avg_inference_time_ms_by_worker_id = np.mean(inference_times_ms_by_worker_id[worker_id][offset:])
        output_dict['avg_inference_time_ms_by_worker_id'][worker_id] = avg_inference_time_ms_by_worker_id
        print("[funnel] Average batch inference time on [worker {}] is {}".format(worker_id, avg_inference_time_ms_by_worker_id))

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

