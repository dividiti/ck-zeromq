#!/usr/bin/env python3

import json
import os
import shutil
import struct
import threading
import time

import numpy as np
import zmq

try:
    raw_input
except NameError:
    # Python 3
    raw_input = input


## Post-detection filtering by confidence score:
#
SCORE_THRESHOLD = float(os.getenv('CK_DETECTION_THRESHOLD', 0.0))


## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
LABELS_PATH             = os.getenv('CK_ENV_TENSORRT_MODEL_FLATLABELS_FILE') or os.environ['ML_MODEL_CLASS_LABELS']
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_INPUT_DATA_TYPE   = os.getenv('ML_MODEL_INPUT_DATA_TYPE', 'float32')
MODEL_DATA_TYPE         = os.getenv('ML_MODEL_DATA_TYPE', '(unknown)')
MODEL_MAX_PREDICTIONS   = int(os.getenv('ML_MODEL_MAX_PREDICTIONS', 100))
MODEL_SKIPPED_CLASSES   = os.getenv("ML_MODEL_SKIPS_ORIGINAL_DATASET_CLASSES", None)
MODEL_IMAGE_HEIGHT      = int(os.getenv('ML_MODEL_IMAGE_HEIGHT',
                              os.getenv('CK_ENV_ONNX_MODEL_IMAGE_HEIGHT',
                              os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT',
                              ''))))
MODEL_IMAGE_WIDTH       = int(os.getenv('ML_MODEL_IMAGE_WIDTH',
                              os.getenv('CK_ENV_ONNX_MODEL_IMAGE_WIDTH',
                              os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH',
                              ''))))

if (MODEL_SKIPPED_CLASSES):
    SKIPPED_CLASSES = [int(x) for x in MODEL_SKIPPED_CLASSES.split(",")]
else:
    SKIPPED_CLASSES = None

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

## Image normalization:
#
MODEL_NORMALIZE_DATA    = os.getenv('ML_MODEL_NORMALIZE_DATA') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_NORMALIZE_LOWER   = float(os.getenv('ML_MODEL_NORMALIZE_LOWER', -1.0))
MODEL_NORMALIZE_UPPER   = float(os.getenv('ML_MODEL_NORMALIZE_UPPER',  1.0))

SUBTRACT_MEAN           = os.getenv('ML_MODEL_SUBTRACT_MEAN', 'YES') in ('YES', 'yes', 'ON', 'on', '1')
GIVEN_CHANNEL_MEANS     = os.getenv('ML_MODEL_GIVEN_CHANNEL_MEANS', '')
if GIVEN_CHANNEL_MEANS:
    GIVEN_CHANNEL_MEANS = np.fromstring(GIVEN_CHANNEL_MEANS, dtype=np.float32, sep=' ')
    if MODEL_COLOURS_BGR:
        GIVEN_CHANNEL_MEANS = GIVEN_CHANNEL_MEANS[::-1]     # swapping Red and Blue colour channels

GIVEN_CHANNEL_STDS      = os.getenv('ML_MODEL_GIVEN_CHANNEL_STDS', '')
if GIVEN_CHANNEL_STDS:
    GIVEN_CHANNEL_STDS = np.fromstring(GIVEN_CHANNEL_STDS, dtype=np.float32, sep=' ')
    if MODEL_COLOURS_BGR:
        GIVEN_CHANNEL_STDS  = GIVEN_CHANNEL_STDS[::-1]      # swapping Red and Blue colour channels

## Input image properties:
#
IMAGE_DIR               = os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DIR')
IMAGE_LIST_FILE_NAME    = os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_SUBSET_FOF')
IMAGE_LIST_FILE         = os.path.join(IMAGE_DIR, IMAGE_LIST_FILE_NAME)
IMAGE_DATA_TYPE         = os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DATA_TYPE', 'uint8')

## Writing the results out:
#
CUR_DIR = os.getcwd()
DETECTIONS_OUT_DIR      = os.path.join(CUR_DIR, os.environ['CK_DETECTIONS_OUT_DIR'])
ANNOTATIONS_OUT_DIR     = os.path.join(CUR_DIR, os.environ['CK_ANNOTATIONS_OUT_DIR'])
RESULTS_OUT_DIR         = os.path.join(CUR_DIR, os.environ['CK_RESULTS_OUT_DIR'])
FULL_REPORT             = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')

## Processing in batches:
#
BATCH_SIZE              = int(os.getenv('CK_BATCH_SIZE', 1))
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))
SKIP_IMAGES             = int(os.getenv('CK_SKIP_IMAGES', 0))


## ZeroMQ communication setup:
#
zmq_context = zmq.Context()

to_workers = zmq_context.socket(zmq.PUSH)
to_workers.bind("tcp://*:{}".format(ZMQ_FAN_PORT))

from_workers = zmq_context.socket(zmq.PULL)
from_workers.bind("tcp://*:{}".format(ZMQ_FUNNEL_PORT))


def load_preprocessed_batch(image_list, image_index):
    batch_data = []

    for _ in range(BATCH_SIZE):
        img_filepath = os.path.join(IMAGE_DIR, image_list[image_index])
        img = np.fromfile(img_filepath, np.dtype(IMAGE_DATA_TYPE))
        img = img.reshape((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3))
        if MODEL_COLOURS_BGR:
            img = img[...,::-1]     # swapping Red and Blue colour channels

        if IMAGE_DATA_TYPE != 'float32':
            img = img.astype(np.float32)

            # Normalize
            if MODEL_NORMALIZE_DATA:
                img = img*(MODEL_NORMALIZE_UPPER-MODEL_NORMALIZE_LOWER)/255.0+MODEL_NORMALIZE_LOWER

            # Subtract mean value
            if SUBTRACT_MEAN:
                if len(GIVEN_CHANNEL_MEANS):
                    img -= GIVEN_CHANNEL_MEANS
                else:
                    img -= np.mean(img, axis=(0,1), keepdims=True)

            if len(GIVEN_CHANNEL_STDS):
                img /= GIVEN_CHANNEL_STDS

        if MODEL_INPUT_DATA_TYPE == 'int8' or TRANSFER_TYPE_NP == np.int8:
            img = np.clip(img, -128, 127)

        # Add img to batch
        batch_data.append( [img.astype(TRANSFER_TYPE_NP)] )
        image_index += 1

    nhwc_data = np.concatenate(batch_data, axis=0)

    if MODEL_DATA_LAYOUT == 'NHWC':
        #print(nhwc_data.shape)
        return nhwc_data, image_index
    else:
        nchw_data = nhwc_data.transpose(0,3,1,2)
        #print(nchw_data.shape)
        return nchw_data, image_index


def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


print("Loading preprocessed image filenames...")
# Load preprocessed image filenames with original sizes:
with open(IMAGE_LIST_FILE, 'r') as f:
    image_list = [s.strip() for s in f]

# Trim the input list of preprocessed files:
image_list = image_list[SKIP_IMAGES: BATCH_COUNT * BATCH_SIZE + SKIP_IMAGES]

# Creating a local list of processed files and parsing it:
image_filenames = []
original_w_h    = []
with open(IMAGE_LIST_FILE_NAME, 'w') as f:
    for line in image_list:
        f.write('{}\n'.format(line))
        file_name, width, height = line.split(";")
        image_filenames.append( file_name )
        original_w_h.append( (int(width), int(height)) )


## (Shared) placeholders:
#
in_progress     = {}    # to be written to by one thread and read by another
output_dict     = {     # to be topped up by both threads
    'batch_size': BATCH_SIZE,
    'batch_count': BATCH_COUNT,
    'avg_inference_time_ms_by_worker_id': {},
}
output_dictionary = {   # object detection postprocessor prefers another level of nestedness
    'run_time_state': output_dict
}


def fan_code():

    print("Press Enter when the workers are ready: ")
    _ = raw_input()
    print("[fan] Submitting jobs...")

    fan_start = time.time()

    image_index = 0
    for batch_index in range(BATCH_COUNT):

        batch_first_index = image_index
        batch_data, image_index = load_preprocessed_batch(image_filenames, image_index)

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
    print("[fan] Done submitting batches. Submission took {} s".format(fan_time_s))

    output_dict['fan_time_s']               = fan_time_s
    output_dict['avg_send_batch_time_ms']   = fan_time_s*1000/BATCH_COUNT


def funnel_code():

    # Cleanup results directory
    if os.path.isdir(DETECTIONS_OUT_DIR):
        shutil.rmtree(DETECTIONS_OUT_DIR)
    os.mkdir(DETECTIONS_OUT_DIR)

    labels          = load_labels(LABELS_PATH)
    bg_class_offset = 1

    ## Workaround for SSD-Resnet34 model incorrectly trained on filtered labels
    class_map = None
    if (SKIPPED_CLASSES):
        class_map = []
        for i in range(len(labels) + bg_class_offset):
            if i not in SKIPPED_CLASSES:
                class_map.append(i)

    funnel_start = time.time()
    inference_times_ms_by_worker_id = {}

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
        raw_batch_results   = np.array(done_job['raw_batch_results'], dtype=np.float32)
        batch_results       = np.split(raw_batch_results, batch_size)

        if worker_id not in inference_times_ms_by_worker_id:
            inference_times_ms_by_worker_id[worker_id] = []
        inference_times_ms_by_worker_id[worker_id].append( inference_time_ms )

        for global_image_index, single_image_predictions in zip(batch_ids, batch_results):
            num_boxes = single_image_predictions[MODEL_MAX_PREDICTIONS*7].view('int32')
            width_orig, height_orig = original_w_h[global_image_index]

            filename_orig = image_filenames[global_image_index]
            detections_filename = os.path.splitext(filename_orig)[0] + '.txt'
            detections_filepath = os.path.join(DETECTIONS_OUT_DIR, detections_filename)

            with open(detections_filepath, 'w') as det_file:
                det_file.write('{:d} {:d}\n'.format(width_orig, height_orig))

                for row in range(num_boxes):
                    (image_id, ymin, xmin, ymax, xmax, confidence, class_number) = single_image_predictions[row*7:(row+1)*7]

                    if confidence >= SCORE_THRESHOLD:
                        class_number    = int(class_number)

                        if class_map:
                            class_number = class_map[class_number]

                        image_id        = int(image_id)
                        x1              = xmin * width_orig
                        y1              = ymin * height_orig
                        x2              = xmax * width_orig
                        y2              = ymax * height_orig
                        class_label     = labels[class_number - bg_class_offset]
                        det_file.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {} {}\n'.format(
                                        x1, y1, x2, y2, confidence, class_number, class_label))


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
    json.dump(output_dictionary, out_file, indent=4, sort_keys=True)

