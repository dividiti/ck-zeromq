#!/bin/bash


function exit_if_error() {
    if [ "${?}" != "0" ]; then exit 1; fi
}


# Refresh CK-MLPerf and its dependencies.
echo "Refreshing CK-MLPerf ..."
ck pull repo:ck-mlperf
exit_if_error


echo


echo "Setting up SSD models and COCO ..."

# Skip SSD models setup: should be NO hub; should be NO for worker.
skip_ssd_setup=${CK_SKIP_SSD_SETUP:-"NO"}
echo "- skip SSD-MobileNet and SSD-ResNet setup (download): ${skip_ssd_setup}"

# Skip COCO setup: should be NO for hub; should be YES or NO for worker.
skip_coco_setup=${CK_SKIP_COCO_SETUP:-"NO"}
echo "- skip COCO setup (download and preprocessing): ${skip_coco_setup}"


echo


if [ "${skip_ssd_setup}" == "NO" ]; then
  # Install SSD models generated from NVIDIA's MLPerf Inference v0.5 submission.
  # TODO: Xavier only at the moment.
  ck install package --tags=model,tensorrt,downloaded,ssd-mobilenet
  ck install package --tags=model,tensorrt,downloaded,ssd-resnet
  exit_if_error
fi


if [ "${skip_coco_setup}" == "NO" ]; then
  # Detect OpenCV in its location in JetPack 4.3.
  # TODO: Only works on Jetson machines at the moment.
  ck detect soft --tags=python-package,cv2 --full_path=/usr/lib/python3.6/dist-packages/cv2/__init__.py
  exit_if_error

  # Install the COCO 2017 validation dataset (5,000 images, 1.6 GB with annotations).
  echo | ck install package --tags=object-detection,dataset,coco.2017,val,original,full
  exit_if_error

  # Remove training annotations (~765 MB), leaving only validation annotations (~52 MB).
  ck virtual env \
    --tags=object-detection,dataset,coco.2017,val,original,full \
    --shell_cmd='rm $CK_ENV_DATASET_COCO_LABELS_DIR/*train2017.json'

  # Preprocess for SSD-MobileNet (300x300 input images, 264 KB each, 1.3 GB in total).
  ck install package --tags=dataset,preprocessed,using-opencv,coco.2017,full,side.300
  exit_if_error

  # Preprocess for SSD-ResNet (1200x1200 input images, 4.2 MB each, 21 GB in total).
  ck install package --tags=dataset,preprocessed,using-opencv,coco.2017,full,side.1200
  exit_if_error
fi


echo


echo "Done."
