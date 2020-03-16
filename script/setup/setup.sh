#!/bin/bash

function exit_if_error() {
    if [ "${?}" != "0" ]; then exit 1; fi
}


#
echo "Setting up CK-ZeroMQ hub ..."
echo


# Refresh CK-ZeroMQ and its dependencies.
echo "Refreshing CK-ZeroMQ ..."
ck pull repo:ck-zeromq --url=https://github.com/dividiti/ck-zeromq
exit_if_error

## Set up Python, NumPy, PyZMQ.
#echo "Setting up Python and essential packages ..."
#
#ck detect soft:compiler.python --full_path=`which python3.6`
#exit_if_error
#
#ck install package --tags=python-package,numpy
#exit_if_error
#
#ck install package --tags=python-package,zmq
#exit_if_error

# Set up additional packages.
ck install package --tags=python-package,absl
exit_if_error

## Install ImageNet labels.
#ck install package --tags=dataset,imagenet,aux
#exit_if_error

## Detect ImageNet validation dataset (50,000 images) preprocessed using OpenCV. 	
#echo "Detecting the ImageNet validation set preprocessed using OpenCV ..."
#ck detect soft:dataset.imagenet.preprocessed --cus.version=using-opencv \
#--full_path=/datasets/dataset-imagenet-preprocessed-using-opencv-crop.875-full-inter.linear-side.224/ILSVRC2012_val_00000001.rgb8 \
#--extra_tags=preprocessed,using-opencv,universal,crop.875,full,inter.linear,side.224

# Install MLPerf Inference packages.
echo "Setting up MLPerf Inference packages ..."

ck install package --tags=mlperf,inference,source
exit_if_error

ck install package --tags=mlperf,loadgen,python-package
exit_if_error

echo
echo "Done."
