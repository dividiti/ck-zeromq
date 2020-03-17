#!/bin/bash

function exit_if_error() {
    if [ "${?}" != "0" ]; then exit 1; fi
}

# Refresh CK-ZeroMQ and its dependencies.
echo "Refreshing CK-ZeroMQ ..."
ck pull repo:ck-zeromq --url=https://github.com/dividiti/ck-zeromq
exit_if_error

echo

#
echo "Setting up CK-ZeroMQ ..."

# Skip Python setup: should be false for hub/worker.
skip_python_setup=${CK_SKIP_PYTHON_SETUP:-""}
echo "- skip Python setup: ${skip_python_setup}"

# Skip NVIDIA setup: should be false for hub/worker.
skip_nvidia_setup=${CK_SKIP_NVIDIA_SETUP:-""}
echo "- skip NVIDIA setup: ${skip_nvidia_setup}"

# Skip LoadGen setup: should be false for hub and true for worker.
skip_loadgen_setup=${CK_SKIP_LOADGEN_SETUP:-""}
echo "- skip LoadGen setup: ${skip_loadgen_setup}"

# Skip ImageNet detection: should be false for hub and true for worker.
skip_imagenet_detection=${CK_SKIP_IMAGENET_DETECTION:-""}
echo "- skip ImageNet detection: ${skip_imagenet_detection}"

# Fake ResNet detection: can be true for hub and must be false for worker.
fake_resnet_detection=${CK_FAKE_RESNET_DETECTION:-NO}
ck_tools=${CK_TOOLS:-"$HOME/CK-TOOLS"}
echo "- fake ResNet detection: ${fake_resnet_detection} (CK_TOOLS=${ck_tools})"

echo


if [ -z "${skip_python_setup}" ]; then
  # Set up Python, NumPy, PyZMQ.
  echo "Setting up Python 3 and essential packages ..."
  ck detect soft:compiler.python --full_path=`which python3`
  exit_if_error

  ck install package --tags=python-package,cython
  exit_if_error

  # NB: Building NumPy 1.18.1 requires Cython >= 0.29.14
  ck virtual env --tags=cython --shell_cmd='ck install package --tags=python-package,numpy'
  exit_if_error

  ck install package --tags=python-package,zmq
  exit_if_error
fi


if [ -z "${skip_nvidia_setup}" ]; then
  # Detect TensorRT and PyTensorRT.
  echo "Setting up TensorRT/PyTensorRT ..."

  ck detect soft:lib.tensorrt --full_path=/usr/lib/aarch64-linux-gnu/libnvinfer.so
  exit_if_error

  ck detect soft:lib.python.tensorrt --full_path=/usr/lib/python3.6/dist-packages/tensorrt/__init__.py
  exit_if_error

  # Detect GCC/CUDA and install PyCUDA.
  echo "Setting up CUDA/PyCUDA ..."

  ck detect soft:compiler.gcc --full_path=`which gcc-7`
  exit_if_error

  ck detect soft:compiler.cuda --full_path=/usr/local/cuda-10.0/bin/nvcc
  exit_if_error

  ck install package --tags=python-package,pycuda
  exit_if_error

  # Install the official MLPerf ONNX model and convert it to TensorRT with predefined options.
  ck install package --tags=image-classification,model,onnx,resnet
  ck install package --tags=image-classification,model,tensorrt,resnet,converted-from-onnx,maxbatch.20,fp16
  exit_if_error
fi


if [ -z "${skip_loadgen_setup}" ]; then
  # Install MLPerf Inference packages.
  echo "Setting up MLPerf Inference packages ..."

  ck install package --tags=mlperf,inference,source
  exit_if_error

  ck install package --tags=python-package,absl
  exit_if_error

  ck install package --tags=mlperf,loadgen,python-package
  exit_if_error
fi


if [ -z "${skip_imagenet_detection}" ]; then
  # Detect a preprocessed ImageNet validation dataset (50,000 images).
  echo "Detecting a preprocessed ImageNet validation set ..."
  imagenet_dir=${CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR:-"/datasets/dataset-imagenet-preprocessed-using-opencv-crop.875-full-inter.linear-side.224/ILSVRC2012_val_00000001.rgb8"}
  imagenet_tags=${CK_ENV_DATASET_IMAGENET_PREPROCESSED_TAGS:-"preprocessed,using-opencv,universal,crop.875,full,inter.linear,side.224"}
  imagenet_version=${CK_ENV_DATASET_IMAGENET_PREPROCESSED_VERSION:-"using-opencv"}
  ck detect soft:dataset.imagenet.preprocessed --full_path=${imagenet_dir} --extra_tags=${imagenet_tags} --cus.version=${imagenet_version}

  # Install ImageNet labels.
  ck install package --tags=dataset,imagenet,aux
  exit_if_error
fi


if [ "${fake_resnet_detection}" != "NO" ]; then
  # Detect fake ResNet model.
  model_dir=${ck_tools}/model-tensorrt-converted-from-onnx-fp16-maxbatch.20-resnet
  model_file=${model_dir}/converted_model.trt
  mkdir -p ${model_dir}
  touch ${model_file}
  ck detect soft:model.tensorrt --cus.version=resnet50-fp16 \
  --full_path=${model_file} \
  --extra_tags=converted,converted-from-onnx,fp16,image-classification,maxbatch.20,model,resnet,tensorrt,trt \
  --ienv.ML_MODEL_MAX_BATCH_SIZE=20 \
  --ienv.ML_MODEL_DATA_TYPE=float16 \
  --ienv.ML_MODEL_DATA_LAYOUT=NCHW \
  --ienv.ML_MODEL_NORMALIZE_DATA=NO \
  --ienv.ML_MODEL_SUBTRACT_MEAN=YES \
  --ienv.ML_MODEL_GIVEN_CHANNEL_MEANS='123.68 116.78 103.94' \
  --ienv.ML_MODEL_IMAGE_HEIGHT=224 \
  --ienv.ML_MODEL_IMAGE_WIDTH=224
  exit_if_error
fi


echo
echo "Done."
