#!/bin/bash


function exit_if_error() {
    if [ "${?}" != "0" ]; then exit 1; fi
}


echo "Setting up CK-ZeroMQ ..."

# Skip Python setup: should be NO for hub; should be NO for worker.
skip_python_setup=${CK_SKIP_PYTHON_SETUP:-"NO"}
echo "- skip Python setup: ${skip_python_setup}"

# Skip NVIDIA setup: can be YES or NO for hub; should be NO for worker.
skip_nvidia_setup=${CK_SKIP_NVIDIA_SETUP:-"NO"}
echo "- skip NVIDIA setup: ${skip_nvidia_setup}"

# Skip LoadGen setup: should be NO for hub; can be YES or NO for worker.
skip_loadgen_setup=${CK_SKIP_LOADGEN_SETUP:-"NO"}
echo "- skip LoadGen setup: ${skip_loadgen_setup}"


echo


# Refresh CK-ZeroMQ and its dependencies.
echo "Refreshing CK-ZeroMQ ..."
ck pull repo:ck-zeromq --url=https://github.com/dividiti/ck-zeromq
exit_if_error


echo


if [ "${skip_python_setup}" == "NO" ]; then
  # Set up Python, NumPy, PyZMQ.
  echo "Setting up Python 3 and essential packages ..."
  ck detect soft:compiler.python --full_path=`which python3`
  exit_if_error

  ck install package --tags=python-package,cython
  exit_if_error

  # NB: Building NumPy 1.18.1 requires Cython >= 0.29.14.
  ck virtual env --tags=cython --shell_cmd='ck install package --tags=python-package,numpy'
  exit_if_error

  ck install package --tags=python-package,zmq
  exit_if_error
fi


if [ "${skip_nvidia_setup}" == "NO" ]; then
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
fi


if [ "${skip_loadgen_setup}" == "NO" ]; then
  # Install MLPerf Inference packages.
  echo "Setting up MLPerf Inference packages ..."

  ck install package --tags=mlperf,inference,source
  exit_if_error

  ck install package --tags=python-package,absl
  exit_if_error

  ck install package --tags=mlperf,loadgen,python-package
  exit_if_error
fi


echo


echo "Done."
