#!/bin/bash


function exit_if_error() {
    if [ "${?}" != "0" ]; then exit 1; fi
}


# Refresh CK-MLPerf and its dependencies.
echo "Refreshing CK-MLPerf ..."
ck pull repo:ck-mlperf
exit_if_error


echo


echo "Setting up ResNet and ImageNet ..."

# Skip ResNet setup: can be YES or NO for hub; should be NO for worker.
skip_resnet_setup=${CK_SKIP_RESNET_SETUP:-"NO"}
echo "- skip ResNet setup: ${skip_resnet_setup}"

# Fake ResNet detection: can be NO or YES for hub; should be NO for worker.
fake_resnet_detection=${CK_FAKE_RESNET_DETECTION:-"NO"}
ck_tools=${CK_TOOLS:-"$HOME/CK-TOOLS"}
echo "- fake ResNet detection: ${fake_resnet_detection} (CK_TOOLS=${ck_tools})"

if [ "${skip_resnet_setup}" == "NO" ] && [ "${fake_resnet_detection}" != "NO" ]; then
  echo "ERROR: You cannot set up ResNet and fake ResNet detection at the same time!"
  exit 1
fi

if [ "${skip_resnet_setup}" != "NO" ] && [ "${fake_resnet_detection}" == "NO" ]; then
  echo "ERROR: You cannot skip ResNet setup and not to fake ResNet detection at the same time!"
  exit 1
fi

# Skip ImageNet detection: should be NO for hub; should be YES for worker.
skip_imagenet_detection=${CK_SKIP_IMAGENET_DETECTION:-"NO"}
echo "- skip ImageNet detection: ${skip_imagenet_detection}"


echo


if [ "${skip_resnet_setup}" == "NO" ]; then
  # Install the official MLPerf ONNX model and convert it to TensorRT with predefined options.
  ck install package --tags=image-classification,model,onnx,resnet
  ck install package --tags=image-classification,model,tensorrt,resnet,converted-from-onnx,maxbatch.20,fp16
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


if [ "${skip_imagenet_detection}" == "NO" ]; then
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


echo


echo "Done."
