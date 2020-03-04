#!/bin/bash

echo "ZeroMQ Push-Pull experiment:"

# Hub-side program.
program="image-classification-zpp-hub-loadgen-py"
program_dir=`ck find ck-zeromq:program:${program}`
echo "- program: ${program}"
echo "- program directory: ${program_dir}"

# LoadGen config file.
config_file=${CK_ENV_LOADGEN_CONFIG_FILE:-${program_dir}/user.conf}
echo "- config file: ${config_file}"

# Dry run - print commands but do not execute them.
dry_run=${CK_DRY_RUN:-""}
echo "- dry run: ${dry_run}"

# Skip existing experiments.
skip_existing=${CK_SKIP_EXISTING:-""}
echo "- skip existing: ${skip_existing}"

# Timestamp.
timestamp=$(date +%Y%m%d-%H%M%S)
echo "- timestamp: ${timestamp}"

# Hub IP.
hub_ip=${CK_HUB_IP:-"192.168.1.102"}
echo "- hub IP: ${hub_ip}"

# Use parentheses to interpret the string as an array.
ids=(${CK_WORKER_IDS:-1})
num_ids=${#ids[@]}
echo "- ${num_ids} worker(s): ${ids[@]}"

# Time each worker should wait after last received work-item before exiting.
postwork_timeout_s=${CK_WORKER_POSTWORK_TIMEOUT_S:-10}
echo "- postwork timeout: ${postwork_timeout_s} s"

# Worker response format: argmax returns class id, softmax returns
# 1000 or 1001-element vector of class probabilities.
worker_output=${CK_WORKER_OUTPUT_FORMAT:-argmax}
echo "- worker output: ${worker_output}"

# Transfer mode: raw, json, pickle, numpy.
transfer_mode=${CK_ZMQ_TRANSFER_MODE:-numpy}
echo "- transfer mode: ${transfer_mode}"

# FP mode: YES, NO. NB: fp_mode_tag is no longer used.
fp_mode=${CK_FP_MODE:-YES}
if [ "${fp_mode}" = "YES" ]; then
  fp_mode_tag="yes"
elif [ "${fp_mode}" = "NO" ]; then
  fp_mode_tag="no"
else
  echo "ERROR: Unsupported FP mode '${fp_mode}'!"
  exit 1
fi
echo "- FP mode: ${fp_mode} (${fp_mode_tag})"

# Preprocess on GPU: NO, YES.
preprocess_on_gpu=${CK_PREPROCESS_ON_GPU:-NO}
if [ "${fp_mode}" = "YES" ] && [ "${preprocess_on_gpu}" = "YES" ]; then
  echo "WARNING: Forcing not to preprocess on GPU since transferring in FP mode!"
  preprocess_on_gpu="NO"
fi
echo "- preprocess on GPU: ${preprocess_on_gpu}"

# Number of samples to discard when warming up:
# by default, use as many as the number of co-processors.
warmup_samples=${CK_LOADGEN_WARMUP_SAMPLES:-${num_ids}}
echo "- warm-up samples: ${warmup_samples}"

# Batch size.
batch_size=${CK_BATCH_SIZE:-1}
echo "- batch size: ${batch_size}"

# LoadGen scenario: MultiStream, SingleStream, Offline.
scenario=${CK_LOADGEN_SCENARIO:-MultiStream}
if [ "${scenario}" = "MultiStream" ]; then
  scenario_tag="multistream"
elif [ "${scenario}" = "SingleStream" ]; then
  scenario_tag="singlestream"
elif [ "${scenario}" = "Offline" ]; then
  scenario_tag="offline"
else
  echo "ERROR: Unsupported LoadGen scenario '${scenario}'!"
  exit 1
fi
echo "- scenario: ${scenario} (${scenario_tag})"

# LoadGen mode: PerformanceOnly, AccuracyOnly.
mode=${CK_LOADGEN_MODE:-PerformanceOnly}
if [ "${mode}" = "PerformanceOnly" ]; then
  mode_tag="performance"
elif [ "${mode}" = "AccuracyOnly" ]; then
  mode_tag="accuracy"
else
  echo "ERROR: Unsupported LoadGen mode '${mode}'!"
  exit 1
fi
echo "- mode: ${mode} (${mode_tag})"

imagenet_size=50000
if [ "${mode}" = "AccuracyOnly" ]; then
  dataset_size=${CK_LOADGEN_DATASET_SIZE:-${imagenet_size}}
  buffer_size=${CK_LOADGEN_BUFFER_SIZE:-500}
else
  dataset_size=${CK_LOADGEN_DATASET_SIZE:-1024}
  buffer_size=${CK_LOADGEN_BUFFER_SIZE:-1024}
fi
echo "- dataset size: ${dataset_size}"
echo "- buffer size: ${buffer_size}"

# In the PerformanceOnly mode, affects the number of samples per query that LoadGen issues
# (aiming to meet the minimum duration of 60 seconds and, in the Offline mode, the minimum
# number of samples of 24,576).
target_qps=${CK_LOADGEN_TARGET_QPS:-70}
if [ "${mode}" = "PerformanceOnly" ]; then
  TARGET_QPS="--env.CK_LOADGEN_TARGET_QPS=${target_qps}"
fi
if [ "${scenario}" == "MultiStream" ]; then
  echo "Warning: TARGET_QPS must not be overriden in the MultiStream scenario!"
  TARGET_QPS=""
fi
echo "- target QPS (queries per second): ${target_qps} ('${TARGET_QPS}')"

# In the MultiStream scenario, affects the number of streams that LoadGen issues
# (aiming to meet the target latency of 50 ms).
multistreamness=${CK_LOADGEN_MULTISTREAMNESS:-1}
if [ "${scenario}" = "MultiStream" ]; then
  MULTISTREAMNESS="--env.CK_LOADGEN_MULTISTREAMNESS=${multistreamness}"
fi
echo "- multistreamness: ${multistreamness} ('${MULTISTREAMNESS}')"

# Allow to override the number of queries in the PerformanceOnly mode.
# FIXME: Only override when TARGET_QPS is not explicitly defined?
# By default, use 6! (6 factorial), which is evenly
# divisible between any number of co-processors 1-6.
count_override=${CK_LOADGEN_COUNT_OVERRIDE:-720}
if [ "${mode}" = "PerformanceOnly" ]; then
  COUNT_OVERRIDE="--env.CK_LOADGEN_COUNT_OVERRIDE=${count_override}"
fi
echo "- count override: ${count_override} ('${COUNT_OVERRIDE}')"

# Prepare record UOA and tags.
mlperf="mlperf"
division="closed"
task="image-classification"
platform="tx2"
library="zpp" # ZeroMQ Push-Pull.
benchmark="resnet"
record_uoa="${mlperf}.${division}.${task}.${platform}.${library}.${benchmark}.${scenario_tag}.${mode_tag}"
record_tags="${mlperf},${division},${task},${platform},${library},${benchmark},${scenario_tag},${mode_tag}"
if [ "${mode_tag}" = "accuracy" ]; then
  # Get substring after "preprocessed," to end, i.e. "using-opencv" here.
  model_preprocessing_tags="full,side.224,preprocessed,using-opencv"
  preprocessing="${model_preprocessing_tags##*preprocessed,}"
  record_uoa+=".${preprocessing}"
  record_tags+=",${preprocessing}"
fi
if [ "${mode_tag}" = "accuracy" ] && [ "${dataset_size}" != "${imagenet_size}" ]; then
  record_uoa+=".${dataset_size}"
  record_tags+=",${dataset_size}"
fi
echo "- record UOA: ${record_uoa}"
echo "- record tags: ${record_tags}"

# Blank line before printing commands.
echo

if (ck find experiment:${record_uoa} >/dev/null) && [[ "${skip_existing}" ]]; then
  echo "Experiment '${record_uoa}' already exists, skipping ..."
  exit 0
fi


# Launch the worker programs.
for id in ${ids[@]}; do
  worker_id="worker-${id}"
  id_plus_1=$((id+1))
  worker_ip="192.168.1.10${id_plus_1}"
  read -d '' CMD <<END_OF_CMD
  ssh -n -f ${USER}@${worker_ip} \
  "bash -c 'nohup \
    ck benchmark program:zpp-worker-tensorrt-py --repetitions=1 \
    --dep_add_tags.weights=converted-from-onnx,maxbatch.20,fp16 \
    --dep_add_tags.lib-python-tensorrt=v5 \
    --env.CK_HUB_IP=${hub_ip} \
    --env.CK_WORKER_ID=${worker_id} \
    --env.CK_WORKER_OUTPUT_FORMAT=${worker_output} \
    --env.CK_WORKER_POSTWORK_TIMEOUT_S=${postwork_timeout_s} \
    --env.CK_PREPROCESS_ON_GPU=${preprocess_on_gpu} \
    --env.CK_ZMQ_TRANSFER_MODE=${transfer_mode} \
    --env.CK_FP_MODE=${fp_mode} \
    --record --record_repo=local \
    --record_uoa=${record_uoa}.${worker_id} \
    --tags=${record_tags},${worker_id} \
    --skip_print_timers --skip_stat_analysis --process_multi_keys \
  > /home/$USER/nohup.log 2>&1 &'"
END_OF_CMD
  echo ${CMD}
  if [ -z "${dry_run}" ]; then
    eval ${CMD}
  fi
  echo
done

# Wait a bit.
sleep 1s

# Launch the hub program.
read -d '' CMD <<END_OF_CMD
ck benchmark program:${program} --repetitions=1 \
--dep_add_tags.weights=converted-from-onnx,maxbatch.20,fp16 \
--dep_add_tags.images=rgb8 \
--env.CK_ENV_LOADGEN_CONFIG_FILE=${config_file} \
--env.CK_LOADGEN_SCENARIO=${scenario} \
--env.CK_LOADGEN_MODE=${mode} \
--env.CK_LOADGEN_DATASET_SIZE=${dataset_size} \
--env.CK_LOADGEN_BUFFER_SIZE=${buffer_size} \
--env.CK_LOADGEN_WARMUP_SAMPLES=${warmup_samples} \
--env.CK_PREPROCESS_ON_GPU=${preprocess_on_gpu} \
--env.CK_ZMQ_TRANSFER_MODE=${transfer_mode} \
--env.CK_FP_MODE=${fp_mode} \
--env.CK_BATCH_SIZE=${batch_size} \
${MULTISTREAMNESS} \
${TARGET_QPS} \
${COUNT_OVERRIDE} \
--env.CK_SILENT_MODE=YES \
--record --record_repo=local \
--record_uoa=${record_uoa}.hub \
--tags=${record_tags},hub \
--skip_print_timers --skip_stat_analysis --process_multi_keys
END_OF_CMD
echo ${CMD}
if [ -z "${dry_run}" ]; then
  eval ${CMD}
fi
echo

# Show the summary table.
if [ -z "${dry_run}" ]; then
  # Show the summary table.
  head -n 12 "${program_dir}/tmp/mlperf_log_summary.txt"
  echo
  echo "Done."
else
  echo "Done (dry run)."
fi
