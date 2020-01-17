#!/bin/bash

echo "Masha experiment:"

# Timestamp.
timestamp=$(date +%Y%m%d-%H%M%S)
echo "- timestamp: ${timestamp}"

# Hub IP.
hub_ip=${CK_HUB_IP:-"192.168.1.101"}
echo "- hub IP: ${hub_ip}"

# Use parentheses to interpret the string as an array.
ids=(${CK_WORKER_IDS:-"1 2"})
num_ids=${#ids[@]}
echo "- ${num_ids} worker(s): ${ids[@]}" # FIXME: prints 1 sometimes?

# Time each worker should wait after last received work-item before exiting.
post_work_timeout_s=${CK_ZMQ_POST_WORK_TIMEOUT:-5}
echo "- post-work timeout: ${post_work_timeout_s} s"

# By default, use 6! (6 factorial), which is evenly
# divisible between any number of co-processors 1-6.
count=${CK_LOADGEN_COUNT_OVERRIDE:-720}
echo "- count: ${count}"

# count_per_copro=$(( ${count} / ${num_ids} ))

# Multistreamness.
multistreamness=${CK_LOADGEN_MULTISTREAMNESS:-1}
echo "- multistreamness: ${multistreamness}"

# Batch size.
batch_size=${CK_BATCH_SIZE:-1}
echo "- batch size: ${batch_size}"

# Transfer mode: raw, json, pickle, numpy.
transfer_mode=${CK_ZMQ_TRANSFER_MODE:-numpy}
echo "- transfer mode: ${transfer_mode}"

# FP mode: NO, YES.
fp_mode=${CK_ZMQ_FP_MODE:-YES}
if [ "${fp_mode}" = "YES" ]; then
  fp_mode_tag="yes"
elif [ "${fp_mode}" = "NO" ]; then
  fp_mode_tag="no"
else
  echo "ERROR: Unsupported FP mode '${fp_mode}'!"
  exit 1
fi
echo "- FP mode: ${fp_mode} (${fp_mode_tag})"

# Number of samples to discard when warming up:
# by default, use as many as the number of co-processors.
warm_up_samples=${CK_LOADGEN_WARM_UP_SAMPLES:-${num_ids}}
echo "- warm up samples: ${warm_up_samples}"

# LoadGen scenario: MultiStream, Offline.
scenario=${CK_LOADGEN_SCENARIO:-MultiStream}
if [ "${scenario}" = "MultiStream" ]; then
  scenario_tag="multistream"
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

if [ "${mode}" = "AccuracyOnly" ]; then
  dataset_size=50000
  buffer_size=500
else
  dataset_size=1024
  buffer_size=1024
fi

echo

# Prepare record UOA and tags.
record_uoa="masha.${timestamp}.scenario-${scenario_tag}.mode-${mode_tag}.count-${count}.multistreamness-${multistreamness}.batch-${batch_size}.transfer-${transfer_mode}.fp-${fp_mode_tag}"
record_tags="masha,${timestamp},scenario-${scenario_tag},mode-${mode_tag},count-${count},multistreamness-${multistreamness},batch-${batch_size},transfer-${transfer_mode},fp-${fp_mode_tag}"

# Launch Masha's workers.
for id in ${ids[@]}; do
    worker_id="worker-${id}"
    id_plus_1=$((id+1))
    worker_ip="192.168.1.10${id_plus_1}"
    read -d '' CMD <<END_OF_CMD
    ssh -n -f ${USER}@${worker_ip} \
    "bash -c 'nohup \
        ck benchmark program:zpp-worker-tensorrt-py --repetitions=1 \
        --dep_add_tags.weights=converted-from-onnx,maxbatch.20,fp16 \
        --env.CK_HUB_IP=${hub_ip} \
        --env.CK_WORKER_ID=${worker_id} \
        --env.CK_ZMQ_POST_WORK_TIMEOUT_S=${post_work_timeout_s} \
        --record --record_repo=local \
        --record_uoa=${record_uoa}.${worker_id} \
        --tags=${record_tags},${worker_id} \
        --skip_print_timers --skip_stat_analysis --process_multi_keys \
    > /home/$USER/nohup.log 2>&1 &'"
END_OF_CMD
    echo ${CMD}
    eval ${CMD}
    echo
done
# Wait a bit.
sleep 1s

# Launch Masha's hub.
read -d '' CMD <<END_OF_CMD
ck benchmark program:image-classification-zpp-hub-loadgen-py --repetitions=1 \
--dep_add_tags.weights=converted-from-onnx,maxbatch.20,fp16 \
--dep_add_tags.images=rgb8 \
--env.CK_LOADGEN_SCENARIO=${scenario} \
--env.CK_LOADGEN_MODE=${mode} \
--env.CK_LOADGEN_COUNT_OVERRIDE=${count} \
--env.CK_LOADGEN_MULTISTREAMNESS=${multistreamness} \
--env.CK_LOADGEN_WARM_UP_SAMPLES=${warm_up_samples} \
--env.CK_BATCH_SIZE=${batch_size} \
--env.CK_SILENT_MODE=YES \
--record --record_repo=local \
--record_uoa=${record_uoa}.hub \
--tags=${record_tags},hub \
--skip_print_timers --skip_stat_analysis --process_multi_keys
END_OF_CMD
echo ${CMD}
eval ${CMD}
echo

# Show the summary table.
tail -n 12 "`ck find program:image-classification-zpp-hub-loadgen-py`/tmp/mlperf_log_summary.txt"
echo

echo "Done."
