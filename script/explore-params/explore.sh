#!/bin/bash

echo "ZeroMQ Push-Pull exploration!"

# Dry run - print commands but do not execute them.
dry_run=${CK_DRY_RUN:-""}
echo "- dry run: ${dry_run}"

# Time each worker should wait after last received work-item before exiting.
postwork_timeout_s=${CK_WORKER_POSTWORK_TIMEOUT_S:-10}
echo "- postwork timeout: ${postwork_timeout_s} s"

# Directory where run.sh is (may not be the current one in the future).
script_dir=`ck find ck-zeromq:script:explore-params`

# LoadGen mode: PerformanceOnly, AccuracyOnly.
mode=${CK_LOADGEN_MODE:-PerformanceOnly}
if [ "${mode}" = "PerformanceOnly" ]; then
  mode_tag="performance"
  dataset_size=${CK_LOADGEN_DATASET_SIZE:-1024}
  buffer_size=${CK_LOADGEN_BUFFER_SIZE:-1024}
elif [ "${mode}" = "AccuracyOnly" ]; then
  mode_tag="accuracy"
  imagenet_size=50000
  dataset_size=${CK_LOADGEN_DATASET_SIZE:-${imagenet_size}}
  buffer_size=${CK_LOADGEN_BUFFER_SIZE:-500}
else
  echo "ERROR: Unsupported LoadGen mode '${mode}'!"
  exit 1
fi
echo "- mode: ${mode} (${mode_tag})"
echo "- dataset size: ${dataset_size}"
echo "- buffer size: ${buffer_size}"

# Define the exploration space.
if [ "${mode_tag}" = "accuracy" ]; then
  list_of_ids=("3")
  batch_sizes=(1)
else
  list_of_ids=("1" "3" "1 3")
  batch_sizes=($(seq 1 4))
fi
echo "- worker ids:  [ ${list_of_ids[@]} ]"
echo "- batch sizes: [ ${batch_sizes[@]} ]"

transfer_modes=("raw" "pickle" "numpy" "json")
echo "- transfer modes: [ ${transfer_modes[@]} ]"

transfer_floats=("YES" "NO")
echo "- transfer floats: [ ${transfer_floats[@]} ]"

# Blank line.
echo

# Run once for each point.
experiment_id=1
for ids in "${list_of_ids[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for transfer_mode in "${transfer_modes[@]}"; do
      for transfer_float in "${transfer_floats[@]}"; do
        if [ "${transfer_float}" = "YES" ] || [ "${transfer_mode}" = "json" ] ; then
          preprocess_on_gpu_list=("NO")
	else
          preprocess_on_gpu_list=("NO" "YES")
        fi
        for preprocess_on_gpu in "${preprocess_on_gpu_list[@]}"; do
            echo "[`date`] Experiment #${experiment_id}: ..."
            experiment_id=$(( ${experiment_id}+1 ))
            read -d '' CMD <<END_OF_CMD
cd ${script_dir};
CK_DRY_RUN=${dry_run} \
CK_LOADGEN_MODE=${mode} \
CK_LOADGEN_DATASET_SIZE=${dataset_size} \
CK_LOADGEN_BUFFER_SIZE=${buffer_size} \
CK_WORKER_POSTWORK_TIMEOUT_S=${postwork_timeout_s} \
CK_WORKER_IDS="${ids}" \
CK_BATCH_SIZE=${batch_size} \
CK_TRANSFER_MODE=${transfer_mode} \
CK_TRANSFER_FLOAT=${transfer_float} \
CK_PREPROCESS_ON_GPU=${preprocess_on_gpu} \
./run.sh
END_OF_CMD
            echo ${CMD}
            if [ -z "${dry_run}" ]; then
              eval ${CMD}
	    fi
            echo
	done # preprocess on gpu
      done # transfer float
    done # transfer mode
  done # batch size
done # ids

if [ -z "${dry_run}" ]; then
  echo "[`date`] Done."
else
  echo "[`date`] Done (dry run)."
fi
echo
