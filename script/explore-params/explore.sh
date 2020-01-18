#!/bin/bash

echo "Masha exploration!"

# Define the exploration space.
list_of_ids=("2 3 4" "2 3 4 5" "2 3 4 5 6" "2 3 4 5 6 1")
multistreamnesses=($(seq 4 10))
batch_sizes=($(seq 1 3))
transfer_modes=("raw" "json" "pickle" "numpy")
fp_modes=("NO" "YES")

echo "- worker ids: ${list_of_ids[@]}"
echo "- multistreamnesses: ${multistreamnesses[@]}"
echo "- batch sizes: ${batch_sizes[@]}"
echo "- transfer modes: ${transfer_modes[@]}"
echo "- FP modes: ${fp_modes[@]}"

script_dir=`ck find ck-zeromq:script:explore-params`

# Run Masha once for each point.
experiment_id=1
for ids in "${list_of_ids[@]}"; do
    for multistreamness in "${multistreamnesses[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for transfer_mode in "${transfer_modes[@]}"; do
                for fp_mode in "${fp_modes[@]}"; do
                    echo "[`date`] Experiment #${experiment_id}: ..."
                    experiment_id=$((${experiment_id}+1))
                    read -d '' CMD <<END_OF_CMD
                    cd ${script_dir};
                    CK_WORKER_IDS="${ids}" \
                    CK_LOADGEN_MULTISTREAMNESS="${multistreamness}" \
                    CK_BATCH_SIZE="${batch_size}" \
                    CK_ZMQ_TRANSFER_MODE="${transfer_mode}" \
                    CK_ZMQ_FP_MODE="${fp_mode}" \
                    ./run.sh
END_OF_CMD
                    echo ${CMD}
                    eval ${CMD}
                    echo
                done # fp mode
            done # transfer mode
        done # batch size
    done # multistreamness
done # ids

echo "[`date`] Done."
echo
