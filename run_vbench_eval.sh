#!/usr/bin/env bash
# Launch HyDRA inference on the wmagent vbench input set across 4 GPUs.
# Each worker handles a 1/N shard of the (scene, trajectory) job list.
# Existing output mp4s are skipped; pass --overwrite to redo them.
#
# Usage:
#   bash run_vbench_eval.sh                # 4 GPUs (0-3), default 4 shards
#   N_GPUS=8 GPUS="0,1,2,3,4,5,6,7" bash run_vbench_eval.sh   # 8 shards
#   EXTRA_ARGS="--overwrite" bash run_vbench_eval.sh          # extra flags
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Conda env activation (skip if already active)
if [[ "${CONDA_DEFAULT_ENV:-}" != "hydra" ]]; then
    source /data/ziqi/miniconda3/etc/profile.d/conda.sh
    conda activate hydra
fi

GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "${GPUS}"
N_GPUS="${N_GPUS:-${#GPU_ARR[@]}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[launch] N=${N_GPUS} GPUs=${GPUS} extra='${EXTRA_ARGS}'"

pids=()
for i in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$i]}"
    log="${LOG_DIR}/shard${i}.log"
    echo "[launch]   shard ${i}/${N_GPUS} -> GPU ${gpu}, log ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup \
        python "${REPO_DIR}/run_vbench_eval.py" \
            --shard "${i}/${N_GPUS}" \
            ${EXTRA_ARGS} \
        > "${log}" 2>&1 &
    pids+=($!)
done

echo "[launch] PIDs: ${pids[*]}"
echo "[launch] tail logs with:  tail -f ${LOG_DIR}/shard*.log"
echo "[launch] waiting for all shards to finish..."
wait "${pids[@]}"
echo "[launch] done."
