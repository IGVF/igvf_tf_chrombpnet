#!/bin/bash
# resume_box.sh
# Purpose: carry a box's unfinished work into the next box: 03.0 tasks still
#   queued (RESUME_BIAS_QUEUE), the bias QC (bias_qc_box.py), every dataset's
#   full chain (chains_box.sh -> full_chain.sh, 04.0 -> 08.0) for the configs
#   in RESUME_CONFIGS, and any extra command (RESUME_EXTRA).
#
# A box has a time limit, and what it has not finished by then is killed. Every
# step skips outputs that exist (04.0 retrains a model cut off before its
# footprints), so the same tools pointed at the same configs in a new box pick
# up where the last one stopped. Submit it with --dependency=afterany:<box> so
# it starts only once that box has ended: never two boxes at once.
#
# 03.0: RESUME_BIAS_QUEUE holds "<config>\t<index>" lines, the format of a
# box's gpu_queue.tsv; RESUME_BIAS_SLOTS workers per GPU (default 5) take them
# in order, each under an 8-core CPU mask (an unmasked JAX process holds ~750
# threads on this node against ulimit -u 32768). bias_qc_box.py reads the same
# gpu_queue.tsv and status.tsv: while tasks are queued or training, models are
# still coming and it uses its busy limits.
#
# Input:  RESUME_CONFIGS: a file listing config.yaml paths, one per line, in
#         the order chains should start (default: every config under
#         ${DATASET_ROOT}/chrombpnet/configs). The environment of the box that
#         trained the bias models (workflows/dcai/env.sh, CHROMBPNET_REPO,
#         CHROMBPNET_PIXI_ENV, BIAS_FACTORS_FROM_SCAN).
#         RESUME_BIAS_QUEUE, RESUME_BIAS_SLOTS: see above.
#         RESUME_EXTRA: a command run in the background beside the rest (e.g. the
#         unfinished part of an experiment), logged to extra.log.
#         RESUME_DRY_RUN=1: print what bias_qc_box.py and chains_box.sh would
#         do, start nothing.
# Output: ${DATASET_ROOT}/chrombpnet/box/<job id>/: bias_qc.tsv,
#         full_chain.tsv, logs/, bias_qc_box.log, chains_box.log, resources.tsv.
# Usage:  source workflows/dcai/env.sh
#         RESUME_CONFIGS=<list> sbatch --dependency=afterany:<box> \
#             --chdir "${DATASET_ROOT}/chrombpnet" workflows/dcai/resume_box.sbatch

set -uo pipefail
: "${REPO_ROOT:?source workflows/dcai/env.sh first}"
# shellcheck source=workflows/dcai/env.sh
source "${REPO_ROOT}/workflows/dcai/env.sh"

box_id="${SLURM_JOB_ID:-local.$$}"
BOX_DIR="${DATASET_ROOT}/chrombpnet/box/${box_id}"
export BOX_DIR
mkdir -p "${BOX_DIR}/logs"
: > "${BOX_DIR}/gpu_queue.tsv"
: > "${BOX_DIR}/status.tsv"
if [[ -n "${RESUME_BIAS_QUEUE:-}" ]]; then
    cp "${RESUME_BIAS_QUEUE}" "${BOX_DIR}/gpu_queue.tsv"
fi

if [[ -n "${RESUME_CONFIGS:-}" ]]; then
    mapfile -t configs < <(grep -v '^\s*$' "${RESUME_CONFIGS}")
else
    mapfile -t configs < <(ls "${DATASET_ROOT}"/chrombpnet/configs/*/config.yaml)
fi
echo "[$(date)] box ${box_id}: ${#configs[@]} dataset(s); logs in ${BOX_DIR}"

if [[ "${RESUME_DRY_RUN:-0}" == "1" ]]; then
    "${REPO_ROOT}/.pixi/envs/qc/bin/python" "${REPO_ROOT}/workflows/dcai/bias_qc_box.py" \
        --box-dir "${BOX_DIR}" --configs "${configs[@]}" --dry-run > "${BOX_DIR}/qc_dry_run.txt"
    head -n 3 "${BOX_DIR}/qc_dry_run.txt"
    echo "03.0 tasks queued: $(wc -l < "${BOX_DIR}/gpu_queue.tsv")"
    CHAINS_DRY_RUN=1 bash "${REPO_ROOT}/workflows/dcai/chains_box.sh" "${configs[@]}"
    exit 0
fi

# Node-local temp files and JAX cache, where full_chain.sh and bias_qc_box.py look.
scratch="/dev/shm/${USER}/box.${box_id}"
rm -rf "/dev/shm/${USER}/box."* 2>/dev/null
mkdir -p "${scratch}/tmp" "${scratch}/jax_cache"
export TMPDIR="${scratch}/tmp" TMP="${scratch}/tmp" TEMP="${scratch}/tmp"
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax_cache" XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

# Memory, load and per-GPU use every 60 s, to size the next box.
(
    echo -e "time\tmem_used_gb\tload1\tgpus(util%,mem_gb)"
    while sleep 60; do
        printf '%s\t%s\t%s\t%s\n' "$(date +%FT%T)" \
            "$(free -g | awk '/^Mem:/ {print $3}')" "$(cut -d' ' -f1 /proc/loadavg)" \
            "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
               | awk -F', ' '{printf "%s%d,%d", (NR > 1 ? " " : ""), $1, $2 / 1024}')"
    done
) > "${BOX_DIR}/resources.tsv" &
sampler=$!

cd "${REPO_ROOT}" || exit 1

# 03.0 workers, as run_box.sh's but CPU-masked; they stop when the queue is empty.
mapfile -t allowed < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
pop_task() {
    (
        flock 9
        head -n 1 "${BOX_DIR}/gpu_queue.tsv"
        sed -i '1d' "${BOX_DIR}/gpu_queue.tsv"
    ) 9> "${BOX_DIR}/gpu_queue.tsv.lock"
}
bias_worker() {
    local gpu="$1" w="$2" task cfg idx name t0 rc cpus i out=()
    for (( i = 0; i < 8; i++ )); do out+=( "${allowed[$(( (8 * w + i) % ${#allowed[@]} ))]}" ); done
    cpus="$(IFS=,; echo "${out[*]}")"
    while task="$(pop_task)"; [[ -n "${task}" ]]; do
        cfg="${task%%$'\t'*}"; idx="${task##*$'\t'}"
        name="$(basename "$(dirname "${cfg}")")"
        mkdir -p "${BOX_DIR}/logs/${name}"
        t0="$(date +%s)"
        CUDA_VISIBLE_DEVICES="${gpu}" SLURM_ARRAY_TASK_ID="${idx}" SLURM_CPUS_PER_TASK=4 \
            DATASET_CONFIG="${cfg}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
            NUMEXPR_MAX_THREADS=4 NUMBA_NUM_THREADS=4 \
            taskset -c "${cpus}" bash workflows/SLURM/03.0.train_bias_model.sh \
            > "${BOX_DIR}/logs/${name}/03.0.${idx}.log" 2>&1
        rc=$?
        printf '%s\t%s\t%d\t%d\n' "${name}" "03.0.train_bias_model[${idx}]@gpu${gpu}" "${rc}" \
            "$(( $(date +%s) - t0 ))" >> "${BOX_DIR}/status.tsv"
    done
}
workers=()
if [[ -s "${BOX_DIR}/gpu_queue.tsv" ]]; then
    n_gpus="$(nvidia-smi -L | wc -l)"
    echo "[$(date)] 03.0: $(wc -l < "${BOX_DIR}/gpu_queue.tsv") task(s), ${RESUME_BIAS_SLOTS:-5} per GPU on ${n_gpus} GPU(s)"
    for (( s = 0; s < ${RESUME_BIAS_SLOTS:-5}; s++ )); do
        for (( g = 0; g < n_gpus; g++ )); do
            bias_worker "${g}" "$(( s * n_gpus + g ))" &
            workers+=( $! )
            sleep 15   # the first compiles fill the JAX cache before the rest need it
        done
    done
fi

extra=""
if [[ -n "${RESUME_EXTRA:-}" ]]; then
    bash -c "${RESUME_EXTRA}" > "${BOX_DIR}/extra.log" 2>&1 &
    extra=$!
fi

"${REPO_ROOT}/.pixi/envs/qc/bin/python" workflows/dcai/bias_qc_box.py \
    --box-dir "${BOX_DIR}" --configs "${configs[@]}" --grace-min 0 > "${BOX_DIR}/bias_qc_box.log" 2>&1 &
qc=$!
bash workflows/dcai/chains_box.sh "${configs[@]}" > "${BOX_DIR}/chains_box.log" 2>&1
wait "${qc}" ${workers[@]+"${workers[@]}"} ${extra}
kill "${sampler}" 2>/dev/null
echo "[$(date)] box ${box_id}: done"
echo "full chain steps failed:"
awk -F'\t' '$3 != 0' "${BOX_DIR}/full_chain.tsv" 2>/dev/null
