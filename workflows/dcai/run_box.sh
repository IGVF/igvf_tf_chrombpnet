#!/bin/bash
# run_box.sh
# Purpose: keep ONE whole DCAI node busy -- its CPUs and its GPUs -- with
#   ChromBPNet dataset work.
#
# DCAI rents a node at a time (216 usable cores, ~2 TB RAM, 8x H100) and bills
# node-time, so instead of one sbatch per step per dataset this runs them all
# in a single exclusive allocation:
#
#   phase 0  Tn5 shift of every fragments file (detect_shifts.py: scPrinter's
#            detector cross-checked against chrombpnet's), then one config per
#            file (make_configs.py) with that shift pinned. Stops if the two
#            detectors disagree or a fit is unreliable. Only wave 1's configs
#            are rewritten with this box's BOX_BIAS_* settings; the others are
#            written only if missing, and a rewrite keeps fold_bias_suffix.
#   wave 1   run_chain.sh for the BOX_FIRST datasets (default: the 10 largest
#            libraries), all at once: {00.0.call_peaks || 00.0.prepare_signal}
#            -> 00.1 -> 01.0 -> 02.0, then, with "bias" in BOX_STAGES, their
#            03.0 bias-sweep tasks go to the GPU queue as each one finishes.
#   GPUs     BOX_GPU_SLOTS_PER_GPU workers per GPU take 03.0 tasks from the
#            queue (CUDA_VISIBLE_DEVICES pinned) while the CPUs keep working.
#   wave 2   (BOX_BACKFILL) the remaining datasets' chains, as many at once as
#            the memory wave 1 was measured to need per chain allows, next to
#            whatever the GPU tasks hold at that moment.
#   last     03.1.select_bias for every dataset whose sweep ran. 03.1 writes the
#            per-fold choice; a person copies it into fold_bias_suffix (the
#            pipeline's deliberate review point) before 03.2/04.0.
#   beside   checks.sh (GPU/driver probe, figwig == numpy bigwig, MACS3 PR ==
#            3.0.4) and a sampler writing memory, /dev/shm, CPU and per-GPU use
#            every 30 s to resources.tsv -- the numbers to size the next box.
#
# Steps skip finished outputs, so resubmitting resumes where a box stopped.
#
# Knobs (environment):
#   BOX_FIRST            fragment stems for wave 1, space-separated; or
#   BOX_FIRST_N          the N largest libraries (default 10)
#   BOX_STAGES           wave 1: "prep" or "prep,bias" (default prep,bias)
#   BOX_BACKFILL         wave 2: "none", "prep" or "prep,bias" (default prep)
#   BOX_GPU_SLOTS_PER_GPU  concurrent 03.0 trainings per GPU (default 2)
#   BOX_GPU_STAGGER      seconds between GPU workers' first tasks (default 15), so
#                        the first compiles fill the cache before the rest need it
#   BOX_MEM_FRACTION     share of RAM wave 2 may plan to fill (default 0.85)
#   BOX_BIAS_PRECISION   written into wave 1's configs as bias_precision (e.g. bf16)
#   BOX_BIAS_PATIENCE    written into wave 1's configs as bias_patience (e.g. 3)
#   BOX_CHECKS           0 skips checks.sh (they need to pass once per setup, not per box)
#   BOX_MPS              NVIDIA MPS for the GPU workers: off (default), all, or half --
#                        GPUs 0..n/2-1 under MPS and the rest without, an A/B on the
#                        same tasks and node (workflows/dcai/box_rates.sh compares them)
#
# Output: under ${DATASET_ROOT}/chrombpnet/box/<job id>/: logs/<dataset>/<step>.log,
#         logs/<dataset>/03.0.<index>.log, status.tsv (dataset, step, exit,
#         seconds), resources.tsv, checks/, sizing.txt; and the steps' outputs.
# Usage:  source workflows/dcai/env.sh && sbatch workflows/dcai/run_box.sbatch
# Prerequisites: env.sh's environments installed; `cli.py download-references`.

set -uo pipefail

: "${REPO_ROOT:?source workflows/dcai/env.sh first}"
# shellcheck source=workflows/dcai/env.sh
source "${REPO_ROOT}/workflows/dcai/env.sh"

box_id="${SLURM_JOB_ID:-local.$$}"
export BOX_DIR="${DATASET_ROOT}/chrombpnet/box/${box_id}"
export BOX_LOG_DIR="${BOX_DIR}/logs"
export BOX_STATUS="${BOX_DIR}/status.tsv"
export BOX_GPU_QUEUE="${BOX_DIR}/gpu_queue.tsv"
mkdir -p "${BOX_LOG_DIR}"
: > "${BOX_STATUS}"
: > "${BOX_GPU_QUEUE}"
chains_done="${BOX_DIR}/.chains_done"
rm -f "${chains_done}"
configs_dir="${DATASET_ROOT}/chrombpnet/configs"
shifts="${DATASET_ROOT}/chrombpnet/shifts.json"
stages="${BOX_STAGES:-prep,bias}"
backfill="${BOX_BACKFILL:-prep}"
slots_per_gpu="${BOX_GPU_SLOTS_PER_GPU:-2}"
mem_fraction="${BOX_MEM_FRACTION:-0.85}"
steps_dir="${REPO_ROOT}/workflows/SLURM"

# Scratch for the pseudoreplicate files: node-local disk if there is one, else
# tmpfs. A killed job skips the EXIT trap, so clear what earlier boxes of ours
# left before starting.
scratch_root=/dev/shm
[[ -d /raid && -w /raid ]] && scratch_root=/raid
rm -rf "${scratch_root}/${USER}/box."* 2>/dev/null
export BOX_SCRATCH="${scratch_root}/${USER}/box.${box_id}"
mkdir -p "${BOX_SCRATCH}"

# Temporary files on the node, never on the shared filesystem. XLA compiles each
# GPU kernel by running ptxas on temp files in $TMPDIR; with an inherited TMPDIR
# on /dcai, 40 trainings compiling at once left ~3,900 ptxas processes blocked
# listing and unlinking one network directory (box 517771: load 4,300, 7% CPU).
export TMPDIR="${BOX_SCRATCH}/tmp" TMP="${BOX_SCRATCH}/tmp" TEMP="${BOX_SCRATCH}/tmp"
mkdir -p "${TMPDIR}"
# One persistent JAX compilation cache for every training on the node: the sweep
# compiles the same model at the same shapes many times over, so all but the
# first compile of each (and its autotuning) become cache reads.
export JAX_COMPILATION_CACHE_DIR="${BOX_SCRATCH}/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

mem_used_gb() { awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%d", (t-a)/1048576}' /proc/meminfo; }
mem_total_gb() { awk '/^MemTotal:/{printf "%d", $2/1048576}' /proc/meminfo; }
n_gpus="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
# chrombpnet's cuda13 environment needs NVIDIA driver >= 580; cuda12 needs 525.
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
if [[ -n "${driver}" && "${driver%%.*}" -lt 580 && "${CHROMBPNET_PIXI_ENV}" == "cuda13" ]]; then
    export CHROMBPNET_PIXI_ENV=cuda12
    echo "[$(date)] driver ${driver} < 580: chrombpnet runs in its cuda12 environment"
fi

echo "[$(date)] box ${box_id} on $(hostname): $(nproc) cores, $(mem_total_gb) GB RAM, ${n_gpus} GPU(s), driver ${driver:-?}"
echo "           scratch ${BOX_SCRATCH} ($(df -h --output=avail "${scratch_root}" | tail -1 | tr -d ' ') free)"
echo "           logs ${BOX_LOG_DIR}"

# ── resource sampler ──────────────────────────────────────────────────────────
sampler() {
    local prev_total=0 prev_idle=0 total idle busy gpus
    printf 'time\tmem_used_gb\tshm_used_gb\tload1\tcpu_busy_pct\tgpus(util%%,mem_gb)\n' > "${BOX_DIR}/resources.tsv"
    while true; do
        read -r _ u n s i w q sq st _ < /proc/stat
        total=$(( u + n + s + i + w + q + sq + st )); idle=$(( i + w ))
        busy=0
        (( prev_total > 0 && total > prev_total )) && \
            busy=$(( 100 * ( (total - prev_total) - (idle - prev_idle) ) / (total - prev_total) ))
        prev_total=${total}; prev_idle=${idle}
        gpus="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', ' '{printf "%s%d,%.0f", (NR>1?" ":""), $1, $2/1024}')"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$(mem_used_gb)" \
            "$(df -BG --output=used /dev/shm | tail -1 | tr -dc 0-9)" \
            "$(cut -d' ' -f1 /proc/loadavg)" "${busy}" "${gpus}" >> "${BOX_DIR}/resources.tsv"
        sleep 30
    done
}
sampler &
sampler_pid=$!
# stop_mps — quit this box's MPS daemon, if it started one.
stop_mps() {
    [[ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]] && echo quit | nvidia-cuda-mps-control > /dev/null 2>&1
    return 0
}
trap 'kill "${sampler_pid}" 2>/dev/null; stop_mps; rm -rf "${BOX_SCRATCH}"' EXIT

# ── NVIDIA MPS ────────────────────────────────────────────────────────────────
# Several trainings share each GPU (BOX_GPU_SLOTS_PER_GPU). Without MPS the GPU
# time-slices between their CUDA contexts, and a batch-64 bias model's small
# kernels leave most of each slice idle; under MPS the contexts' kernels run on
# the GPU side by side. The box starts its own daemon (pipe and log in its
# scratch) for the chosen GPUs, proves one JAX client goes through it, and
# otherwise carries on without MPS rather than failing every task.
mps_gpus=""
case "${BOX_MPS:-off}" in
    all)  (( n_gpus > 0 )) && mps_gpus="$(seq -s, 0 $(( n_gpus - 1 )))" ;;
    half) (( n_gpus > 1 )) && mps_gpus="$(seq -s, 0 $(( n_gpus / 2 - 1 )))" ;;
    off) ;;
    *) echo "ERROR: BOX_MPS must be off, all or half (got '${BOX_MPS}')" >&2; exit 1 ;;
esac
if [[ -n "${mps_gpus}" ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="${BOX_SCRATCH}/mps/pipe" CUDA_MPS_LOG_DIRECTORY="${BOX_SCRATCH}/mps/log"
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    if CUDA_VISIBLE_DEVICES="${mps_gpus}" nvidia-cuda-mps-control -d \
        && CUDA_VISIBLE_DEVICES="${mps_gpus%%,*}" pixi run --frozen \
            --manifest-path "${CHROMBPNET_REPO}/pyproject.toml" -e "${CHROMBPNET_PIXI_ENV}" \
            python -c 'import jax, jax.numpy as jnp; print(jax.devices(), float(jnp.ones((256, 256)).sum()))' \
            > "${BOX_DIR}/mps_check.log" 2>&1 \
        && grep -qi "new client\|starting new server" "${CUDA_MPS_LOG_DIRECTORY}/control.log" 2>/dev/null; then
        echo "[$(date)] MPS on GPU(s) ${mps_gpus}; the others time-slice as before"
    else
        echo "[$(date)] WARNING: MPS did not come up (see ${BOX_DIR}/mps_check.log and" >&2
        echo "           ${CUDA_MPS_LOG_DIRECTORY}/control.log); running without it" >&2
        cp -r "${CUDA_MPS_LOG_DIRECTORY}" "${BOX_DIR}/mps_log" 2>/dev/null
        stop_mps
        unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
        mps_gpus=""
    fi
fi

# ── checks, in the background ─────────────────────────────────────────────────
checks_pid=""
if [[ "${BOX_CHECKS:-1}" == "1" ]]; then
    bash "${REPO_ROOT}/workflows/dcai/checks.sh" > "${BOX_DIR}/checks.log" 2>&1 &
    checks_pid=$!
fi

# ── phase 0: shift, then configs ──────────────────────────────────────────────
pixi_peaks=( pixi run --frozen --manifest-path "${REPO_ROOT}/pixi.toml" -e peaks )
if [[ ! -s "${shifts}" ]]; then
    echo "[$(date)] phase 0: detecting the Tn5 shift of every fragments file"
    n_files="$(find "${DATASET_ROOT}/fragments" -maxdepth 1 -name '*.fragments.tsv.gz' | wc -l)"
    (cd "${REPO_ROOT}" && "${pixi_peaks[@]}" python workflows/dcai/detect_shifts.py \
        --fragments-dir "${DATASET_ROOT}/fragments" \
        --genome "${REFERENCE_ROOT}/hg38/Sequence/IGVFFI0653VCGH.fasta" \
        --out "${shifts}.partial" --jobs "${n_files}" --tmp-dir "${BOX_SCRATCH}" \
        --chrombpnet-python "${CHROMBPNET_REPO}/.pixi/envs/${CHROMBPNET_PIXI_ENV}/bin/python") \
        > "${BOX_DIR}/detect_shifts.log" 2>&1
    rc=$?
    if (( rc != 0 )); then
        echo "ERROR: shift detection failed or was not trustworthy (exit ${rc}); nothing pinned." >&2
        echo "  See ${BOX_DIR}/detect_shifts.log" >&2
        exit 1
    fi
    mv "${shifts}.partial" "${shifts}"
    echo "[$(date)] shifts: $(python3 -c "import json,collections; d=json.load(open('${shifts}')); print(dict(collections.Counter(f\"{v['plus_shift']:+d}/{v['minus_shift']:+d}\" for v in d.values())))")"
fi
# Wave 1: named stems, or the N largest libraries.
if [[ -n "${BOX_FIRST:-}" ]]; then
    first_stems="${BOX_FIRST}"
else
    first_stems="$(find "${DATASET_ROOT}/fragments" -maxdepth 1 -name '*.fragments.tsv.gz' -printf '%s %f\n' \
        | sort -rn | head -n "${BOX_FIRST_N:-10}" | awk '{sub(/\.fragments\.tsv\.gz$/, "", $2); print $2}' | tr '\n' ' ')"
fi
# Only wave 1 gets this box's sweep settings: another dataset's bias_precision /
# bias_patience name the directories its 03.0 models are already in.
python3 "${REPO_ROOT}/workflows/dcai/make_configs.py" \
    --fragments-dir "${DATASET_ROOT}/fragments" --dataset-root "${DATASET_ROOT}" \
    --shifts "${shifts}" --configs-dir "${configs_dir}" \
    --bias-precision "${BOX_BIAS_PRECISION:-}" --bias-patience "${BOX_BIAS_PATIENCE:-}" \
    --rewrite "${first_stems}" > "${BOX_DIR}/configs.txt" || exit 1
: > "${BOX_DIR}/wave1.txt"; : > "${BOX_DIR}/wave2.txt"
while read -r c; do
    stem="$(basename "$(dirname "${c}")")"; stem="${stem#amsc_}"
    if [[ " ${first_stems} " == *" ${stem} "* ]]; then echo "${c}" >> "${BOX_DIR}/wave1.txt"; else echo "${c}" >> "${BOX_DIR}/wave2.txt"; fi
done < "${BOX_DIR}/configs.txt"
n1="$(wc -l < "${BOX_DIR}/wave1.txt")"; n2="$(wc -l < "${BOX_DIR}/wave2.txt")"
echo "[$(date)] phase 0 done: ${n1} dataset(s) in wave 1 (${stages}), ${n2} in wave 2 (${backfill})"

# ── GPU workers ───────────────────────────────────────────────────────────────
# pop_task — the first queued "<config>\t<index>", or nothing.
pop_task() {
    (
        flock 9
        head -n 1 "${BOX_GPU_QUEUE}"
        sed -i '1d' "${BOX_GPU_QUEUE}"
    ) 9> "${BOX_GPU_QUEUE}.lock"
}
gpu_worker() {
    local gpu="$1" delay="${2:-0}" task cfg idx name t0 rc mps=off
    local -a env_mps=( env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY )
    # A GPU outside the daemon's set must not inherit its pipe: without it a
    # client opens a plain context, as before.
    if [[ ",${mps_gpus}," == *",${gpu},"* ]]; then
        mps=on
        env_mps=( env )
    fi
    sleep "${delay}"
    while true; do
        task="$(pop_task)"
        if [[ -z "${task}" ]]; then
            [[ -f "${chains_done}" && ! -s "${BOX_GPU_QUEUE}" ]] && return 0
            sleep 20
            continue
        fi
        cfg="${task%%$'\t'*}"; idx="${task##*$'\t'}"
        name="$(basename "$(dirname "${cfg}")")"
        mkdir -p "${BOX_LOG_DIR}/${name}"
        t0="$(date +%s)"
        {
            # First line of every task log: where it ran, for box_rates.sh.
            echo "[box] gpu=${gpu} mps=${mps}"
            "${env_mps[@]}" CUDA_VISIBLE_DEVICES="${gpu}" SLURM_ARRAY_TASK_ID="${idx}" SLURM_CPUS_PER_TASK=4 \
                DATASET_CONFIG="${cfg}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
                NUMEXPR_MAX_THREADS=4 NUMBA_NUM_THREADS=4 \
                bash "${steps_dir}/03.0.train_bias_model.sh"
        } > "${BOX_LOG_DIR}/${name}/03.0.${idx}.log" 2>&1
        rc=$?
        printf '%s\t%s\t%d\t%d\n' "${name}" "03.0.train_bias_model[${idx}]@gpu${gpu}" "${rc}" \
            "$(( $(date +%s) - t0 ))" >> "${BOX_STATUS}"
    done
}
worker_pids=()
if [[ ",${stages},${backfill}," == *",bias,"* ]] && (( n_gpus > 0 )); then
    # Slot-major, so each GPU gets its first worker before any gets a second.
    for (( k = 0; k < slots_per_gpu; k++ )); do
        for (( g = 0; g < n_gpus; g++ )); do
            gpu_worker "${g}" "$(( ${#worker_pids[@]} * ${BOX_GPU_STAGGER:-15} ))" &
            worker_pids+=( $! )
        done
    done
    echo "[$(date)] ${#worker_pids[@]} GPU worker(s): ${slots_per_gpu} per GPU on ${n_gpus} GPU(s), MPS on: ${mps_gpus:-none}"
fi

# ── wave 1 ────────────────────────────────────────────────────────────────────
mem_before="$(mem_used_gb)"
wave1_start="$(date +%FT%T)"
echo "[$(date)] wave 1: ${n1} chain(s) at once (memory in use: ${mem_before} GB)"
CHAIN_STAGES="${stages}" xargs -a "${BOX_DIR}/wave1.txt" -P "$(( n1 > 0 ? n1 : 1 ))" -I{} \
    bash "${REPO_ROOT}/workflows/dcai/run_chain.sh" {}
echo "[$(date)] wave 1 done"

# ── wave 2: size it from what wave 1 used ─────────────────────────────────────
if [[ "${backfill}" != "none" ]] && (( n2 > 0 )); then
    # Peak memory over wave 1 above where it started, per chain, from the
    # sampler. It includes GPU tasks that started meanwhile, so it overstates
    # a chain's share -- the safe direction.
    peak="$(awk -F'\t' -v s="${wave1_start}" 'NR > 1 && $1 >= s && $2 > m {m = $2} END {print m + 0}' "${BOX_DIR}/resources.tsv")"
    per_chain=$(( (peak - mem_before) / (n1 > 0 ? n1 : 1) ))
    (( per_chain < 16 )) && per_chain=16
    budget="$(awk -v t="$(mem_total_gb)" -v f="${mem_fraction}" -v u="$(mem_used_gb)" 'BEGIN {printf "%d", t * f - u}')"
    p2=$(( budget / per_chain ))
    (( p2 < 1 )) && p2=1
    (( p2 > n2 )) && p2=${n2}
    {
        echo "wave 1: ${n1} chains; memory ${mem_before} GB before, ${peak} GB at peak"
        echo "per chain: ${per_chain} GB; memory in use now $(mem_used_gb) GB of $(mem_total_gb) GB"
        echo "wave 2: ${p2} of ${n2} chains at once (budget ${budget} GB at fraction ${mem_fraction})"
    } | tee "${BOX_DIR}/sizing.txt"
    CHAIN_STAGES="${backfill}" xargs -a "${BOX_DIR}/wave2.txt" -P "${p2}" -I{} \
        bash "${REPO_ROOT}/workflows/dcai/run_chain.sh" {}
    echo "[$(date)] wave 2 done"
fi
touch "${chains_done}"

# ── GPUs drain, then the bias selection ───────────────────────────────────────
if (( ${#worker_pids[@]} )); then
    echo "[$(date)] waiting for the GPU queue to drain ($(wc -l < "${BOX_GPU_QUEUE}") task(s) left)"
    wait "${worker_pids[@]}"
    echo "[$(date)] GPU work done"
    for name in $(awk -F'\t' '$2 ~ /^03\.0\.train_bias_model/ {print $1}' "${BOX_STATUS}" | sort -u); do
        DATASET_CONFIG="${configs_dir}/${name}/config.yaml" \
            bash "${steps_dir}/03.1.select_bias.sh" > "${BOX_LOG_DIR}/${name}/03.1.select_bias.log" 2>&1
        printf '%s\t%s\t%d\t%d\n' "${name}" "03.1.select_bias" "$?" 0 >> "${BOX_STATUS}"
    done
fi

[[ -n "${checks_pid}" ]] && wait "${checks_pid}"

# ── summary ───────────────────────────────────────────────────────────────────
failed="$(awk -F'\t' '$3 != 0' "${BOX_STATUS}")"
echo ""
echo "steps run: $(wc -l < "${BOX_STATUS}"), failed: $(printf '%s' "${failed}" | grep -c .)"
[[ -n "${failed}" ]] && printf 'FAILED:\n%s\n' "${failed}"
echo ""
echo "checks:"
cat "${BOX_DIR}/checks/summary.tsv" 2>/dev/null
[[ -z "${failed}" ]]
