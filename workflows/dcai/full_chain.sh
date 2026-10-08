#!/bin/bash
# full_chain.sh
# Purpose: one dataset from its selected bias models to TF-MoDISco on the
#   fold-averaged scores, inside a running box: 04.0 through 08.0.
#
# The box's GPU workers only run 03.0, so this runs beside them (`srun --jobid
# <box> --overlap`). Every step file is run unchanged, as run_chain.sh does,
# with SLURM_ARRAY_TASK_ID set to what that step's array index means (a fold
# for 04.0/04.4/04.5/05.0, the dataset for 04.3/06.0/07.0/08.0, the only one
# in a generated config).
#
#   per fold, on its own GPU:  04.0 -> 05.0 -> 04.4, and 04.5 (CPU) after 04.4
#     (04.0 only, for a dataset outside the attribution list below).
#     05.0 goes before 04.4 because 06.0-08.0 wait on it and 08.0 is the long
#     pole; 04.4/04.5 are per-fold QC that nothing downstream reads.
#   once every fold's 04.0 is done:  04.1 (CPU), 04.3 (GPU of the first fold).
#   once every fold's 05.0 is done:  06.0 -> 07.0 and 08.0 (CPU).
#
# Keeping the box: the box's workers exit once their queue is empty and
# <box>/.chains_done exists, and the box then finishes. This removes that file,
# holds the box with <box>/.hold.<dataset> while it runs, and on exit puts
# .chains_done back if no other .hold* file is left and bias_qc_box.py is not
# running (that one releases the box itself when it finishes).
#
# GPUs: each GPU step (04.0, 05.0, 04.4, 04.3) runs on the GPU with the fewest
#   such steps at the moment it starts, not on a fixed GPU per fold: chains of
#   different lengths left one GPU idle while others ran two (box 517917). The
#   pick is made under a lock, and the step holds a claim file in the box's
#   scratch while it runs; GPU steps without a claim (started by an older
#   full_chain.sh, or by hand) are counted from their CUDA_VISIBLE_DEVICES. The
#   GPU list argument is still accepted; it no longer pins anything.
#
# Threads: every step runs under a CPU mask (taskset) of as many cores as its
#   threads -- 16 for a GPU step. XLA sizes its thread pools to the cores a
#   process may use, and on a 224-core node one unmasked JAX process holds ~750
#   threads; with the box's 40 trainings at ~445 each, five unmasked 04.0 runs
#   hit the per-user limit (ulimit -u 32768, threads included) and died
#   creating threads (EAGAIN). Masked to 16 cores the same process holds 75.
#   The limit is not raised on purpose: the box's own trainings keep it, so
#   going past it here would make THEIR next start fail instead.
#
# Attribution scope: when <box>/attribution_datasets.txt exists (or the file
#   CHAIN_ATTRIBUTION_LIST names), only the datasets it lists (one dataset_name
#   per line) run the attribution half of the chain -- 05.0 -> 06.0 -> 07.0/08.0,
#   04.3's prediction tracks and the per-fold interpretation QC 04.4/04.5. Every
#   other dataset trains its full models (04.0) and gets 04.1's QC, and stops:
#   contribution scores on every peak cost about as much GPU time as training
#   itself, so a campaign can train every dataset and interpret a chosen few.
#   With no list, every dataset runs the whole chain, as before.
#
# Steps skip finished outputs, so a rerun resumes.
#
# Input:  $1 = the dataset's config.yaml; $2 = the GPUs for its folds, in fold
#         order, comma-separated (e.g. 0,1,2,3,4); BOX_DIR; the environment the
#         box ran 03.0 with (workflows/dcai/env.sh, CHROMBPNET_REPO,
#         CHROMBPNET_PIXI_ENV, BIAS_FACTORS_FROM_SCAN).
#         CHAIN_MODISCO_THREADS (default 32) for 04.5, CHAIN_08_THREADS
#         (default 64) for 08.0.
# Output: the steps' outputs; <box>/full_chain.tsv (dataset, step, exit,
#         seconds); logs under <box>/logs/<dataset>/<step>[.<fold>].log.
# Usage:  srun --jobid <box> --overlap -N1 -n1 bash workflows/dcai/full_chain.sh \
#             <config.yaml> 0,1,2,3,4

set -uo pipefail
config="${1:?usage: full_chain.sh <config.yaml> <gpu,gpu,...>}"
IFS=, read -r -a gpus <<< "${2:?usage: full_chain.sh <config.yaml> <gpu,gpu,...>}"
: "${BOX_DIR:?set BOX_DIR to the directory of the running box}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT
steps_dir="${REPO_ROOT}/workflows/SLURM"

name="$(sed -n 's/^dataset_name: *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/p' "${config}")"
mapfile -t folds < <(sed -n 's/^folds: *\[\(.*\)\]/\1/p' "${config}" | tr -d '" ' | tr ',' '\n')
(( ${#gpus[@]} >= ${#folds[@]} )) || { echo "ERROR: ${#folds[@]} folds, ${#gpus[@]} GPU(s) given" >&2; exit 1; }
log_dir="${BOX_DIR}/logs/${name}"
mkdir -p "${log_dir}"
record="${BOX_DIR}/full_chain.tsv"

# Node-local temp files and the box's JAX cache, as run_box.sh sets them.
scratch="/dev/shm/${USER}/box.${SLURM_JOB_ID:-local}"
export TMPDIR="${scratch}/tmp" TMP="${scratch}/tmp" TEMP="${scratch}/tmp"
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax_cache" XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "${TMPDIR}" "${JAX_COMPILATION_CACHE_DIR}"

attribution_list="${CHAIN_ATTRIBUTION_LIST:-${BOX_DIR}/attribution_datasets.txt}"
attribution=1
if [[ -f "${attribution_list}" ]] && ! grep -qxF "${name}" "${attribution_list}"; then
    attribution=0
    echo "[$(date)] ${name}: not in ${attribution_list}; 04.0 and 04.1 only"
fi
hold="${BOX_DIR}/.hold.${name}"
touch "${hold}"
rm -f "${BOX_DIR}/.chains_done"
release() {
    rm -f "${hold}"
    if ! compgen -G "${BOX_DIR}/.hold*" > /dev/null && ! pgrep -f "bias_qc_box.py --box-dir ${BOX_DIR}" > /dev/null; then
        touch "${BOX_DIR}/.chains_done"
        echo "[$(date)] no other holder: released the box (.chains_done)"
    fi
}
trap release EXIT

# CPUs this step may use, in order; slices of them become each step's mask.
mapfile -t allowed < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
# cpu_slice <offset> <n> -- n of the allowed CPUs from offset (wrapping), comma-separated.
cpu_slice() {
    local i out=()
    for (( i = 0; i < $2; i++ )); do out+=( "${allowed[$(( ($1 + i) % ${#allowed[@]} ))]}" ); done
    local IFS=,
    echo "${out[*]}"
}

claims="${scratch}/gpu_claims"
mkdir -p "${claims}"
n_gpus_node="$(nvidia-smi -L 2>/dev/null | wc -l)"
gpu_steps_re='SLURM/(03\.0\.train_bias_model|03\.2\.qc_selected_bias|04\.0\.train_full_model|04\.3\.generate_predictions|04\.4\.qc_full_model_interpret|05\.0\.get_contrib_scores)\.sh'
# claim_gpu -- under the lock, the least-loaded GPU; prints the claim file made for it.
claim_gpu() {
    (
        flock 8
        local g p env_g best=0 file
        local -a load=()
        for (( g = 0; g < n_gpus_node; g++ )); do load[g]=0; done
        for file in "${claims}"/*.claim; do
            [[ -e "${file}" ]] || continue
            g="${file##*/}"; g="${g%%.*}"
            load[g]=$(( load[g] + 1 ))
        done
        for p in $(pgrep -f "${gpu_steps_re}"); do
            tr '\0' '\n' < "/proc/${p}/environ" 2>/dev/null | grep -qx 'GPU_CLAIM=1' && continue
            env_g="$(tr '\0' '\n' < "/proc/${p}/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')"
            [[ "${env_g}" =~ ^[0-9]+$ ]] && (( env_g < n_gpus_node )) && load[env_g]=$(( load[env_g] + 1 ))
        done
        for (( g = 1; g < n_gpus_node; g++ )); do
            (( load[g] < load[best] )) && best="${g}"
        done
        file="${claims}/${best}.${BASHPID}.${RANDOM}.claim"
        touch "${file}"
        echo "${file}"
    ) 8> "${claims}/.lock"
}

# step <script> <array index> <log> <threads> <gpu or ""> <cpu offset> -- one step, recorded.
# A non-empty GPU argument means "this step needs a GPU"; which one is claim_gpu's choice.
step() {
    local script="$1" idx="$2" log="$3" threads="$4" gpu="$5" cpus t0 rc claim=""
    cpus="$(cpu_slice "$6" "$(( threads > 16 ? threads : 16 ))")"
    if [[ -n "${gpu}" && "${n_gpus_node}" -gt 0 ]]; then
        claim="$(claim_gpu)"
        gpu="${claim##*/}"; gpu="${gpu%%.*}"
    fi
    t0="$(date +%s)"
    echo "[$(date)] ${name} ${log}: start${claim:+ on GPU ${gpu}}"
    DATASET_CONFIG="${config}" SLURM_ARRAY_TASK_ID="${idx}" SLURM_CPUS_PER_TASK="${threads}" \
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${threads}" NUMBA_NUM_THREADS="${threads}" \
        MKL_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}" GPU_CLAIM="${claim:+1}" \
        taskset -c "${cpus}" bash "${steps_dir}/${script}.sh" > "${log_dir}/${log}.log" 2>&1
    rc=$?
    [[ -n "${claim}" ]] && rm -f "${claim}"
    printf '%s\t%s\t%d\t%d\n' "${name}" "${log}" "${rc}" "$(( $(date +%s) - t0 ))" >> "${record}"
    echo "[$(date)] ${name} ${log}: exit ${rc}"
    return "${rc}"
}

# Per fold: 04.0 -> 05.0 -> 04.4 on its GPU, 04.5 after 04.4.
fold_chain() {
    local i="$1" f="${folds[$1]}" g="${gpus[$1]}"
    local o=$(( 16 * $1 ))
    step 04.0.train_full_model "${i}" "04.0.${f}" 8 "${g}" "${o}" || return 1
    touch "${scratch}/${name}.04.0.${f}.ok"
    (( attribution )) || return 0
    step 05.0.get_contrib_scores "${i}" "05.0.${f}" 8 "${g}" "${o}" || return 1
    touch "${scratch}/${name}.05.0.${f}.ok"
    step 04.4.qc_full_model_interpret "${i}" "04.4.${f}" 8 "${g}" "${o}" \
        && step 04.5.modisco_full_model "${i}" "04.5.${f}" "${CHAIN_MODISCO_THREADS:-32}" "" "$(( 96 + 32 * $1 ))"
}
pids=()
for i in "${!folds[@]}"; do
    fold_chain "${i}" &
    pids+=( $! )
done

# Dataset-level steps, each once every fold has passed what it needs.
all_ok() { local f; for f in "${folds[@]}"; do [[ -f "${scratch}/${name}.$1.${f}.ok" ]] || return 1; done; }
fold_chains_running() { local p; for p in "${pids[@]}"; do kill -0 "${p}" 2> /dev/null && return 0; done; return 1; }

until all_ok 04.0 || ! fold_chains_running; do sleep 60; done
if all_ok 04.0; then
    step 04.1.qc_run_full_model 0 04.1 8 "" 80 &
    if (( attribution )); then
        step 04.3.generate_predictions 0 04.3 8 "${gpus[0]}" 96 &
    fi
fi
if (( attribution )); then
    until all_ok 05.0 || ! fold_chains_running; do sleep 60; done
    if all_ok 05.0; then
        step 06.0.average_contrib_scores 0 06.0 16 "" 0 \
            && { step 07.0.contribs_to_bigwig 0 07.0 16 "" 16 & step 08.0.run_modisco 0 08.0 "${CHAIN_08_THREADS:-64}" "" 32; wait; }
    else
        echo "[$(date)] ${name}: a fold did not reach 05.0; 06.0-08.0 not run" >&2
    fi
fi
wait
echo "[$(date)] ${name}: chain finished"
