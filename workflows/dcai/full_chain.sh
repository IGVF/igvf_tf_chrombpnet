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
#   per fold, on its own GPU:  04.0 -> 05.0 -> 04.4, and 04.5 (CPU) after 04.4.
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

# step <script> <array index> <log> <threads> <gpu or ""> -- one step, recorded.
step() {
    local script="$1" idx="$2" log="$3" threads="$4" gpu="$5" t0 rc
    t0="$(date +%s)"
    DATASET_CONFIG="${config}" SLURM_ARRAY_TASK_ID="${idx}" SLURM_CPUS_PER_TASK="${threads}" \
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${threads}" NUMBA_NUM_THREADS="${threads}" \
        MKL_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}" \
        bash "${steps_dir}/${script}.sh" > "${log_dir}/${log}.log" 2>&1
    rc=$?
    printf '%s\t%s\t%d\t%d\n' "${name}" "${log}" "${rc}" "$(( $(date +%s) - t0 ))" >> "${record}"
    echo "[$(date)] ${name} ${log}: exit ${rc}"
    return "${rc}"
}

# Per fold: 04.0 -> 05.0 -> 04.4 on its GPU, 04.5 after 04.4.
fold_chain() {
    local i="$1" f="${folds[$1]}" g="${gpus[$1]}"
    step 04.0.train_full_model "${i}" "04.0.${f}" 8 "${g}" || return 1
    touch "${scratch}/${name}.04.0.${f}.ok"
    step 05.0.get_contrib_scores "${i}" "05.0.${f}" 8 "${g}" || return 1
    touch "${scratch}/${name}.05.0.${f}.ok"
    step 04.4.qc_full_model_interpret "${i}" "04.4.${f}" 8 "${g}" \
        && step 04.5.modisco_full_model "${i}" "04.5.${f}" "${CHAIN_MODISCO_THREADS:-32}" ""
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
    step 04.1.qc_run_full_model 0 04.1 8 "" &
    step 04.3.generate_predictions 0 04.3 8 "${gpus[0]}" &
fi
until all_ok 05.0 || ! fold_chains_running; do sleep 60; done
if all_ok 05.0; then
    step 06.0.average_contrib_scores 0 06.0 16 "" \
        && { step 07.0.contribs_to_bigwig 0 07.0 16 "" & step 08.0.run_modisco 0 08.0 "${CHAIN_08_THREADS:-64}" ""; wait; }
else
    echo "[$(date)] ${name}: a fold did not reach 05.0; 06.0-08.0 not run" >&2
fi
wait
echo "[$(date)] ${name}: chain finished"
