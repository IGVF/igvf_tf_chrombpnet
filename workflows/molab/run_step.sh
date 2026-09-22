#!/bin/bash
# run_step.sh
# Purpose: Run one workflows/SLURM step on a molab box, in the environment that
#   step actually needs, without SLURM and without conda.
#
# Two things stand in for the cluster:
#
#   sbatch     -> this script. `--array` is emulated by looping
#                 SLURM_ARRAY_TASK_ID over the requested indices, in sequence
#                 (one GPU, so parallelism would only contend).
#   conda      -> either the pixi `preprocess` environment or the chrombpnet
#                 container, chosen per step by the table below. That table is
#                 derived from which env each step's `activate_env` call names:
#                 `preprocess_conda` -> pixi, `CONDA_ENV` -> container.
#
# Input:  a step filename under workflows/SLURM/
# Output: the step's own outputs, plus a log per array index under ${MOLAB_LOG_DIR}
# Usage:
#   source workflows/molab/env.sh
#   bash workflows/molab/run_step.sh 00.0.prepare_signal.sh
#   bash workflows/molab/run_step.sh --array 0-3 03.0.train_bias_model.sh
#   bash workflows/molab/run_step.sh --array 0,2 03.2.qc_selected_bias.sh
#   bash workflows/molab/run_step.sh --dry-run 04.0.train_full_model.sh
# Prerequisites: setup_molab.sh has run; env.sh is sourced.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workflows/molab/env.sh
source "${SCRIPT_DIR}/env.sh"

STEPS_DIR="${REPO_ROOT}/workflows/SLURM"

# Which environment each step needs. Steps not listed default to the container,
# which is the larger of the two and has chrombpnet in it.
step_env() {
    case "$1" in
        00.0.prepare_signal.sh|00.1.preprocess_peaks.sh|02.0.qc_signal_peaks.sh) echo pixi ;;
        *) echo container ;;
    esac
}

usage() { sed -n '2,26p' "$0"; exit "${1:-0}"; }

# --list: every step in execution order, the environment it runs in, and
# whether it has already produced its marker output. The step files live in
# workflows/SLURM/ and are NOT duplicated here -- that directory stays the one
# definition of what each step does; molab only changes how it is launched.
list_steps() {
    printf "%-34s %-10s %s\n" "STEP" "ENV" "ARRAY"
    for f in "${STEPS_DIR}"/[0-9]*.sh; do
        b="$(basename "${f}")"
        arr="$(grep -oE '^#SBATCH --array=[0-9,-]+' "${f}" | head -1 | sed 's/.*=//')"
        printf "%-34s %-10s %s\n" "${b}" "$(step_env "${b}")" "${arr:-—}"
    done
    echo
    echo "Run one with:   bash workflows/molab/run_step.sh [--array N-M] <step>"
    echo "Run the lot:    bash workflows/molab/run_all.sh"
}

array_spec=""
dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --array)   array_spec="$2"; shift 2 ;;
        --array=*) array_spec="${1#*=}"; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage 0 ;;
        --list)    list_steps; exit 0 ;;
        -*) echo "ERROR: unknown option $1" >&2; usage 1 ;;
        *) break ;;
    esac
done

step="${1:-}"
[[ -n "${step}" ]] || { echo "ERROR: no step given" >&2; usage 1; }
step="$(basename "${step}")"
[[ -f "${STEPS_DIR}/${step}" ]] || { echo "ERROR: no such step: ${STEPS_DIR}/${step}" >&2; exit 1; }
shift || true

# Expand an sbatch-style array spec: "0-3", "0,2,4", "7", or empty -> just 0.
expand_array() {
    local spec="$1" part lo hi out=()
    [[ -z "${spec}" ]] && { echo 0; return; }
    IFS=',' read -ra parts <<< "${spec}"
    for part in "${parts[@]}"; do
        if [[ "${part}" == *-* ]]; then
            lo="${part%-*}"; hi="${part#*-}"
            for ((i = lo; i <= hi; i++)); do out+=("$i"); done
        else
            out+=("${part}")
        fi
    done
    echo "${out[@]}"
}

env_kind="$(step_env "${step}")"
mkdir -p "${MOLAB_LOG_DIR}"

# Variables the step needs that must survive into the container. Apptainer only
# forwards names prefixed APPTAINERENV_, so they are set per-invocation below.
container_exec() {
    local idx="$1"; shift
    APPTAINERENV_SLURM_ARRAY_TASK_ID="${idx}" \
    APPTAINERENV_SLURM_CPUS_PER_TASK="${MOLAB_CPUS}" \
    APPTAINERENV_DATASET="${DATASET}" \
    APPTAINERENV_REPO_ROOT="${REPO_ROOT}" \
    APPTAINERENV_REFERENCE_ROOT="${REFERENCE_ROOT}" \
    APPTAINERENV_DATASET_ROOT="${DATASET_ROOT}" \
    APPTAINERENV_CONDA_INIT="" \
    APPTAINERENV_LOG_LEVEL="${LOG_LEVEL}" \
    APPTAINERENV_OMP_NUM_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_OPENBLAS_NUM_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_MKL_NUM_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_NUMEXPR_NUM_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_TF_NUM_INTRAOP_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_TF_NUM_INTEROP_THREADS=1 \
    APPTAINERENV_TF_FORCE_GPU_ALLOW_GROWTH=true \
    APPTAINERENV_CUDA_VISIBLE_DEVICES=0 \
    APPTAINERENV_PYTHONUNBUFFERED=1 \
    APPTAINERENV_MPLCONFIGDIR=/mplconfig \
    APPTAINERENV_TF_CPP_MIN_LOG_LEVEL=2 \
    APPTAINERENV_CUDA_CACHE_DISABLE=0 \
    APPTAINERENV_CUDA_CACHE_PATH=/cudacache \
    APPTAINERENV_CUDA_CACHE_MAXSIZE=4294967296 \
    apptainer exec --nv \
        --bind "${MOLAB_BIND}:${MOLAB_BIND}" \
        --bind "${MOLAB_CUDA_CACHE}:/cudacache" \
        --bind "${MOLAB_MPLCONFIG}:/mplconfig" \
        "${MOLAB_SANDBOX}" "$@"
}

overall=0
for idx in $(expand_array "${array_spec}"); do
    log="${MOLAB_LOG_DIR}/${step%.sh}.task${idx}.log"
    echo "=== [$(date '+%F %T')] ${step} (${env_kind}, array index ${idx}) -> ${log}"

    if [[ "${dry_run}" == "1" ]]; then
        echo "    (dry run, not executing)"
        continue
    fi

    if [[ "${env_kind}" == "pixi" ]]; then
        ( cd "${REPO_ROOT}" && \
          SLURM_ARRAY_TASK_ID="${idx}" SLURM_SUBMIT_DIR="${STEPS_DIR}" \
          SLURM_CPUS_PER_TASK="${MOLAB_CPUS}" \
          OMP_NUM_THREADS="${MOLAB_CPUS}" \
          pixi run -e preprocess bash "${STEPS_DIR}/${step}" "$@" ) 2>&1 | tee "${log}"
    else
        container_exec "${idx}" bash -c \
            "cd '${STEPS_DIR}' && SLURM_SUBMIT_DIR='${STEPS_DIR}' bash '${STEPS_DIR}/${step}' $*" \
            2>&1 | grep --line-buffered -vE 'INFO: +underlay of|starship: command not found' | tee "${log}"
    fi

    rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then
        echo "=== FAILED: ${step} index ${idx} (exit ${rc}) -- see ${log}" >&2
        overall=${rc}
        break
    fi
    echo "=== OK: ${step} index ${idx}"
done
exit "${overall}"
