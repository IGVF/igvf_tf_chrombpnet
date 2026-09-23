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
# Steps already done are skipped, and results never live only on this box.
# A molab session that dies comes back as a fresh box with part of /marimo,
# so, when GCP_BUCKET/GCP_SA_JSON are set:
#
#   before     sync_to_gcs.sh --restore pulls back any result missing here
#              (never overwriting a local file);
#   per index  step_done.py skips it if a run-metadata record says it ran ok
#              AND every output that record declared is still on disk with
#              its recorded md5 -- the record, not the files alone, because a
#              killed run leaves outputs but no record;
#   after      sync_to_gcs.sh uploads what the index produced, so the next
#              death costs at most the index that was running.
#
# --force runs anyway (a record cannot tell that the config changed since);
# --no-bucket skips the restore and the uploads, not the skip check.
#
# Input:  a step filename under workflows/SLURM/
# Output: the step's own outputs, plus a log per array index under ${MOLAB_LOG_DIR}
# Usage:
#   source workflows/molab/env.sh
#   bash workflows/molab/run_step.sh 00.0.prepare_signal.sh
#   bash workflows/molab/run_step.sh --array 0-3 03.0.train_bias_model.sh
#   bash workflows/molab/run_step.sh --array 0,2 03.2.qc_selected_bias.sh
#   bash workflows/molab/run_step.sh --dry-run 04.0.train_full_model.sh
#   bash workflows/molab/run_step.sh --force 00.1.preprocess_peaks.sh
# Prerequisites: setup_molab.sh has run; env.sh is sourced.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workflows/molab/env.sh
source "${SCRIPT_DIR}/env.sh"

STEPS_DIR="${REPO_ROOT}/workflows/SLURM"

# A step's Python must come from its own environment, never from the shell
# that launched it. marimo's kernel -- and every terminal or subprocess it
# starts -- exports PYTHONPATH=/tmp/uv-venv/lib/python3.13/site-packages, and
# Apptainer binds /tmp and passes the host environment through, so the
# container's python 3.8 imported the notebook's numpy 2.x (built for 3.13)
# and 01.0 died in `import pandas`. pixi's python 3.13 would not even fail:
# it would silently prefer the notebook's packages over its own lock.
unset PYTHONPATH PYTHONHOME PYTHONSAFEPATH VIRTUAL_ENV

# Which environment each step needs. Steps not listed default to the container,
# which is the larger of the two and has chrombpnet in it.
step_env() {
    case "$1" in
        00.0.prepare_signal.sh|00.1.preprocess_peaks.sh|02.0.qc_training_data.sh) echo pixi ;;
        *) echo container ;;
    esac
}

usage() { sed -n '2,41p' "$0"; exit "${1:-0}"; }

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
force=0
use_bucket=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --array)   array_spec="$2"; shift 2 ;;
        --array=*) array_spec="${1#*=}"; shift ;;
        --dry-run) dry_run=1; shift ;;
        --force)   force=1; shift ;;
        --no-bucket) use_bucket=0; shift ;;
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

# Where the step's run-metadata lands: lib/bash/config.sh's metadata_dir,
# from the same output_dir env.sh derives MOLAB_LOG_DIR from.
_cfg="${DATASET_CONFIG:-${REPO_ROOT}/config/${DATASET}/config.yaml}"
output_dir=$(python3 "${REPO_ROOT}/lib/python/utils/config.py" export "${_cfg}" \
    | sed -n 's/^output_dir=//p' | tr -d '"')
[[ -n "${output_dir}" ]] || { echo "ERROR: could not read output_dir from ${_cfg}" >&2; exit 1; }
metadata_dir="${METADATA_DIR:-${output_dir}/metadata}"

[[ -n "${GCP_BUCKET:-}" && -n "${GCP_SA_JSON:-}" ]] || use_bucket=0
if [[ "${use_bucket}" == "1" && "${dry_run}" == "0" ]]; then
    # Fail rather than run: without the bucket's copy, work already done
    # elsewhere would be redone. --no-bucket is the explicit way past this.
    bash "${SCRIPT_DIR}/sync_to_gcs.sh" --restore \
        || { echo "ERROR: restore from the bucket failed; rerun, or pass --no-bucket to run without it" >&2; exit 1; }
fi

# Variables the step needs that must survive into the container. Apptainer only
# forwards names prefixed APPTAINERENV_, so they are set per-invocation below.
# Every thread pool is capped at MOLAB_CPUS, because the box reports the HOST's
# cores (os.cpu_count() = 20 on a 4-CPU slice). NUMBA_NUM_THREADS is the easy
# one to miss: TF-MoDISco (03.3) runs on numba, and without it ran 24 threads
# at ~1,660% CPU on 4 cores, beside the GPU training that needs one of them.
container_exec() {
    local idx="$1"; shift
    APPTAINERENV_SLURM_ARRAY_TASK_ID="${idx}" \
    APPTAINERENV_SLURM_CPUS_PER_TASK="${MOLAB_CPUS}" \
    APPTAINERENV_DATASET="${DATASET}" \
    APPTAINERENV_REPO_ROOT="${REPO_ROOT}" \
    APPTAINERENV_BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON}" \
    APPTAINERENV_REFERENCE_ROOT="${REFERENCE_ROOT}" \
    APPTAINERENV_DATASET_ROOT="${DATASET_ROOT}" \
    APPTAINERENV_CONDA_INIT="" \
    APPTAINERENV_LOG_LEVEL="${LOG_LEVEL}" \
    APPTAINERENV_OMP_NUM_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_OPENBLAS_NUM_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_MKL_NUM_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_NUMEXPR_NUM_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_NUMEXPR_MAX_THREADS="${MOLAB_CPUS}" \
    APPTAINERENV_NUMBA_NUM_THREADS="${MOLAB_CPUS}" \
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
sync_failed=0
for idx in $(expand_array "${array_spec}"); do
    log="${MOLAB_LOG_DIR}/${step%.sh}.task${idx}.log"
    echo "=== [$(date '+%F %T')] ${step} (${env_kind}, array index ${idx}) -> ${log}"

    if [[ "${force}" == "0" ]] && record=$(python3 "${SCRIPT_DIR}/step_done.py" "${metadata_dir}" "${step}" "${idx}"); then
        echo "=== SKIP: ${step} index ${idx} already ran ok, outputs intact (${record##*/}); --force to rerun"
        continue
    fi

    if [[ "${dry_run}" == "1" ]]; then
        echo "    (dry run, not executing)"
        continue
    fi

    if [[ "${env_kind}" == "pixi" ]]; then
        ( cd "${REPO_ROOT}" && \
          SLURM_ARRAY_TASK_ID="${idx}" SLURM_SUBMIT_DIR="${STEPS_DIR}" \
          SLURM_CPUS_PER_TASK="${MOLAB_CPUS}" \
          OMP_NUM_THREADS="${MOLAB_CPUS}" \
          NUMEXPR_NUM_THREADS="${MOLAB_CPUS}" NUMEXPR_MAX_THREADS="${MOLAB_CPUS}" \
          NUMBA_NUM_THREADS="${MOLAB_CPUS}" \
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

    if [[ "${use_bucket}" == "1" ]]; then
        bash "${SCRIPT_DIR}/sync_to_gcs.sh" \
            || { echo "=== SYNC FAILED after ${step} index ${idx}: results are only on this box" >&2; sync_failed=1; }
    fi
done
(( overall == 0 && sync_failed == 1 )) && overall=1
exit "${overall}"
