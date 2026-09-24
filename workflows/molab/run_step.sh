#!/bin/bash
# run_step.sh
# Purpose: Run one workflows/SLURM step on a molab box, without SLURM, exactly
#   as the cluster would: `bash <step>` from workflows/SLURM, with the few
#   variables sbatch would have set.
#
# What stands in for the cluster:
#
#   sbatch     -> this script. `--array` is emulated by looping
#                 SLURM_ARRAY_TASK_ID over the requested indices, in sequence
#                 (one GPU, so parallelism would only contend).
#   environment -> nothing here. Every step enters its own through
#                 activate_env (lib/bash/common.sh: `pixi shell-hook` for the
#                 pixi environments), the same call it makes on the cluster, so
#                 there is no second table of which step needs which. --list
#                 reads each step's activate_env line to show it.
#   resources  -> SLURM_CPUS_PER_TASK and every thread pool (OpenMP, OpenBLAS,
#                 MKL, numexpr, numba) are set to MOLAB_CPUS, because the box
#                 reports the HOST's cores (os.cpu_count() = 20 on a 4-CPU
#                 slice). numba is the easy one to miss: TF-MoDISco runs on it,
#                 and unpinned it ran 24 threads at ~1,660% CPU on 4 cores.
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
# --no-bucket skips the restore and the uploads, not the skip check;
# --dry-run prints the command each index would run, and runs nothing.
#
# Input:  a step filename under workflows/SLURM/
# Output: the step's own outputs, plus a log per array index under ${MOLAB_LOG_DIR}
# Usage:
#   source workflows/molab/env.sh
#   bash workflows/molab/run_step.sh --list
#   bash workflows/molab/run_step.sh 00.0.prepare_signal.sh
#   bash workflows/molab/run_step.sh --array 0-3 03.0.train_bias_model.sh
#   bash workflows/molab/run_step.sh --array 0,2 03.2.qc_selected_bias.sh
#   bash workflows/molab/run_step.sh --dry-run 04.0.train_full_model.sh
#   bash workflows/molab/run_step.sh --force 00.1.preprocess_peaks.sh
# Prerequisites: setup_molab.sh has run; env.sh is sourced.

set -uo pipefail

# A step's Python must come from its own environment, never from the shell
# that launched it. marimo's kernel -- and every terminal or subprocess it
# starts -- exports PYTHONPATH=/tmp/uv-venv/lib/python3.13/site-packages, and
# a python that honours it prefers the notebook's packages over its own lock.
# activate_env drops these too; dropping them here, before env.sh, also covers
# what runs before it (env.sh's config lookups, config.sh's bootstrap python,
# the metadata trap).
unset PYTHONPATH PYTHONHOME PYTHONSAFEPATH VIRTUAL_ENV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workflows/molab/env.sh
source "${SCRIPT_DIR}/env.sh"

STEPS_DIR="${REPO_ROOT}/workflows/SLURM"

usage() { sed -n '2,50p' "$0"; exit "${1:-0}"; }

# step_env <step> — the environment a step activates, read from its own
# `activate_env "${<name>_env}"` line (lib/bash/common.sh defines the four),
# so this cannot drift from what the step does. "—" if it activates none.
step_env() {
    local e
    e="$(sed -n 's/^activate_env "\${\([a-z_]*\)_env}".*/\1/p' "${STEPS_DIR}/$1" | head -n1)"
    echo "${e:-—}"
}

# --list: every step in execution order, the environment it runs in, and its
# own #SBATCH --array default. The step files live in workflows/SLURM/ and
# are NOT duplicated here -- that directory stays the one definition of what
# each step does; molab only changes how it is launched.
list_steps() {
    printf "%-36s %-18s %s\n" "STEP" "ENV" "ARRAY"
    for f in "${STEPS_DIR}"/[0-9]*.sh; do
        b="$(basename "${f}")"
        arr="$(grep -oE '^#SBATCH --array=[0-9,-]+' "${f}" | head -1 | sed 's/.*=//')"
        printf "%-36s %-18s %s\n" "${b}" "$(step_env "${b}")" "${arr:-—}"
    done
    echo
    echo "ENV is the \${<name>_env} each step passes to activate_env; chrombpnet is"
    echo "pixi ${CHROMBPNET_PIXI_ENV} in ${CHROMBPNET_REPO}, the rest are this repo's pixi.toml."
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
# shellcheck disable=SC2218  # defined above; shellcheck 0.11 false positive (CLAUDE.md)
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

# shellcheck disable=SC2218  # defined above; shellcheck 0.11 false positive (CLAUDE.md)
env_name="$(step_env "${step}")"
[[ "${dry_run}" == "1" ]] || mkdir -p "${MOLAB_LOG_DIR}"

# Where the step's run-metadata lands: lib/bash/config.sh's own metadata_dir,
# resolved the way the step resolves it.
# shellcheck disable=SC2016  # expanded by molab_config's child bash
metadata_dir="$(molab_config '${metadata_dir}')"
[[ -n "${metadata_dir}" ]] \
    || { echo "ERROR: could not resolve the dataset config (${DATASET_CONFIG:-config/${DATASET}/config.yaml})" >&2; exit 1; }

[[ -n "${GCP_BUCKET:-}" && -n "${GCP_SA_JSON:-}" ]] || use_bucket=0
if [[ "${use_bucket}" == "1" && "${dry_run}" == "0" ]]; then
    # Fail rather than run: without the bucket's copy, work already done
    # elsewhere would be redone. --no-bucket is the explicit way past this.
    bash "${SCRIPT_DIR}/sync_to_gcs.sh" --restore \
        || { echo "ERROR: restore from the bucket failed; rerun, or pass --no-bucket to run without it" >&2; exit 1; }
fi

overall=0
sync_failed=0
for idx in $(expand_array "${array_spec}"); do
    log="${MOLAB_LOG_DIR}/${step%.sh}.task${idx}.log"
    echo "=== [$(date '+%F %T')] ${step} (env ${env_name}, array index ${idx}) -> ${log}"

    if [[ "${force}" == "0" ]] && record=$(python3 "${SCRIPT_DIR}/step_done.py" "${metadata_dir}" "${step}" "${idx}"); then
        echo "=== SKIP: ${step} index ${idx} already ran ok, outputs intact (${record##*/}); --force to rerun"
        continue
    fi

    # What sbatch would have provided, and nothing else: the step does its own
    # environment. SLURM_SUBMIT_DIR is what the step's bootstrap block uses to
    # find REPO_ROOT.
    launch=( env
        SLURM_ARRAY_TASK_ID="${idx}"
        SLURM_SUBMIT_DIR="${STEPS_DIR}"
        SLURM_CPUS_PER_TASK="${MOLAB_CPUS}"
        OMP_NUM_THREADS="${MOLAB_CPUS}"
        OPENBLAS_NUM_THREADS="${MOLAB_CPUS}"
        MKL_NUM_THREADS="${MOLAB_CPUS}"
        NUMEXPR_NUM_THREADS="${MOLAB_CPUS}"
        NUMEXPR_MAX_THREADS="${MOLAB_CPUS}"
        NUMBA_NUM_THREADS="${MOLAB_CPUS}"
        PYTHONUNBUFFERED=1
        bash "${STEPS_DIR}/${step}" "$@" )

    if [[ "${dry_run}" == "1" ]]; then
        printf '    (dry run) in %s:' "${STEPS_DIR}"
        printf ' %q' "${launch[@]}"
        echo
        continue
    fi

    ( cd "${STEPS_DIR}" && "${launch[@]}" ) 2>&1 | tee "${log}"

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
