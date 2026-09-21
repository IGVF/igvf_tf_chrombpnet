#!/bin/bash
# shellcheck disable=SC2034  # everything here is consumed by the steps that source it
# lib/bash/common.sh
# Shared settings and helpers that do NOT depend on a dataset.
#
# Source this directly from cross-dataset steps (09, 04.2.qc_combined_boxplot,
# qc_datasets) that have no DATASET_DIR. Per-dataset steps source
# lib/bash/config.sh instead, which sources this file first.
#
# Requires REPO_ROOT to be set by the caller's bootstrap block.
#
# This file is sourced, never executed: it uses `return`, not `exit`, so a
# failure in an interactive shell does not kill the shell.

if [[ -z "${REPO_ROOT}" ]]; then
    echo "ERROR: REPO_ROOT is not set. Source this from a script with the standard bootstrap block." >&2
    return 1
fi

# ── Machine-specific values ───────────────────────────────────────────────────
# There is no second config file for these. They are the same for every dataset
# and must not be committed, so they are environment variables with defaults —
# export them in your shell profile, or in the conda env's activate.d. The
# defaults below are the Engreitz-lab Sherlock install.

# ── Cluster software ──────────────────────────────────────────────────────────
# Recreate the envs from the pinned specs: conda env create -f envs/<name>.yml
CONDA_INIT="${CONDA_INIT:-/home/groups/engreitz/Software/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CHROMBPNET_ENV:-/home/groups/engreitz/Users/opushkar/.conda/envs/chrombpnet}"
finemo_conda="${FINEMO_ENV:-/home/groups/engreitz/Users/opushkar/.conda/envs/finemo}"
motif_compendium_conda="${MOTIF_COMPENDIUM_ENV:-/home/groups/engreitz/Users/opushkar/.conda/envs/motif_compendium}"
# Steps 00/01 and the reference builder: pyranges1 needs Python >= 3.12, which
# none of the three envs above have. Create it with:
#   conda env create -f envs/preprocess.yml
# Path is a placeholder in the same location as the others until it exists.
preprocess_conda="${PREPROCESS_ENV:-/home/groups/engreitz/Users/opushkar/.conda/envs/preprocess}"

# Default root for dataset data/results; config/site.sh usually repoints this
# at a shared collaboration tree.
# Where dataset data/ and results/ live. Defaults to the checkout, which is
# fine for one dataset but not for a shared collaboration tree.
data_root="${DATASET_ROOT:-${REPO_ROOT}}"

# ── Shared references ─────────────────────────────────────────────────────────
# Genome, chrom.sizes, blacklist and the motif DB all come from one place, shared
# with `cli.py download-references` so the writer and the readers cannot
# disagree. Override the root with REFERENCE_ROOT; see that file.
# shellcheck source=./references.sh
source "${REPO_ROOT}/lib/bash/references.sh" || return 1

# ── Algorithm parameters ──────────────────────────────────────────────────────
finemo_alpha="0.8" # Fi-NeMo hit-calling threshold (lower = more hits)
motif_compendium_threshold="0.95" # Leiden clustering similarity cutoff

# ── Repo layout ───────────────────────────────────────────────────────────────
src_dir="${REPO_ROOT}/src"          # atomic Python scripts the workflows call
folds_dir="${REPO_ROOT}/folds"      # cross-validation splits shipped with the repo

# Run-metadata output. config.sh repoints this at ${results_path}/metadata for
# per-dataset steps; cross-dataset steps keep this collaboration-root default.
metadata_dir="${METADATA_DIR:-${REPO_ROOT}/results/metadata}"

# ── Helpers ───────────────────────────────────────────────────────────────────
#
# Only the four below exist, because only they are used. The steps write their
# own progress lines with `echo "[$(date)] ..."` and their own input checks with
# `[[ -f ... ]] || { echo ...; exit 1; }`. Adding a log()/require_file() here
# means converting those ~40 call sites in the same change, not leaving a second
# way to do it.

# activate_env <conda-env-path> — initialise conda and activate an env.
# Deliberately does NOT set -euo pipefail: conda's activation scripts are not
# written against `set -u`, which is why the scripts that do use it set it
# after this call rather than at the top of the file.
activate_env() {
    local env_path="${1:?activate_env: missing env path}"
    if [[ ! -f "${CONDA_INIT}" ]]; then
        echo "ERROR: conda init script not found: ${CONDA_INIT}" >&2
        echo "  Set CONDA_INIT in config/site.sh (see config/site.example.sh)." >&2
        exit 1
    fi
    # shellcheck disable=SC1090  # path is a cluster location, not resolvable here
    source "${CONDA_INIT}"
    conda activate "${env_path}"
}

# load_render_modules — cairo/pango, needed by the motif-report rendering
# (weasyprint/logomaker) that chrombpnet and modisco do at the end of a run.
load_render_modules() {
    # No `module` system (a laptop, a container, a non-Lmod cluster): skip
    # quietly so a step reaches its own preflight check instead of dying here.
    command -v ml >/dev/null 2>&1 || { echo "[modules] no 'ml' on PATH, skipping module load" >&2; return 0; }
    ml devel
    ml system
    ml cairo
    ml pango/1.40.10
}

# load_gpu_modules — the render stack plus CUDA, loaded by every GPU step
# before activating conda.
#
# The cuda/11.5.0 here is what the --constraint="GPU_CC:7.0|7.5|8.0|8.6" in
# 03.0 and 04.0 is pinned against: it cannot drive Ada (8.9) or Hopper (9.0).
# Steps that load this without that constraint can land on a GPU cuda 11.5
# does not support.
load_gpu_modules() {
    load_render_modules
    command -v ml >/dev/null 2>&1 || return 0
    ml cuda/11.5.0
    ml cudnn/8.6.0.163
}

# gpu_env — TensorFlow GPU settings shared by every GPU step.
gpu_env() {
    export CUDA_VISIBLE_DEVICES=0
    export TF_FORCE_GPU_ALLOW_GROWTH=true
}

# ── Run metadata ──────────────────────────────────────────────────────────────
#
# Every step emits one JSON record: inputs, outputs, md5s, tool versions, the
# git commit and a GitHub permalink to the step script. Many runs load into
# DuckDB as one table -- see workflows/README.md for the queries.
#
# In a step:
#
#     metadata_start "00.1.preprocess_peaks"   # names the step, installs the EXIT trap
#     metadata_inputs+=( "peaks=${peaks_file}" )
#     metadata_params+=( "fold=${fold}" )
#     ... do the work ...
#     metadata_outputs+=( "narrowpeak=${out}" )
#
# The trap fires on success, on failure and on SIGTERM (SLURM cancellation or
# soft preemption), so a failed run still records what it meant to produce.
# SIGKILL -- OOM, hard preemption -- leaves no record; that absence is the
# signal. Emission never changes the step's exit status.

metadata_step="${metadata_step:-}"
metadata_inputs=()
metadata_outputs=()
metadata_params=()
metadata_tools=()

# metadata_start <step-name> — record the start time and arrange for emission
# on exit. Taking the name as an argument (rather than presetting a variable)
# keeps the step scripts free of assignments shellcheck cannot see a use for.
metadata_start() {
    metadata_step="${1:?metadata_start: missing step name}"
    _metadata_started_at="$(date +%s)"
    _metadata_script="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    trap 'metadata_emit $?' EXIT
    trap 'exit 143' TERM        # turn SIGTERM into a normal exit so EXIT runs
}

# metadata_emit <exit-status> — write the record. Never fails the step.
metadata_emit() {
    local rc="${1:-0}"
    [[ -n "${metadata_step}" ]] || return 0
    [[ -n "${metadata_dir}" ]]  || return 0

    local args=(
        --step       "${metadata_step}"
        --out-dir    "${metadata_dir}"
        --script     "${_metadata_script}"
        --started-at "${_metadata_started_at:-}"
        --exit-status "${rc}"
    )
    [[ -n "${dataset:-}" ]] && args+=( --dataset "${dataset}" )

    local item
    for item in "${metadata_inputs[@]+"${metadata_inputs[@]}"}";  do args+=( --input  "${item}" ); done
    for item in "${metadata_outputs[@]+"${metadata_outputs[@]}"}"; do args+=( --output "${item}" ); done
    for item in "${metadata_params[@]+"${metadata_params[@]}"}";  do args+=( --param  "${item}" ); done
    for item in "${metadata_tools[@]+"${metadata_tools[@]}"}";    do args+=( --tool   "${item}" ); done

    python "${src_dir}/emit_metadata.py" "${args[@]}" || \
        echo "[metadata] could not write run metadata (step still exited ${rc})" >&2
    return 0
}


# ── ChromBPNet inputs ─────────────────────────────────────────────────────────

# set_signal_args — populate ${signal_args[@]} with the chrombpnet input flag,
# and ${prepared_required} when the signal can only be used via a prepared bigwig.
#
# signal_type is derived from the extension by lib/python/utils/config.py, so
# this only maps a known kind to a flag; unknown extensions fail earlier.
#
# chrombpnet's training input is a mutually exclusive, REQUIRED group:
#   -ibam/--input-bam-file  -ifrag/--input-fragment-file  -itag/--input-tagalign-file
#
# A bigwig is not in that group. We can still use one, because the pipeline
# installs a prepared bigwig and skips chrombpnet's conversion entirely — but
# the parser still demands one of the three flags, so we pass the path under
# -ifrag purely to satisfy it. That value is never read: reads_to_bigwig is
# replaced before it runs. To make sure it stays never-read,
# prepared_required=1 tells src/chrombpnet_train.py to abort rather than fall
# back to converting, which would otherwise parse a bigwig as if it were
# fragments and produce silent garbage.
signal_args=()
prepared_required=0
set_signal_args() {
    prepared_required=0
    case "${signal_type}" in
        fragments) signal_args=( -ifrag "${signal_path}" ) ;;
        bam)       signal_args=( -ibam  "${signal_path}" ) ;;
        tagalign)  signal_args=( -itag  "${signal_path}" ) ;;
        bigwig)
            signal_args=( -ifrag "${signal_path}" )   # placeholder, never read
            prepared_required=1
            ;;
        *)
            echo "ERROR: unsupported signal_type '${signal_type}'" >&2
            exit 1
            ;;
    esac
}

# ── Preconditions ─────────────────────────────────────────────────────────────
#
# A step declares what it needs and which step makes it; if anything is absent
# the run stops before doing work and prints the command to fix it.
#
#     require_input "${peaks_file}"     00.1.preprocess_peaks.sh
#     require_input "${negatives_file}" 02.0.preprocess_nonpeaks.sh
#     preflight_check
#
# All missing inputs are reported together, not one per re-run: submitting a
# GPU job to be told about one missing file, fixing it, and being told about
# the next is the failure mode this exists to avoid.

preflight_missing=()

# require_input <path> <producing-step> — record a required input.
require_input() {
    local path="${1:?require_input: missing path}" producer="${2:-}"
    [[ -e "${path}" ]] || preflight_missing+=( "${path}|${producer}" )
}

# preflight_check — report every missing input and exit 1, or return quietly.
preflight_check() {
    (( ${#preflight_missing[@]} )) || return 0

    local entry path producer producers=()
    echo "" >&2
    echo "ERROR: ${metadata_step:-this step} cannot run yet — ${#preflight_missing[@]} missing input(s):" >&2
    echo "" >&2
    for entry in "${preflight_missing[@]}"; do
        path="${entry%%|*}"
        producer="${entry##*|}"
        echo "  missing : ${path}" >&2
        if [[ -n "${producer}" ]]; then
            echo "  produced by : ${producer}" >&2
            # shellcheck disable=SC2076  # literal match is intended
            [[ " ${producers[*]-} " =~ " ${producer} " ]] || producers+=( "${producer}" )
        else
            echo "  produced by : (an external input — stage it yourself)" >&2
        fi
        echo "" >&2
    done

    if (( ${#producers[@]} )); then
        echo "Run first, in this order:" >&2
        for producer in "${producers[@]}"; do
            if [[ "${producer}" == *.sh ]]; then
                echo "  cd ${REPO_ROOT}/workflows/SLURM && DATASET=${DATASET:-<dataset>} sbatch ${producer}" >&2
            else
                echo "  ${producer}" >&2
            fi
        done
        echo "" >&2
        echo "Check overall progress with:  bash ${REPO_ROOT}/workflows/SLURM/status.sh" >&2
    fi
    exit 1
}
