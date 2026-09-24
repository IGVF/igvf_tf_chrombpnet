#!/bin/bash
# shellcheck disable=SC2218  # false positive in shellcheck 0.11.0: metadata_start and
# activate_env both come from lib/bash/common.sh, sourced above.
# Plot combined full-model QC metrics for every dataset side by side.
# Run once after 04.1.qc_run_full_model.sh has completed for the datasets.
# No DATASET selection needed: it discovers the configs under config/ (plus
# DATASET_CONFIG when set), resolves each one's results through config.sh, and
# combines whichever have 04.1 output.
# CPU only and small: src/qc_full_model.py --combined reads the model_metrics.tsv
# tables and needs only pandas, numpy, matplotlib and scipy, which the
# chrombpnet env it activates provides.

#SBATCH --job-name=model_qc_combined
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --partition=engreitz,normal
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# --- bootstrap: locate the repo root (identical block in every workflow step) --
# sbatch copies the submitted script to a node-local spool dir, so BASH_SOURCE
# does not point into the repo; SLURM_SUBMIT_DIR (the cwd at submit time) does.
# Walk up from whichever is usable until lib/bash/common.sh turns up. Override
# with REPO_ROOT if you submit from outside the checkout.
# An inherited REPO_ROOT is only trusted if it really is this repo -- the name
# is generic enough that another project may have exported it.
[[ -n "${REPO_ROOT}" && -f "${REPO_ROOT}/lib/bash/common.sh" ]] || REPO_ROOT=""
if [[ -z "${REPO_ROOT}" ]]; then
    for _d in "${SLURM_SUBMIT_DIR}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
        while [[ -n "${_d}" && "${_d}" != "/" ]]; do
            [[ -f "${_d}/lib/bash/common.sh" ]] && { REPO_ROOT="${_d}"; break 2; }
            _d="$(dirname "${_d}")"
        done
    done
fi
[[ -n "${REPO_ROOT}" ]] || {
    echo "ERROR: cannot locate the repo root. Submit from inside the checkout, or:" >&2
    echo "  export REPO_ROOT=/path/to/this/checkout" >&2
    exit 1
}
export REPO_ROOT
# --- end bootstrap -------------------------------------------------------------

# Cross-dataset step: no DATASET_DIR, so source common.sh directly rather than
# config.sh.
# shellcheck source=lib/bash/common.sh
source "${REPO_ROOT}/lib/bash/common.sh" || exit 1

# Cross-dataset step. The datasets are the configs that exist -- every
# config/*/config.yaml in the repo, plus DATASET_CONFIG when set (a dataset whose
# config lives with its data, like molab's d0) -- not a hardcoded list that goes
# stale. WHERE each keeps its results is its own config's business (output_dir
# can be anywhere), so each is resolved by sourcing lib/bash/config.sh for it in
# a subshell: the same derivation every per-dataset step uses, not a guess at
# <root>/<name>/results.
core_path="${DATASET_ROOT:-${REPO_ROOT}}"

_configs=()
for _cfg in "${REPO_ROOT}"/config/*/config.yaml; do
    [[ -f "${_cfg}" ]] || continue
    [[ "$(basename "$(dirname "${_cfg}")")" == "example_dataset" ]] && continue   # the template
    _configs+=( "${_cfg}" )
done
[[ -n "${DATASET_CONFIG:-}" && -f "${DATASET_CONFIG}" ]] && _configs+=( "${DATASET_CONFIG}" )

combined_datasets=()
metrics_args=()
declare -A _seen=()
for _cfg in "${_configs[@]}"; do
    # shellcheck disable=SC2016  # expanded by the subshell, after config.sh defines them
    _resolved="$(DATASET_CONFIG="${_cfg}" bash -c 'source "$1/lib/bash/config.sh" >/dev/null 2>&1 && printf "%s\t%s\n" "${dataset_name}" "${results_path}"' _ "${REPO_ROOT}")" \
        || { echo "[WARN] cannot resolve ${_cfg}; skipped" >&2; continue; }
    _name="${_resolved%%$'\t'*}"; _results="${_resolved#*$'\t'}"
    [[ -n "${_seen[${_name}]:-}" ]] && continue
    _seen["${_name}"]=1
    _tsv="${_results}/plots/full_model_qc/model_metrics.tsv"
    # Only datasets that actually have 04.1 output to combine.
    if [[ -f "${_tsv}" ]]; then
        combined_datasets+=( "${_name}" )
        metrics_args+=( "${_name}=${_tsv}" )
    fi
done
unset _cfg _configs _resolved _name _results _tsv _seen

if [[ ${#combined_datasets[@]} -eq 0 ]]; then
    echo "ERROR: no dataset config resolves to a results tree with full_model_qc/model_metrics.tsv." >&2
    echo "  Run 04.1.qc_run_full_model.sh for each dataset first." >&2
    exit 1
fi

out_dir="${core_path}/results/plots/full_model_qc_combined"

metadata_start "04.2.qc_combined_boxplot"
metadata_params+=( "datasets=${combined_datasets[*]}" )
# The names qc_full_model.py --combined actually writes (this used to declare
# cross_dataset_boxplot.pdf, which nothing writes, so every run recorded its
# output as missing).
metadata_outputs+=( "metrics=${out_dir}/combined_model_metrics.tsv" )
metadata_outputs+=( "plot=${out_dir}/combined_performance_boxplot.pdf" )
for _m in "${metrics_args[@]}"; do metadata_inputs+=( "metrics=${_m#*=}" ); done

activate_env "${chrombpnet_env}"

echo "[$(date)] Combining full-model QC across: ${combined_datasets[*]}"

python "${src_dir}/qc_full_model.py" \
    --combined \
    --metrics   "${metrics_args[@]}" \
    --out-dir   "${out_dir}"
