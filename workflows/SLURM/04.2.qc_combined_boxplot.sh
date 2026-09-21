#!/bin/bash
# shellcheck disable=SC2218  # false positive in shellcheck 0.11.0: metadata_start and
# activate_env both come from lib/bash/common.sh, sourced above.
# Plot combined full-model QC metrics for all four datasets side by side.
# Run once after 04.1.qc_run_full_model.sh has completed for all datasets.
# No DATASET selection needed: it discovers datasets from config/ and combines
# whichever have 04.1 output.

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

# Cross-dataset step: the root is DATASET_ROOT (config/README.md), and the
# datasets are whatever configs exist under config/ -- not a hardcoded list that
# silently goes stale when a dataset is added or renamed.
core_path="${DATASET_ROOT:-${REPO_ROOT}}"

combined_datasets=()
for _cfg in "${REPO_ROOT}"/config/*/config.yaml; do
    [[ -f "${_cfg}" ]] || continue
    _name="$(basename "$(dirname "${_cfg}")")"
    [[ "${_name}" == "example_dataset" ]] && continue     # the template, not a dataset
    # Only include datasets that actually have per-dataset QC output to combine.
    [[ -f "${core_path}/${_name}/results/plots/full_model_qc/model_metrics.tsv" ]] \
        && combined_datasets+=( "${_name}" )
done
unset _cfg _name

if [[ ${#combined_datasets[@]} -eq 0 ]]; then
    echo "ERROR: no dataset under ${core_path} has full_model_qc/model_metrics.tsv." >&2
    echo "  Run 04.1.qc_run_full_model.sh for each dataset first." >&2
    exit 1
fi

out_dir="${core_path}/results/plots/full_model_qc_combined"

metadata_start "04.2.qc_combined_boxplot"
metadata_params+=( "datasets=${combined_datasets[*]}" )
metadata_outputs+=( "metrics_plot=${out_dir}/cross_dataset_boxplot.pdf" )

activate_env "${CONDA_ENV}"

echo "[$(date)] Combining full-model QC across: ${combined_datasets[*]}"

python "${src_dir}/qc_full_model.py" \
    --combined \
    --core-path "${core_path}" \
    --datasets  "${combined_datasets[@]}" \
    --out-dir   "${out_dir}"
