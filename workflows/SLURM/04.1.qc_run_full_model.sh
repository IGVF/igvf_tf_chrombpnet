#!/bin/bash
# Run after step 04.0 (train_full_model.sh) to evaluate full model performance.
#SBATCH --job-name=model_qc
#SBATCH --mem=32G
#SBATCH --time=2:00:00
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
# shellcheck source=lib/bash/config.sh
source "${REPO_ROOT}/lib/bash/config.sh" || exit 1

activate_env "${CONDA_ENV}"

metadata_start "04.1.qc_run_full_model"
metadata_inputs+=( "full_models=${full_model_dir}" )
metadata_outputs+=( "metrics=${results_path}/plots/full_model_qc/model_metrics.tsv" )
metadata_params+=( "peak_type=${peak_type}" )


python "${src_dir}/qc_full_model.py" \
    --full-model-dir "${full_model_dir}" \
    --data-path "${data_path}" \
    --datasets "${datasets[@]}" \
    --folds "${folds[@]}" \
    --peak-type "${peak_type}" \
    --out-dir "${results_path}/plots/full_model_qc" \
    --save-plots
