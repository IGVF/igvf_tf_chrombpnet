#!/bin/bash
#SBATCH --job-name=avg_contribs_bw
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 07.0.contribs_to_bigwig.sh
# Purpose: Convert the fold-averaged contribution score H5 (from step 06)
#          into a bigwig for each dataset. One SLURM array job per dataset.
#
# score_types below mirrors step 06: only "profile" is (re-)generated here
# since a counts bigwig already exists; add "counts" back to regenerate it.
#
# Input:  {averaged_dir}/{dataset}/{dataset}_average_shaps.{score_type}.h5  (step 06 output)
#         {full_model_dir}/{dataset}_all_fold_0/interpretation/
#             interpretation.interpreted_regions.bed          (any fold)
# Output: {averaged_dir}/{dataset}/{dataset}_average_shaps.{score_type}.bw
#
# Usage:
#   export DATASET_DIR=/path/to/igvf_tf_collab/<dataset>
#   sbatch 07.0.contribs_to_bigwig.sh            # all datasets (array 0-4)
#   sbatch --array=0 07.0.contribs_to_bigwig.sh  # d0 only
#
# Prerequisites: 06.0.average_contrib_scores.sh must have completed.

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

dataset="${datasets[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${dataset}" ]] && { echo "No dataset at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

score_types=("profile")  # counts bigwig already exists; add "counts" here to redo/extend


metadata_start "07.0.contribs_to_bigwig"


# Use fold 0's interpreted regions — identical across folds (same peaks input)
regions_file="${full_model_dir}/${dataset}_${peak_type}_fold_0/interpretation/interpretation.interpreted_regions.bed"


metadata_inputs+=( "regions=${regions_file}" )
require_input "${regions_file}" 05.0.get_contrib_scores.sh
preflight_check

activate_env "${CONDA_ENV}"
metadata_tools+=( "$(tool_version chrombpnet chrombpnet --version)" )
metadata_params+=( "dataset=${dataset}" )
if [[ ! -f "${regions_file}" ]]; then
    echo "[${dataset}] Regions BED not found: ${regions_file}" >&2
    exit 1
fi

for score_type in "${score_types[@]}"; do
    h5_file="${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.h5"
    out_bw="${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.bw"

    if [[ ! -f "${h5_file}" ]]; then
        echo "[${dataset} ${score_type}] Averaged H5 not found: ${h5_file}" >&2
        echo "  Run 06.0.average_contrib_scores.sh first." >&2
        exit 1
    fi

    echo "[$(date)] [${dataset} ${score_type}] Writing averaged contribution score bigwig..."

    python3 "${src_dir}/contribs_to_bigwig.py" \
        --h5 "${h5_file}" \
        --regions "${regions_file}" \
        --chrom-sizes "${chrom_sizes}" \
        --output-bw "${out_bw}"

    echo "[$(date)] [${dataset} ${score_type}] Done: ${out_bw}"
done
