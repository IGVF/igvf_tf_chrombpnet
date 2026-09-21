#!/bin/bash
#SBATCH --job-name=avg_contribs
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# =============================================================================
# 06.0.average_contrib_scores.sh
# Purpose: Average DeepLIFT contribution scores across all 5 folds for
#          each dataset. One SLURM array job per dataset.
#
# Why: Each fold's model was trained on a different 80/20 data split, so
#      its contribution scores carry fold-specific noise. Averaging reduces
#      this noise and gives a more robust signal for motif discovery.
#      This follows the Greenleaf lab approach (their step 05-average_deepshaps).
#
# score_types below controls which contribution head(s) get (re-)averaged.
#   Counts was already run previously; it is left out here so reruns of
#   this script do not touch counts.h5. average_contrib_scores.py also
#   skips a score-type if its output already exists, so re-adding "counts"
#   to score_types later is safe and will not recompute it.
#
# Input:  ${full_model_dir}/{dataset}_{peak_type}_fold_{0..4}/interpretation/
#             interpretation.{score_type}_scores.h5  (step 05 output)
# Output: ${averaged_dir}/{dataset}/{dataset}_average_shaps.{score_type}.h5
#
# Usage:
#   export DATASET_DIR=/path/to/igvf_tf_collab/<dataset>
#   sbatch 06.0.average_contrib_scores.sh            # all datasets (array 0-4)
#   sbatch --array=0 06.0.average_contrib_scores.sh  # dataset 0 only
#
# Prerequisites: 05.0.get_contrib_scores.sh must have completed for all folds.
# =============================================================================

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

score_types=("profile")  # counts already done; add "counts" here to redo/extend

out_dir="${averaged_dir}/${dataset}"
mkdir -p "${out_dir}" "${log_dir}"

activate_env "${CONDA_ENV}"

metadata_start "06.0.average_contrib_scores"
metadata_inputs+=( "full_models=${full_model_dir_selected}" )
metadata_outputs+=( "averaged=${averaged_dir}/${dataset}/${dataset}_average_shaps.counts.h5" )
metadata_params+=( "dataset=${dataset}" )


for score_type in "${score_types[@]}"; do
    echo "[$(date)] [${dataset} ${score_type}] Averaging contribution scores..."
    python "${src_dir}/average_contrib_scores.py" \
        --dataset "${dataset}" \
        --folds "${folds[@]}" \
        --full-model-dir "${full_model_dir_selected}" \
        --peak-type "${peak_type}" \
        --score-type "${score_type}" \
        --out-dir "${out_dir}"
done

echo "Done"
