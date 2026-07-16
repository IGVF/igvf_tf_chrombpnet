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
# 06.average_contrib_scores.sh
# Purpose: Average DeepLIFT contribution scores across all 5 folds for
#          each dataset. One SLURM array job per dataset.
#
# Why: Each fold's model was trained on a different 80/20 data split, so
#      its contribution scores carry fold-specific noise. Averaging reduces
#      this noise and gives a more robust signal for motif discovery.
#      This follows the Greenleaf lab approach (their step 05-average_deepshaps).
#
# Input:  ${full_model_dir}/{dataset}_{peak_type}_fold_{0..4}/interpretation/
#             interpretation.counts_scores.h5  (step 05 output)
# Output: ${averaged_dir}/{dataset}/{dataset}_average_shaps.counts.h5
#
# Usage:
#   sbatch 06.average_contrib_scores.sh            # all datasets (array 0-4)
#   sbatch --array=0 06.average_contrib_scores.sh  # dataset 0 only
#
# Prerequisites: 05.get_contrib_scores.sh must have completed for all folds.
# =============================================================================

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

dataset="${datasets[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${dataset}" ]] && { echo "No dataset at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

out_dir="${averaged_dir}/${dataset}"
mkdir -p "${out_dir}" "${log_dir}"

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/average_contrib_scores.py" \
    --dataset "${dataset}" \
    --folds "${folds[@]}" \
    --full-model-dir "${full_model_dir_selected}" \
    --peak-type "${peak_type}" \
    --out-dir "${out_dir}"

echo "Done"
