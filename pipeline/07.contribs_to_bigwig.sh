#!/bin/bash
#SBATCH --job-name=avg_contribs_bw
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 07.contribs_to_bigwig.sh
# Purpose: Convert the fold-averaged contribution score H5 (from step 06)
#          into a bigwig for each dataset. One SLURM array job per dataset.
#
# Input:  {averaged_dir}/{dataset}/{dataset}_average_shaps.counts.h5      (step 06 output)
#         {full_model_dir}/{dataset}_all_fold_0/interpretation/
#             interpretation.interpreted_regions.bed          (any fold)
# Output: {averaged_dir}/{dataset}/average_shaps.counts.bw
#
# Usage:
#   sbatch 07.contribs_to_bigwig.sh            # all datasets (array 0-4)
#   sbatch --array=0 07.contribs_to_bigwig.sh  # d0 only
#
# Prerequisites: 06.average_contrib_scores.sh must have completed.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

dataset="${datasets[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${dataset}" ]] && { echo "No dataset at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

h5_file="${averaged_dir}/${dataset}/${dataset}_average_shaps.counts.h5"
out_bw="${averaged_dir}/${dataset}/average_shaps.counts.bw"

if [[ ! -f "${h5_file}" ]]; then
    echo "[${dataset}] Averaged H5 not found: ${h5_file}" >&2
    echo "  Run 06.average_contrib_scores.sh first." >&2
    exit 1
fi

# Use fold 0's interpreted regions — identical across folds (same peaks input)
regions_file="${full_model_dir}/${dataset}_${peak_type}_fold_0/interpretation/interpretation.interpreted_regions.bed"

if [[ ! -f "${regions_file}" ]]; then
    echo "[${dataset}] Regions BED not found: ${regions_file}" >&2
    exit 1
fi

echo "[$(date)] [${dataset}] Writing averaged contribution score bigwig..."

python3 "${SCRIPT_DIR}/contribs_to_bigwig.py" \
    --h5          "${h5_file}" \
    --regions     "${regions_file}" \
    --chrom-sizes "${chrom_sizes}" \
    --output-bw   "${out_bw}"

echo "[$(date)] [${dataset}] Done: ${out_bw}"
