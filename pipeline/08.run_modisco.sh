#!/bin/bash
#SBATCH --job-name=modisco_avg
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=2-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 08.run_modisco.sh
# Purpose: Run TF-MoDISco on fold-averaged contribution scores (step 06).
#          One SLURM array job per dataset; produces ONE modisco result per dataset
#          (rather than one per fold), which is then used for the unified
#          motif compendium (steps 5.1–5.4).
#
# Why run on averaged scores:
#   Averaging contribution scores across folds before MoDISco improves
#   signal-to-noise ratio, so the discovered patterns are more reproducible
#   and biologically meaningful. This is the standard approach in the
#   Greenleaf lab ChromBPNet pipeline.
#
# Input:  results/contrib_scores/{dataset}/{dataset}_average_shaps.counts.h5  (step 06)
# Output: ${averaged_dir}/{dataset}/modisco/
#             modisco_counts_results.h5
#             counts_report/
#
# Usage:
#   sbatch 08.run_modisco.sh            # all datasets (array 0-4)
#   sbatch --array=0 08.run_modisco.sh  # d0 only
#
# Prerequisites: 06.average_contrib_scores.sh must have completed.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

dataset="${datasets[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${dataset}" ]] && { echo "No dataset at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

mkdir -p "${averaged_dir}/${dataset}/modisco/counts_report" "${log_dir}"

ml devel
ml system
ml cairo
ml pango/1.40.10
ml cuda/11.5.0
ml cudnn/8.6.0.163

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES=0,1
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "[$(date)] dataset ${dataset}: running MoDISco on averaged counts scores..."

modisco motifs \
    -i "${averaged_dir}/${dataset}/${dataset}_average_shaps.counts.h5" \
    -n 500000 \
    -o "${averaged_dir}/${dataset}/modisco/modisco_counts_results.h5" \
    -w 500 \
    -v

modisco report \
    -i "${averaged_dir}/${dataset}/modisco/modisco_counts_results.h5" \
    -o "${averaged_dir}/${dataset}/modisco/counts_report" \
    -s "${averaged_dir}/${dataset}/modisco/counts_report" \
    -m "${ref_db_meme}"

echo "${dataset} MoDISco complete"
