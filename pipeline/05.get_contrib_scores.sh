#!/bin/bash
#SBATCH --job-name=contribs
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=2-0
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 05.get_contrib_scores.sh
# Purpose: Compute DeepLIFT contribution scores for each dataset x fold using the
#          bias-corrected chrombpnet_nobias model. One SLURM array job per fold;
#          each job processes all datasets.
#
# Outputs per dataset/fold (inside ${full_model_dir}/{dataset}_{peak_type}_fold_{fold}/interpretation/):
#   interpretation.counts_scores.h5 / .bw
#
# Usage:
#   sbatch 05.get_contrib_scores.sh            # all folds (array 0-4)
#   sbatch --array=0 05.get_contrib_scores.sh  # fold 0 only
#
# Prerequisites: 04.0.train_full_model.sh must have completed.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

fold="${folds[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${fold}" ]] && { echo "No fold at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

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

echo "[$(date)] Fold ${fold}: computing contribution scores for datasets [${datasets[*]}]"
for dataset in "${datasets[@]}"; do
    model_file="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}/models/chrombpnet_nobias.h5"

    if [[ ! -f "${model_file}" ]]; then
        echo "  [${dataset} fold ${fold}] Model not found, skipping: ${model_file}" >&2
        echo "  Run 04.0.train_full_model.sh first." >&2
        continue
    fi

    interp_dir="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}/interpretation"

    mkdir -p "${interp_dir}"
    peaks_file="${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"

    echo "[$(date)] [${dataset} fold ${fold}] Computing contribution scores..."

    chrombpnet contribs_bw \
        -m "${model_file}" \
        -r "${peaks_file}" \
        -g "${genome_fa}" \
        -c "${chrom_sizes}" \
        -op "${interp_dir}/interpretation"

    echo "[$(date)] [${dataset} fold ${fold}] Done."
done

echo "[$(date)] Fold ${fold}: contribution score computation complete."
