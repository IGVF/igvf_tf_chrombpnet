#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu,owners
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 09.generate_predictions.sh
# Purpose: Generate genome-wide accessibility prediction bigwigs for one dataset,
#          averaged across all available trained folds.
#          One SLURM array job per dataset.  Both bias-corrected and uncorrected
#          predictions are generated.
#
# If only one fold is trained, the "average" is just that single model.
# The script collects whichever folds are present in full_model_dir.
#
# Outputs (inside <predictions_dir>/<dataset>_<peak_type>/):
#   <dataset>_avg_chrombpnet_nobias.bw
#   <dataset>_avg_chrombpnet_nobias_preds_w_logcounts.bed
#   <dataset>_avg_chrombpnet_uncorrected.bw
#   <dataset>_avg_chrombpnet_uncorrected_preds_w_logcounts.bed
#
# Usage:
#   sbatch 09.generate_predictions.sh            # dataset 0
#   sbatch 09.generate_predictions.sh            # (override with --array=0 if needed)
#
# Prerequisites: 05.train_full_model.sh must have completed.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

dataset="${datasets[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${dataset}" ]] && { echo "No dataset at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

ml devel
ml system
ml cairo
ml pango/1.40.10
ml cuda/11.5.0
ml cudnn/8.6.0.163

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

peaks_file="${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
out_dir="${predictions_dir}/${dataset}_${peak_type}"
mkdir -p "${out_dir}"

# Collect all available fold model files for this dataset
nobias_models=()
full_models=()
for fold in "0" "1" "2" "3" "4"; do
    nb="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}/models/chrombpnet_nobias.h5"
    fm="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}/models/chrombpnet.h5"
    [[ -f "${nb}" ]] && nobias_models+=("${nb}")
    [[ -f "${fm}" ]] && full_models+=("${fm}")
done

if [[ ${#nobias_models[@]} -eq 0 ]]; then
    echo "[${dataset}] No trained models found in ${full_model_dir}. Run 05.train_full_model.sh first." >&2
    exit 1
fi

echo "[$(date)] [${dataset}] Found ${#nobias_models[@]} fold(s). Generating predictions..."

for mode in "bias_corrected" "uncorrected"; do
    if [[ "${mode}" == "bias_corrected" ]]; then
        models=( "${nobias_models[@]}" )
        out_key="nobias"
    else
        models=( "${full_models[@]}" )
        out_key="uncorrected"
    fi

    out_prefix="${out_dir}/${dataset}_avg"
    done_file="${out_prefix}_chrombpnet_${out_key}.bw"

    if [[ -f "${done_file}" ]]; then
        echo "  [${dataset} ${mode}] Already exists, skipping."
        continue
    fi

    model_flags=""
    for m in "${models[@]}"; do
        model_flags="${model_flags} --chrombpnet-model ${m}"
    done

    echo "[$(date)] [${dataset} ${mode}] Running predictions (${#models[@]} model(s))..."

    python3 "${SCRIPT_DIR}/predict_and_avg.py" \
        --regions       "${peaks_file}" \
        --genome        "${genome_fa}" \
        --chrom-sizes   "${chrom_sizes}" \
        --output-prefix "${out_prefix}" \
        --output-key    "${out_key}" \
        --output-bed    True \
        --batch-size    64 \
        ${model_flags}

    echo "[$(date)] [${dataset} ${mode}] Done."
done

echo "[$(date)] [${dataset}] Predictions complete."
