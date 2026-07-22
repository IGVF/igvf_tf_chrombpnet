#!/bin/bash
#SBATCH --job-name=full_model_selected
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
# Pin to GPUs the loaded cuda/11.5 supports (compute capability <= 8.6:
# Volta/Turing/Ampere). Excludes Ada (GPU_CC 8.9) and Hopper H100/H200 (9.0),
# which cuda 11.5 cannot drive efficiently.
#SBATCH --constraint="GPU_CC:7.0|GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=2-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0-4

# 05.train_full_model.sh
# Purpose: Train the bias-factorised ChromBPNet full model for all datasets and
#          folds, using the per-fold optimal bias model selected in step 04
#          (04.0.select_bias_model.py). Each fold uses the bias suffix recorded
#          in fold_bias_suffix in config.sh.
#
# Output directory: ${full_model_dir} (set in config.sh)
#
# Usage:
#   sbatch 04.0.train_full_model.sh            # all folds (array 0-4)
#   sbatch --array=0 04.0.train_full_model.sh  # fold 0 only
#
# Prerequisites: 03.0.train_bias_model.sh must have completed for all folds.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

fold="${folds[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${fold}" ]] && { echo "No fold at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

suffix="${fold_bias_suffix[${fold}]}"
if [[ -z "${suffix}" ]]; then
    echo "ERROR: No bias suffix defined for fold ${fold} in fold_bias_suffix (config.sh)." >&2
    exit 1
fi

bias_model="${results_path}/bias_models/bias_model${suffix}/${bias_dataset}_${peak_type}_fold_${fold}/models/${bias_dataset}_${peak_type}_fold_${fold}_bias.h5"

if [[ ! -f "${bias_model}" ]]; then
    echo "ERROR: Bias model not found for fold ${fold} (suffix ${suffix}):" >&2
    echo "  ${bias_model}" >&2
    echo "  Run 03.0.train_bias_model.sh first." >&2
    exit 1
fi

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

echo "[$(date)] Fold ${fold}: training full models (bias suffix ${suffix})"
echo "  bias model : ${bias_model}"
echo "  output dir : ${full_model_dir_selected}"

echo ${datasets}
for dataset in "${datasets[@]}"; do
    out_dir="${full_model_dir_selected}/${dataset}_${peak_type}_fold_${fold}"
    model_file="${out_dir}/models/chrombpnet_nobias.h5"

    if [[ -f "${model_file}" ]]; then
        echo "  [${dataset} fold ${fold}] Already done, skipping."
        continue
    fi

    mkdir -p "${out_dir}"

    fragments_file="${fragments_path}/${dataset}_atac_fragments_main_chrs.tsv.gz"
    peaks_file="${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
    negatives_file="${data_path}/${dataset}/output_${peak_type}_fold_${fold}_negatives.bed"
    fold_json="${folds_dir}/fold_${fold}.json"

    for f in "${fragments_file}" "${peaks_file}" "${negatives_file}" "${fold_json}"; do
        [[ -f "${f}" ]] || { echo "  [${dataset} fold ${fold}] Missing input: ${f}" >&2; continue 2; }
    done

    echo "[$(date)] [${dataset} fold ${fold}] Training full model (bias ${suffix})..."

    chrombpnet pipeline \
        -ifrag "${fragments_file}" \
        -d "ATAC" \
        -g "${genome_fa}" \
        -c "${chrom_sizes}" \
        -p "${peaks_file}" \
        -n "${negatives_file}" \
        -fl "${fold_json}" \
        -b "${bias_model}" \
        -o "${out_dir}"

    echo "[$(date)] [${dataset} fold ${fold}] Done."
done

echo "[$(date)] Fold ${fold}: full model training complete."
