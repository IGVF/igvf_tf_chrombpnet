#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# GPU: JAX brings its own CUDA 13 wheels (no cuda module is loaded), so what
# gates a node is its NVIDIA driver (>= 580), not the card generation.
# GPU_CC 8.0 (A100) and 8.6 (A40, RTX_3090) are the ones the pipeline has
# run on so far; widen the constraint at submit time to try others.
#SBATCH --constraint="GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=4:00:00
#SBATCH --partition=gpu,owners
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 04.3.generate_predictions.sh
# Purpose: Predicted accessibility bigwigs over the dataset's peaks, averaged
#          across all available trained folds (src/predict_and_avg.py).
#          One SLURM array job per dataset.  Both bias-corrected
#          (chrombpnet_nobias.h5) and uncorrected (chrombpnet.h5) predictions
#          are generated.
#
# If only one fold is trained, the "average" is just that single model.
# The script collects whichever folds are present in full_model_dir, scanning
# folds 0-4 whatever the config's folds list says.
#
# Inference runs on Keras 3 / JAX with no device switch, so the step calls
# require_gpu jax after activate_env: without it a node whose JAX cannot see
# the GPU would predict on the CPU for the whole time limit.
#
# Input:  ${peaks_dir}/<dataset>_<peak_type>_peaks_no_blacklist.narrowPeak (00.1)
#         ${full_model_dir}/<dataset>_<peak_type>_fold_<fold>/models/ (04.0)
# Outputs (inside <predictions_dir>/<dataset>_<peak_type>/):
#   <dataset>_avg_chrombpnet_nobias.bw
#   <dataset>_avg_chrombpnet_nobias_preds_w_logcounts.bed
#   <dataset>_avg_chrombpnet_uncorrected.bw
#   <dataset>_avg_chrombpnet_uncorrected_preds_w_logcounts.bed
#
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 04.3.generate_predictions.sh            # array 0: the config's one dataset
#
# Prerequisites: 00.1.preprocess_peaks.sh and 04.0.train_full_model.sh.

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

metadata_start "04.3.generate_predictions"

peaks_file="${peaks_dir}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
out_dir="${predictions_dir}/${dataset}_${peak_type}"

metadata_inputs+=( "peaks=${peaks_file}" "genome=${genome_fa}" )
metadata_outputs+=( "predictions=${out_dir}" )
metadata_params+=( "dataset=${dataset}" )
require_input "${peaks_file}"  00.1.preprocess_peaks.sh
require_input "${genome_fa}"   "cli.py download-references"
require_input "${chrom_sizes}" "cli.py download-references"
preflight_check

activate_env "${chrombpnet_env}"
gpu_env
require_gpu jax

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
    echo "[${dataset}] No trained models found in ${full_model_dir}. Run 04.0.train_full_model.sh first." >&2
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

    model_flags=()
    for m in "${models[@]}"; do
        model_flags+=( --chrombpnet-model "${m}" )
    done

    echo "[$(date)] [${dataset} ${mode}] Running predictions (${#models[@]} model(s))..."

    python "${src_dir}/predict_and_avg.py" \
        --regions       "${peaks_file}" \
        --genome        "${genome_fa}" \
        --chrom-sizes   "${chrom_sizes}" \
        --output-prefix "${out_prefix}" \
        --output-key    "${out_key}" \
        --output-bed    True \
        --batch-size    64 \
        "${model_flags[@]}"
    # No `set -e`, and this runs inside a `for mode` loop, so an unguarded
    # failure both prints "Done." and lets the NEXT mode start as though this
    # one had produced its bigwig. ${done_file} is the same marker the
    # skip-check above uses.
    if [[ $? -ne 0 || ! -f "${done_file}" ]]; then
        echo "ERROR: predict_and_avg.py failed for ${dataset} ${mode}; ${done_file} was not written." >&2
        exit 1
    fi

    echo "[$(date)] [${dataset} ${mode}] Done."
done

echo "[$(date)] [${dataset}] Predictions complete."
