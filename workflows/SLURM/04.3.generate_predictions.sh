#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Same GPU limits as 03.0/04.0: cuda/11.5 cannot drive Ada (8.9) or Hopper
# (9.0), and 7.0/7.5 are excluded for speed. See 03.0 for the measurements.
# This step previously carried NO constraint while loading the same cuda
# module, so it could land on an H100 and fail obscurely.
#SBATCH --constraint="GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=4:00:00
#SBATCH --partition=gpu,owners
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 04.3.generate_predictions.sh
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
#   sbatch 04.3.generate_predictions.sh            # dataset 0
#   sbatch 04.3.generate_predictions.sh            # (override with --array=0 if needed)
#
# Prerequisites: 04.0.train_full_model.sh must have completed.

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

load_gpu_modules

activate_env "${CONDA_ENV}"

metadata_start "04.3.generate_predictions"


gpu_env

peaks_file="${peaks_dir}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"

metadata_inputs+=( "peaks=${peaks_file}" "genome=${genome_fa}" )
require_input "${peaks_file}"  00.1.preprocess_peaks.sh
require_input "${genome_fa}"   "cli.py download-references"
require_input "${chrom_sizes}" "cli.py download-references"
preflight_check
metadata_outputs+=( "predictions=${out_dir}" )
metadata_params+=( "dataset=${dataset}" )
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

    model_flags=""
    for m in "${models[@]}"; do
        model_flags="${model_flags} --chrombpnet-model ${m}"
    done

    echo "[$(date)] [${dataset} ${mode}] Running predictions (${#models[@]} model(s))..."

    python3 "${src_dir}/predict_and_avg.py" \
        --regions       "${peaks_file}" \
        --genome        "${genome_fa}" \
        --chrom-sizes   "${chrom_sizes}" \
        --output-prefix "${out_prefix}" \
        --output-key    "${out_key}" \
        --output-bed    True \
        --batch-size    64 \
        ${model_flags}
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
