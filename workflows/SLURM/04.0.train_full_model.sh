#!/bin/bash
#SBATCH --job-name=full_model_selected
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Two limits at once, an upper and a lower.
#   upper: the loaded cuda/11.5 cannot drive Ada (GPU_CC 8.9, L40S) or Hopper
#          (9.0, H100/H200), so those are excluded for correctness. Lifting
#          that needs the env off tensorflow==2.8/CUDA 11, not a flag.
#   lower: 7.0/7.5 (V100, TITAN_V, RTX_2080Ti) are ELIGIBLE for cuda 11.5 but
#          are excluded on purpose, to avoid the slowest silicon. The July 2026
#          bias_sweep runs, by node: TITAN_V (7.0) 20:51 and 22:24; RTX_2080Ti
#          (7.5) 12:09 and 05:45; A100_SXM4 (8.0) 08:31. So CC 7.0 is clearly
#          the slow tier and worth excluding. Note the evidence does NOT show
#          8.0 beating 7.5 -- the single fastest run was a 2080Ti -- and the
#          runs differ by fold and bias factor, so this is a coarse signal, not
#          a benchmark. If the 8.0|8.6 queue is deep, adding GPU_CC:7.5 back at
#          submit time is defensible.
# Leaves GPU_CC 8.0 (A100_SXM4/A100_PCIE) and 8.6 (A40, RTX_3090).
#SBATCH --constraint="GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=2-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0-4

# 04.0.train_full_model.sh
# Purpose: Train the bias-factorised ChromBPNet full model for all datasets and
#          folds, using the per-fold optimal bias model selected in step 03
#          (03.1.select_bias.sh / select_bias_model.py). Each fold uses the
#          bias suffix recorded in fold_bias_suffix in dataset_config.sh.
#
# Output directory: ${full_model_dir_selected} (set in config.sh)
#
# Usage:
#   export DATASET_DIR=/path/to/igvf_tf_collab/<dataset>
#   sbatch 04.0.train_full_model.sh            # all folds (array 0-4)
#   sbatch --array=0 04.0.train_full_model.sh  # fold 0 only
#
# Prerequisites: 03.0.train_bias_model.sh, 03.1.select_bias.sh, and
#   03.2.qc_selected_bias.sh must have completed for all folds. Bias suffixes
#   must be set in dataset_config.sh.

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

fold="${folds[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${fold}" ]] && { echo "No fold at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

suffix="${fold_bias_suffix[${fold}]}"
if [[ -z "${suffix}" ]]; then
    echo "ERROR: No bias suffix defined for fold ${fold} in fold_bias_suffix (dataset_config.sh)." >&2
    exit 1
fi

bias_model="${results_path}/bias_models/bias_model${suffix}/${bias_dataset}_${peak_type}_fold_${fold}/models/${bias_dataset}_${peak_type}_fold_${fold}_bias.h5"

if [[ ! -f "${bias_model}" ]]; then
    echo "ERROR: Bias model not found for fold ${fold} (suffix ${suffix}):" >&2
    echo "  ${bias_model}" >&2
    echo "  Run 03.0.train_bias_model.sh first." >&2
    exit 1
fi



metadata_start "04.0.train_full_model"
metadata_inputs+=( "bias_model=${bias_model}" )
require_input "${bias_model}" 03.0.train_bias_model.sh
require_input "${signal_path}" 00.0.prepare_signal.sh
require_input "${genome_fa}"   "cli.py download-references"
require_input "${chrom_sizes}" "cli.py download-references"
require_input "${folds_dir}/fold_${fold}.json" ""
for _ds in "${datasets[@]}"; do
    require_input "${peaks_dir}/${_ds}_${peak_type}_peaks_no_blacklist.narrowPeak" 00.1.preprocess_peaks.sh
    require_input "${data_path}/${_ds}/output_${peak_type}_fold_${fold}_negatives.bed" 01.0.preprocess_nonpeaks.sh
done
unset _ds
preflight_check

load_gpu_modules
activate_env "${CONDA_ENV}"
metadata_params+=( "fold=${fold}" "bias_suffix=${suffix}" )

# Epoch cap. chrombpnet trains up to 50 epochs and early stopping decides; at
# ~11 min an epoch on molab that is 3-9 hours, which a pipeline TEST does not
# need. full_model_epochs (config) or FULL_MODEL_EPOCHS (environment, which
# wins) passes -e. The model lands in the SAME directory -- 04.1 and 04.3 look
# for it there, and exercising them is the point of a capped run -- so the
# cap is recorded as max_epochs, and a real training afterwards needs a
# forced rerun (run_step.sh --force).
max_epochs="${FULL_MODEL_EPOCHS:-${full_model_epochs:-}}"
epoch_args=()
if [[ -n "${max_epochs}" ]]; then
    [[ "${max_epochs}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: epoch cap must be a positive integer, got '${max_epochs}'" >&2; exit 1; }
    epoch_args=( -e "${max_epochs}" )
fi
metadata_params+=( "max_epochs=${max_epochs:-50}" )


gpu_env

echo "[$(date)] Fold ${fold}: training full models (bias suffix ${suffix})"
echo "  bias model : ${bias_model}"
echo "  output dir : ${full_model_dir_selected}"

for dataset in "${datasets[@]}"; do
    out_dir="${full_model_dir_selected}/${dataset}_${peak_type}_fold_${fold}"
    model_file="${out_dir}/models/chrombpnet_nobias.h5"
    # Last file written by the pipeline's evaluation stage; absence means
    # training finished (model_file exists) but evaluation was cut short,
    # e.g. by preemption.
    eval_marker="${out_dir}/evaluation/chrombpnet_nobias_profile.pdf"

    if [[ -f "${model_file}" && -f "${eval_marker}" ]]; then
        echo "  [${dataset} fold ${fold}] Already done, skipping."
        continue
    fi

    set_signal_args
    peaks_file="${peaks_dir}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
    negatives_file="${data_path}/${dataset}/output_${peak_type}_fold_${fold}_negatives.bed"
    fold_json="${folds_dir}/fold_${fold}.json"

    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"

    echo "[$(date)] [${dataset} fold ${fold}] Training full model (bias ${suffix}, max epochs ${max_epochs:-50})..."

    python "${src_dir}/chrombpnet_train.py" \
        --prepared-bigwig "${data_path}/signal" \
        ${prepared_args[@]+"${prepared_args[@]}"} -- \
        pipeline \
        "${signal_args[@]}" \
        -d "${assay}" \
        -g "${genome_fa}" \
        -c "${chrom_sizes}" \
        -p "${peaks_file}" \
        -n "${negatives_file}" \
        -fl "${fold_json}" \
        -b "${bias_model}" \
        -o "${out_dir}" \
        ${epoch_args[@]+"${epoch_args[@]}"}
    # No `set -e` here. Guard on BOTH markers, the same pair the skip-check at
    # the top of this block uses: the model alone is written partway through,
    # and the profile PDF is the last file the evaluation stage emits. Without
    # this a failed run prints "Done." and exits 0, and the next rerun sees a
    # model with no eval, rm -rf's it and retrains from scratch -- silently
    # burning a second GPU allocation to rediscover the same failure.
    if [[ $? -ne 0 || ! -f "${model_file}" || ! -f "${eval_marker}" ]]; then
        echo "ERROR: chrombpnet pipeline failed for ${dataset} fold ${fold} (bias ${suffix})." >&2
        echo "       expected ${model_file} and ${eval_marker}" >&2
        exit 1
    fi

    echo "[$(date)] [${dataset} fold ${fold}] Done."
done

echo "[$(date)] Fold ${fold}: full model training complete."
