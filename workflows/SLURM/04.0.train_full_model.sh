#!/bin/bash
#SBATCH --job-name=full_model_selected
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# GPU classes: GPU_CC 8.0 (A100_SXM4/A100_PCIE) and 8.6 (A40, RTX_3090), the
# same as 03.0, whose header has the reasoning and the timings. In short: the
# gate is now the NVIDIA driver (the default cuda13 chrombpnet environment
# needs >= 580, the cuda12 one >= 525), not a cuda/11.5 module that could not
# drive Ada or Hopper; only 8.0 and 8.6 have been used with the 2.x wheels so
# far, every other class is untested; 7.0/7.5 were left out for speed.
#SBATCH --constraint="GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=2-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0-4

# 04.0.train_full_model.sh
# Purpose: Train the bias-factorised ChromBPNet full model for all datasets and
#          folds, using the per-fold optimal bias model selected in step 03
#          (03.1.select_bias.sh / select_bias_model.py). Each fold uses the
#          bias suffix recorded in fold_bias_suffix in config.yaml.
#
# Runs `chrombpnet pipeline -bw <prepared bigwig> --skip-interpretation
# --device gpu`: training, predictions and marginal footprinting -- what 04.1
# and 04.3 read -- then chrombpnet's train-mode report, and nothing more.
# DeepLIFT and TF-MoDISco, which pipeline would otherwise run inside this GPU
# job with the GPU idle through TF-MoDISco, are 04.4 (GPU) and 04.5 (CPU), the
# same split 03.2/03.3 make for the bias model. -bw trains from the bigwig 00.0
# built once on CPU, so no GPU job converts reads. It is launched in-process by
# src/chrombpnet_train.py, which first checks that bigwig's sidecar against the
# configured signal (path, md5, assay) -- a mismatch stops the job, there is
# no fallback conversion -- and records the training process's peak RSS.
# --device gpu makes chrombpnet fail at once, rather than train on the CPU,
# when JAX sees no GPU.
#
# Input:
#   ${bias_model}: <results>/bias_models/bias_model<suffix>/.../models/*_bias.h5  (03.0)
#   ${data_path}/signal/data_unstranded.bw, prepared_bigwig.json  (00.0)
#   ${signal_path}                  read only to md5 it for the sidecar check
#   ${peaks_dir}/<dataset>_<peak_type>_peaks_no_blacklist.narrowPeak  (00.1)
#   ${data_path}/<dataset>/output_<peak_type>_fold_<fold>_negatives.bed  (01.0)
#   ${folds_dir}/fold_<fold>.json, ${genome_fa}, ${chrom_sizes}
# Output, per dataset, in ${full_model_dir_selected}/<dataset>_<peak_type>_fold_<fold>/
# (${full_model_dir_selected} is set in config.sh):
#   models/chrombpnet_nobias.h5, models/chrombpnet.h5   (Keras 3 .h5)
#   evaluation/chrombpnet_metrics.json, evaluation/chrombpnet_predictions.h5
#   evaluation/chrombpnet_nobias_max_bias_response.txt
#   auxiliary/chrombpnet_nobias_footprints.h5           (the completion marker)
#
# Usage:
#   export DATASET=<name>
#   cd workflows/SLURM
#   sbatch 04.0.train_full_model.sh            # all folds (array 0-4)
#   sbatch --array=0 04.0.train_full_model.sh  # fold 0 only
#   FULL_MODEL_EPOCHS=2 sbatch --export=ALL --array=0 04.0.train_full_model.sh
#                                              # capped test run (see below)
#
# Prerequisites: 03.0.train_bias_model.sh, 03.1.select_bias.sh, and
#   03.2.qc_selected_bias.sh must have completed for all folds, and the bias
#   suffixes must be set in fold_bias_suffix in config.yaml. Also 00.0, 00.1
#   and 01.0, references installed, and the chrombpnet 2.x environment
#   (${chrombpnet_env}).

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
    echo "ERROR: No bias suffix defined for fold ${fold} in fold_bias_suffix (config.yaml)." >&2
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
# Training reads the prepared bigwig (${prepared_bigwig}, via -bw); the raw
# signal is only md5'd by the launcher's sidecar check. See common.sh.
set_signal_args
metadata_inputs+=( "bias_model=${bias_model}" "signal=${prepared_bigwig}" "signal=${prepared_bigwig_json}" )
require_input "${bias_model}" 03.0.train_bias_model.sh
require_input "${signal_path}" ""
require_input "${prepared_bigwig}" 00.0.prepare_signal.sh
require_input "${prepared_bigwig_json}" 00.0.prepare_signal.sh
require_input "${genome_fa}"   "cli.py download-references"
require_input "${chrom_sizes}" "cli.py download-references"
require_input "${folds_dir}/fold_${fold}.json" ""
for _ds in "${datasets[@]}"; do
    require_input "${peaks_dir}/${_ds}_${peak_type}_peaks_no_blacklist.narrowPeak" 00.1.preprocess_peaks.sh
    require_input "${data_path}/${_ds}/output_${peak_type}_fold_${fold}_negatives.bed" 01.0.preprocess_nonpeaks.sh
done
unset _ds
preflight_check

activate_env "${chrombpnet_env}"
metadata_params+=( "fold=${fold}" "bias_suffix=${suffix}" )

# Epoch cap. chrombpnet trains up to 50 epochs and early stopping decides; at
# ~11 min an epoch on molab (measured with chrombpnet 1.x) that is 3-9 hours,
# which a pipeline TEST does not need. full_model_epochs (config) or
# FULL_MODEL_EPOCHS (environment, which wins) passes -e. The model lands in
# the SAME directory -- 04.1 and 04.3 look for it there, and exercising them
# is the point of a capped run -- so the cap is recorded as max_epochs, and a
# real training afterwards needs RETRAIN=1 (below). run_step.sh --force is not
# enough: it only bypasses molab's step_done check, and this step's own skip
# rule would still keep the capped model.
#
# What a capped run keeps: chrombpnet 2.x's Keras 3 EarlyStopping restores
# the weights of the best epoch even when training runs all -e epochs without
# stopping early (1.x kept the last epoch in that case). So the model is the
# best of the epochs run, by validation loss, not simply the last one.
max_epochs="${FULL_MODEL_EPOCHS:-${full_model_epochs:-}}"
epoch_args=()
if [[ -n "${max_epochs}" ]]; then
    [[ "${max_epochs}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: epoch cap must be a positive integer, got '${max_epochs}'" >&2; exit 1; }
    epoch_args=( -e "${max_epochs}" )
fi
metadata_params+=( "max_epochs=${max_epochs:-50}" )
metadata_params+=( "retrain=${RETRAIN:-0}" )


gpu_env

echo "[$(date)] Fold ${fold}: training full models (bias suffix ${suffix})"
echo "  bias model : ${bias_model}"
echo "  output dir : ${full_model_dir_selected}"

for dataset in "${datasets[@]}"; do
    out_dir="${full_model_dir_selected}/${dataset}_${peak_type}_fold_${fold}"
    model_file="${out_dir}/models/chrombpnet_nobias.h5"
    # `pipeline --skip-interpretation` (below; 04.4/04.5 do DeepLIFT and
    # TF-MoDISco off this GPU job) moves the footprints file into auxiliary/
    # AFTER the predictions and max-bias-response 04.1/04.3 read, and then
    # only writes the train-mode report. So the footprints file means
    # everything this step owes downstream was written; the report after it is
    # a rendering of files already on disk, not something any step reads. A
    # preempted job leaves the model but not this, and is retrained.
    eval_marker="${out_dir}/auxiliary/chrombpnet_nobias_footprints.h5"
    metadata_outputs+=( "model=${model_file}" )
    metadata_outputs+=( "model=${out_dir}/models/chrombpnet.h5" )
    metadata_outputs+=( "metrics=${out_dir}/evaluation/chrombpnet_metrics.json" )
    metadata_outputs+=( "predictions=${out_dir}/evaluation/chrombpnet_predictions.h5" )
    metadata_outputs+=( "footprints=${out_dir}/evaluation/chrombpnet_nobias_max_bias_response.txt" )
    metadata_outputs+=( "footprints=${eval_marker}" )

    # RETRAIN=1 retrains over a finished model: the way to replace a capped
    # test model (above) with a real one. The rm -rf below clears it first.
    if [[ "${RETRAIN:-0}" != "1" && -f "${model_file}" && -f "${eval_marker}" ]]; then
        echo "  [${dataset} fold ${fold}] Already done, skipping (RETRAIN=1 to retrain)."
        continue
    fi

    peaks_file="${peaks_dir}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
    negatives_file="${data_path}/${dataset}/output_${peak_type}_fold_${fold}_negatives.bed"
    fold_json="${folds_dir}/fold_${fold}.json"

    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    METADATA_RSS_FILE="${out_dir}/.peak_rss_gb"   # see 03.0: the training process's real footprint
    export METADATA_RSS_FILE

    echo "[$(date)] [${dataset} fold ${fold}] Training full model (bias ${suffix}, max epochs ${max_epochs:-50})..."

    # --signal/--assay: the launcher checks ${prepared_bigwig}'s sidecar
    # against them and exits 1 before chrombpnet starts if 00.0 made it from
    # anything else.
    python "${src_dir}/chrombpnet_train.py" \
        --signal "${signal_path}" \
        --assay "${assay}" -- \
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
        ${epoch_args[@]+"${epoch_args[@]}"} \
        --skip-interpretation \
        --device gpu
    # No `set -e` here. Guard on BOTH markers, the same pair the skip-check at
    # the top of this block uses: the no-bias model is written as soon as
    # training ends, and the footprints file only after the predictions and
    # footprinting. Without this a failed run prints "Done." and exits 0, and
    # the next rerun sees a model with no footprints, rm -rf's it and retrains
    # from scratch -- silently burning a second GPU allocation to rediscover
    # the same failure.
    _train_status=$?
    [[ -s "${METADATA_RSS_FILE}" ]] && metadata_metrics+=( "peak_rss_gb=$(<"${METADATA_RSS_FILE}")" )
    if [[ ${_train_status} -ne 0 || ! -f "${model_file}" || ! -f "${eval_marker}" ]]; then
        echo "ERROR: chrombpnet pipeline failed for ${dataset} fold ${fold} (bias ${suffix}); see the log above." >&2
        echo "       expected ${model_file} and ${eval_marker}" >&2
        exit 1
    fi

    echo "[$(date)] [${dataset} fold ${fold}] Done."
done

echo "[$(date)] Fold ${fold}: full model training complete."
