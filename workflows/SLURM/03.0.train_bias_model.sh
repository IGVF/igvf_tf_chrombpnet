#!/bin/bash
#SBATCH --job-name=bias_sweep
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Pin to GPUs the loaded cuda/11.5 supports (compute capability <= 8.6:
# Volta/Turing/Ampere). Excludes Ada (GPU_CC 8.9) and Hopper H100/H200 (9.0),
# which cuda 11.5 cannot drive efficiently.
#SBATCH --constraint="GPU_CC:7.0|GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=2-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0-19
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 03.0.train_bias_model.sh
# Purpose: Train Tn5 bias models for a sweep of bias threshold factors on
#   bias_dataset (defined in dataset_config.sh), then compute fast QC metrics
#   (counts/profile Pearson r, JSD) for each - the metrics select_bias_model.py
#   (03.1) needs to pick a winner per fold. Deliberately skips the expensive
#   interpretation + TF-MoDISco QC (that only runs on the selected bias model,
#   in 03.2.qc_selected_bias.sh) - running it on every fold x factor combo is
#   what made this step slow before.
#
# Cost: this array is len(bias_sweep_folds) x len(bias_factors) GPU jobs — 20 by
#   default. Set bias_sweep_folds in config.yaml to pilot on fewer folds first;
#   note that per-fold winners genuinely differ in practice (igvf3 selected
#   _08/_08/_06/_08/_07), so a pilot informs the shortlist rather than replacing
#   the per-fold choice. `cli.py config validate` prints the required --array.
#
#   Array index maps to fold x bias_factor: task_id = fold_idx * n_factors + factor_idx
#     e.g. with 4 bias_factors: tasks 0-3 = fold 0, tasks 4-7 = fold 1, etc.
#   --array default (0-19) assumes 5 folds x 4 factors; override for datasets
#   with a different bias_factors length (e.g. igvf11_h7_hesc has 6 -> 0-29).
#
# Usage:
#   export DATASET_DIR=/path/to/igvf_tf_collab/<dataset>
#   sbatch 03.0.train_bias_model.sh              # all folds x factors
#   sbatch --array=0 03.0.train_bias_model.sh    # fold 0, first bias factor only (quick test)
#
# After all jobs complete, run 03.1.select_bias.sh, then 03.2.qc_selected_bias.sh,
# then 04.0.train_full_model.sh.

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

n_factors=${#bias_factors[@]}
fold_idx=$(( SLURM_ARRAY_TASK_ID / n_factors ))
factor_idx=$(( SLURM_ARRAY_TASK_ID % n_factors ))
# bias_sweep_folds defaults to folds; set it in config.yaml to pilot the sweep
# on a subset (see "Cost" in the header).
fold="${bias_sweep_folds[$fold_idx]}"
bf="${bias_factors[$factor_idx]}"
suffix="${bias_suffixes_sweep[$factor_idx]}"
[[ -z "${fold}" || -z "${bf}" ]] && { echo "Invalid array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }



metadata_start "03.0.train_bias_model"



set_signal_args
signal_file="${signal_path}"
peaks_file="${data_path}/${bias_dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
negatives_file="${data_path}/${bias_dataset}/output_${peak_type}_fold_${fold}_negatives.bed"
fold_json="${folds_dir}/fold_${fold}.json"
file_prefix="${bias_dataset}_${peak_type}_fold_${fold}"
out_dir="${results_path}/bias_models/bias_model${suffix}/${bias_dataset}_${peak_type}_fold_${fold}"
model_file="${out_dir}/models/${file_prefix}_bias.h5"


metadata_inputs+=( "signal=${signal_file}" "peaks=${peaks_file}" "negatives=${negatives_file}" "fold_json=${fold_json}" )
require_input "${signal_file}" 00.0.prepare_signal.sh
require_input "${peaks_file}" 00.1.preprocess_peaks.sh
require_input "${negatives_file}" 01.0.preprocess_nonpeaks.sh
require_input "${fold_json}"
require_input "${genome_fa}"   "cli.py download-references"
require_input "${chrom_sizes}" "cli.py download-references"
preflight_check

load_gpu_modules
activate_env "${CONDA_ENV}"
gpu_env
metadata_outputs+=( "bias_model=${model_file}" )
metadata_params+=( "fold=${fold}" "bias_factor=${bf}" "bias_suffix=${suffix}" )
echo "[$(date)] [fold ${fold} bias=${bf}] Training bias model"
echo "  output dir : ${out_dir}"

if [[ -f "${model_file}" ]]; then
    echo "  Model already trained, skipping training."
else
    for f in "${signal_file}" "${peaks_file}" "${negatives_file}" "${fold_json}"; do
        [[ -f "${f}" ]] || { echo "  Missing input: ${f}" >&2; exit 1; }
    done

    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"

    python "${src_dir}/chrombpnet_train.py" \
        --prepared-bigwig "${data_path}/signal" \
        ${prepared_required:+--require-prepared} -- \
        bias train \
        "${signal_args[@]}" \
        -d "${assay}" \
        -g "${genome_fa}" \
        -c "${chrom_sizes}" \
        -p "${peaks_file}" \
        -n "${negatives_file}" \
        -fl "${fold_json}" \
        -b "${bf}" \
        -o "${out_dir}" \
        -fp "${file_prefix}"

    if [[ $? -ne 0 || ! -f "${model_file}" ]]; then
        echo "ERROR: chrombpnet bias train failed for fold ${fold} bias=${bf} (bias threshold factor may be too low/high for this fold - see stdout above)." >&2
        exit 1
    fi
fi

echo "[$(date)] [fold ${fold} bias=${bf}] Computing fast QC metrics."

python "${src_dir}/predict_bias_metrics.py" \
    --bias-model "${model_file}" \
    --output-dir "${out_dir}" \
    --file-prefix "${file_prefix}" \
    --genome "${genome_fa}" \
    --fold-json "${fold_json}"

echo "[$(date)] [fold ${fold} bias=${bf}] Done."
