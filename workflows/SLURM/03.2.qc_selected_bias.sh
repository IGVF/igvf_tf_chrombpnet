#!/bin/bash
#SBATCH --job-name=qc_selected_bias
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Same GPU limits as 03.0/04.0: cuda/11.5 cannot drive Ada (8.9) or Hopper
# (9.0), and 7.0/7.5 are excluded for speed. See 03.0 for the measurements.
# This step previously carried NO constraint while loading the same cuda
# module, so it could land on an H100 and fail obscurely.
#SBATCH --constraint="GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=1-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 03.2.qc_selected_bias.sh
# Purpose: Full QC (marginal footprinting + DeepLIFT interpretation +
#   TF-MoDISco) on only the bias model selected per fold in
#   dataset_config.sh (fold_bias_suffix), populated by 03.1.select_bias.sh.
#   This is the expensive step that verifies the Tn5 signal has actually
#   been learned - inspect evaluation/*_bias_profile.pdf for Tn5 vs GC-rich
#   motifs. Deliberately not run on the full fold x bias-factor sweep (03.0).
#
# Array index = fold index (one task per fold, not per bias-factor).
#
# Usage:
#   export DATASET_DIR=/path/to/igvf_tf_collab/<dataset>
#   sbatch 03.2.qc_selected_bias.sh            # all folds (array 0-4)
#   sbatch --array=0 03.2.qc_selected_bias.sh  # fold 0 only (quick test)
#
# Prerequisites: 03.0.train_bias_model.sh and 03.1.select_bias.sh must have
#   completed, with fold_bias_suffix set in dataset_config.sh.

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

fold="${folds[$SLURM_ARRAY_TASK_ID]}"
[[ -z "${fold}" ]] && { echo "Invalid array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

suffix="${fold_bias_suffix[${fold}]}"
if [[ -z "${suffix}" ]]; then
    echo "ERROR: No bias suffix defined for fold ${fold} in fold_bias_suffix (dataset_config.sh)." >&2
    exit 1
fi



metadata_start "03.2.qc_selected_bias"



fold_json="${folds_dir}/fold_${fold}.json"
file_prefix="${bias_dataset}_${peak_type}_fold_${fold}"
out_dir="${results_path}/bias_models/bias_model${suffix}/${bias_dataset}_${peak_type}_fold_${fold}"
model_file="${out_dir}/models/${file_prefix}_bias.h5"


metadata_inputs+=( "bias_model=${model_file}" "genome=${genome_fa}" "fold_json=${fold_json}" )
require_input "${model_file}" 03.0.train_bias_model.sh
require_input "${genome_fa}" "cli.py download-references"
require_input "${fold_json}"
require_input "${chrom_sizes}" "cli.py download-references"
metadata_outputs+=( "bias_qc=${out_dir}/evaluation" )
preflight_check

load_gpu_modules
activate_env "${CONDA_ENV}"
gpu_env
metadata_params+=( "fold=${fold}" "bias_suffix=${suffix}" )
echo "[$(date)] [fold ${fold}] Full QC on selected bias model (suffix ${suffix})"
echo "  model      : ${model_file}"
echo "  output dir : ${out_dir}"

if [[ ! -f "${model_file}" ]]; then
    echo "ERROR: Bias model not found for fold ${fold} (suffix ${suffix}):" >&2
    echo "  ${model_file}" >&2
    echo "  Run 03.0.train_bias_model.sh first." >&2
    exit 1
fi

# No `set -e` in this step, and this is the last real command, so without the
# guard a DeepLIFT/TF-MoDISco failure prints "Done." and exits 0.
python "${src_dir}/run_bias_qc.py" \
    --bias-model "${model_file}" \
    --output-dir "${out_dir}" \
    --file-prefix "${file_prefix}" \
    --genome "${genome_fa}" \
    --chrom-sizes "${chrom_sizes}" \
    --fold-json "${fold_json}"
if [[ $? -ne 0 || ! -d "${out_dir}/evaluation" ]]; then
    echo "ERROR: run_bias_qc.py failed for fold ${fold}; ${out_dir}/evaluation was not written." >&2
    exit 1
fi

echo "[$(date)] [fold ${fold}] Done."
