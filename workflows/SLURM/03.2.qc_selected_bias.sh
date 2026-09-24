#!/bin/bash
#SBATCH --job-name=qc_selected_bias
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# GPU: JAX brings its own CUDA 13 wheels (no cuda module is loaded), so what
# gates a node is its NVIDIA driver (>= 580), not the card generation.
# GPU_CC 8.0 (A100) and 8.6 (A40, RTX_3090) are the ones the pipeline has
# run on so far; widen the constraint at submit time to try others.
#SBATCH --constraint="GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=1-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 03.2.qc_selected_bias.sh
# Purpose: GPU half of the QC on only the bias model selected per fold
#   (fold_bias_suffix in the dataset config.yaml, from 03.1.select_bias.sh):
#   src/run_bias_qc.py runs predictions with metrics on the test chromosomes,
#   then DeepLIFT contribution scores for BOTH heads on a 30K subsample of the
#   bias peaks. 03.3 (CPU) finds the motifs in those scores, which is what
#   verifies the Tn5 signal has actually been learned. Deliberately not run
#   on the full fold x bias-factor sweep (03.0).
#
# This step writes no motifs, footprints or PDF report: TF-MoDISco is
#   CPU-only and the long pole, so it runs in 03.3 rather than holding a GPU.
#   Interpretation runs with --device gpu and the step calls require_gpu jax,
#   because JAX would otherwise fall back to the CPU and run for the whole
#   time limit.
#
# Array index = fold index (one task per fold, not per bias-factor).
#
# Input:  ${results_path}/bias_models/bias_model<suffix>/<prefix>/ from 03.0
#           (models/<prefix>_bias.h5, auxiliary/<prefix>_filtered.bias_{peaks,nonpeaks}.bed)
#         ${data_path}/signal/data_unstranded.bw from 00.0 (the observed signal)
# Output: evaluation/<prefix>_bias_predictions.h5, _bias_metrics.json
#         auxiliary/interpret_subsample/<prefix>_bias.{counts,profile}_scores.h5
#
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 03.2.qc_selected_bias.sh            # all folds (array 0-4)
#   sbatch --array=0 03.2.qc_selected_bias.sh  # fold 0 only (quick test)
#
# Prerequisites: 00.0.prepare_signal.sh, 03.0.train_bias_model.sh and
#   03.1.select_bias.sh, with fold_bias_suffix set in the dataset config.yaml.

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
# The observed signal. Training reads 00.0's prepared bigwig in place (-bw),
# so there is no copy under ${out_dir}/auxiliary/ to fall back on.
prepared_bw="${data_path}/signal/data_unstranded.bw"


metadata_inputs+=( "bias_model=${model_file}" "genome=${genome_fa}" "fold_json=${fold_json}" )
metadata_inputs+=( "signal=${prepared_bw}" )
require_input "${model_file}" 03.0.train_bias_model.sh
require_input "${prepared_bw}" 00.0.prepare_signal.sh
require_input "${genome_fa}" "cli.py download-references"
require_input "${fold_json}"
# Files, not the evaluation/ directory: 03.3 consumes the two score files, a
# record that names only a directory cannot say whether they were written, and
# molab's step_done.py treats a directory output as missing, so the step would
# never count as done.
metadata_outputs+=( "bias_metrics=${out_dir}/evaluation/${file_prefix}_bias_metrics.json" )
metadata_outputs+=( "contributions=${out_dir}/auxiliary/interpret_subsample/${file_prefix}_bias.counts_scores.h5" )
metadata_outputs+=( "contributions=${out_dir}/auxiliary/interpret_subsample/${file_prefix}_bias.profile_scores.h5" )
preflight_check

activate_env "${chrombpnet_env}"
gpu_env
require_gpu jax
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
# guard a prediction/DeepLIFT failure prints "Done." and exits 0.
# --stage gpu: predictions + DeepLIFT only. TF-MoDISco is CPU-only and is the
# long pole, so it runs in 03.3 on a CPU partition rather than holding this
# allocation's GPU idle for hours. Same reasoning as 08.0.
METADATA_RSS_FILE="${out_dir}/.peak_rss_gb_032"   # see 03.0: the step's real footprint
export METADATA_RSS_FILE
python "${src_dir}/run_bias_qc.py" \
    --stage gpu \
    --bias-model "${model_file}" \
    --output-dir "${out_dir}" \
    --file-prefix "${file_prefix}" \
    --genome "${genome_fa}" \
    --fold-json "${fold_json}" \
    --bigwig "${prepared_bw}"
_rc=$?
[[ -s "${METADATA_RSS_FILE}" ]] && metadata_metrics+=( "peak_rss_gb=$(<"${METADATA_RSS_FILE}")" )
_scores="${out_dir}/auxiliary/interpret_subsample/${file_prefix}_bias.profile_scores.h5"
if [[ ${_rc} -ne 0 || ! -f "${_scores}" ]]; then
    echo "ERROR: run_bias_qc.py --stage gpu failed for fold ${fold}; ${_scores} was not written." >&2
    exit 1
fi

echo "[$(date)] [fold ${fold}] Done. Next: 03.3.modisco_selected_bias.sh (CPU)."
