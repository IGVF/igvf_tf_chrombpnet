#!/bin/bash
#SBATCH --job-name=full_model_interpret
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Same GPU limits as 03.0/04.0: cuda/11.5 cannot drive Ada (8.9) or Hopper
# (9.0), and 7.0/7.5 are excluded for speed. See 03.0/04.0.
#SBATCH --constraint="GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=1-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 04.4.qc_full_model_interpret.sh
# Purpose: Per-fold interpretation QC of the full model, GPU half: DeepLIFT
#   contribution scores on a 30K subsample of the fold's filtered peaks, seed
#   1234 -- what `chrombpnet pipeline` would have run next inside 04.0, except
#   that both heads are scored: pipeline scores profile only, and the counts
#   head is where a model absorbs composition (see src/run_full_model_qc.py).
#
# Why its own step: pipeline ran this and then TF-MoDISco inside the training
# job, and TF-MoDISco is CPU-only and the long pole, so the GPU sat idle for
# hours. 04.0 now stops before interpretation; this step is the GPU half and
# 04.5 the CPU half -- the same split as 03.2/03.3 for the bias model. Nothing
# downstream waits on either: they are QC (per-fold motifs), while the
# analysis-grade scores come from 05 on all peaks.
#
# Array index = fold index.
#
# Input:  ${full_model_dir}/<dataset>_<peak_type>_fold_<fold>/ from 04.0
# Output: auxiliary/interpret_subsample/chrombpnet_nobias.{profile,counts}_scores.h5
#         auxiliary/30K_subsample_peaks.bed
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 04.4.qc_full_model_interpret.sh              # all folds
#   sbatch --array=0 04.4.qc_full_model_interpret.sh    # fold 0 only
# Prerequisites: 04.0.train_full_model.sh for the fold.

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


metadata_start "04.4.qc_full_model_interpret"
metadata_params+=( "fold=${fold}" "subsample=30000" "seed=1234" "scores=profile,counts" )
for dataset in "${datasets[@]}"; do
    _dir="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}"
    metadata_inputs+=( "model=${_dir}/models/chrombpnet_nobias.h5" )
    for _head in profile counts; do
        metadata_outputs+=( "contributions=${_dir}/auxiliary/interpret_subsample/chrombpnet_nobias.${_head}_scores.h5" )
    done
    require_input "${_dir}/models/chrombpnet_nobias.h5" 04.0.train_full_model.sh
    require_input "${_dir}/auxiliary/chrombpnet_nobias_footprints.h5" 04.0.train_full_model.sh
done
require_input "${genome_fa}" "cli.py download-references"
preflight_check

load_gpu_modules
activate_env "${CONDA_ENV}"
gpu_env

for dataset in "${datasets[@]}"; do
    out_dir="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}"
    echo "[$(date)] [${dataset} fold ${fold}] DeepLIFT (profile + counts) on the full model: ${out_dir}"
    METADATA_RSS_FILE="${out_dir}/.peak_rss_gb_044"
    export METADATA_RSS_FILE
    # No `set -e` in this step: guard the call and its output explicitly.
    python "${src_dir}/run_full_model_qc.py" \
        --stage gpu \
        --model-dir "${out_dir}" \
        --genome "${genome_fa}" \
        --data-type "${assay}"
    _rc=$?
    [[ -s "${METADATA_RSS_FILE}" ]] && metadata_metrics+=( "peak_rss_gb=$(<"${METADATA_RSS_FILE}")" )
    _scores="${out_dir}/auxiliary/interpret_subsample/chrombpnet_nobias.counts_scores.h5"
    if [[ ${_rc} -ne 0 || ! -f "${_scores}" ]]; then
        echo "ERROR: run_full_model_qc.py --stage gpu failed for ${dataset} fold ${fold}; ${_scores} was not written." >&2
        exit 1
    fi
done

echo "[$(date)] Fold ${fold}: done. Next: 04.5.modisco_full_model.sh (CPU)."
