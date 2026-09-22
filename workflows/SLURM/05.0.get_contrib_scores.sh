#!/bin/bash
#SBATCH --job-name=contribs
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 05.0.get_contrib_scores.sh
# Purpose: Compute DeepLIFT contribution scores for each dataset x fold using the
#          bias-corrected chrombpnet_nobias model. One SLURM array job per fold;
#          each job processes all datasets.
#
# Outputs per dataset/fold (inside ${full_model_dir}/{dataset}_{peak_type}_fold_{fold}/interpretation/):
#   interpretation.counts_scores.h5 / .bw
#
# Usage:
#   export DATASET_DIR=/path/to/igvf_tf_collab/<dataset>
#   sbatch 05.0.get_contrib_scores.sh            # all folds (array 0-4)
#   sbatch --array=0 05.0.get_contrib_scores.sh  # fold 0 only
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

fold="${folds[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${fold}" ]] && { echo "No fold at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

load_gpu_modules

activate_env "${CONDA_ENV}"

metadata_start "05.0.get_contrib_scores"
for _ds in "${datasets[@]}"; do
    metadata_inputs+=( "model=${full_model_dir}/${_ds}_${peak_type}_fold_${fold}/models/chrombpnet_nobias.h5" )
    metadata_outputs+=( "contributions=${full_model_dir}/${_ds}_${peak_type}_fold_${fold}/interpretation/interpretation.counts_scores.h5" )
done
unset _ds
metadata_params+=( "fold=${fold}" )


gpu_env

echo "[$(date)] Fold ${fold}: computing contribution scores for datasets [${datasets[*]}]"
for dataset in "${datasets[@]}"; do
    model_file="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}/models/chrombpnet_nobias.h5"

    if [[ ! -f "${model_file}" ]]; then
        echo "ERROR: Model not found for ${dataset} fold ${fold}: ${model_file}" >&2
        echo "  Run 04.0.train_full_model.sh first." >&2
        exit 1
    fi

    interp_dir="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}/interpretation"
    peaks_file="${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
    done_file_h5="${interp_dir}/interpretation.counts_scores.h5"
    done_file_bw="${interp_dir}/interpretation.counts_scores.bw"

    echo "[$(date)] [${dataset} fold ${fold}] Computing contribution scores"
    echo "  model : ${model_file}"
    echo "  output: ${interp_dir}/"

    if [[ -f "${done_file_h5}" && -f "${done_file_bw}" ]]; then
        echo "  Already done, skipping."
        continue
    fi

    for f in "${model_file}" "${peaks_file}"; do
        [[ -f "${f}" ]] || { echo "  Missing input: ${f}" >&2; exit 1; }
    done

    mkdir -p "${interp_dir}"

    chrombpnet contribs_bw \
        -m "${model_file}" \
        -r "${peaks_file}" \
        -g "${genome_fa}" \
        -c "${chrom_sizes}" \
        -op "${interp_dir}/interpretation"

    echo "[$(date)] [${dataset} fold ${fold}] Done."
done

echo "[$(date)] Fold ${fold}: contribution score computation complete."
