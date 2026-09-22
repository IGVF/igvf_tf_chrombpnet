#!/bin/bash
#SBATCH --job-name=modisco_avg
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=4-0
#SBATCH --partition=engreitz
#SBATCH --qos=high_p
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 08.0.run_modisco.sh
# Purpose: Run TF-MoDISco on fold-averaged contribution scores (step 06/07).
#          One SLURM array job per dataset; produces ONE modisco result per
#          dataset per score type (rather than one per fold), which is then
#          used for the unified motif compendium.
#
# Why run on averaged scores:
#   Averaging contribution scores across folds before MoDISco improves
#   signal-to-noise ratio, so the discovered patterns are more reproducible
#   and biologically meaningful. This is the standard approach in the
#   Greenleaf lab ChromBPNet pipeline.
#
# score_types below: only "profile" is (re-)run here since counts modisco
# results already exist; add "counts" back to redo/extend.
#
# Partition/GPU/QOS: modisco motifs (tfmodisco-lite) is CPU-only, so this
# runs on the engreitz partition (no GPU request) with --qos=high_p. The
# default QOS on any partition caps walltime at 2 days for this account
# regardless of what sh_part's per-partition ceiling shows; --qos=high_p
# (7-day MaxWall, granted to the engreitz account) is required to actually
# get more than 2 days, which some datasets need.
#
# Input:  results/contrib_scores/{dataset}/{dataset}_average_shaps.{score_type}.h5  (step 06/07)
# Output: ${averaged_dir}/{dataset}/modisco/
#             modisco_{score_type}_results.h5
#             {score_type}_report/
#
# Usage:
#   export DATASET_DIR=/path/to/igvf_tf_collab/<dataset>
#   sbatch 08.0.run_modisco.sh            # dataset 0
#
# Prerequisites: 06.0.average_contrib_scores.sh must have completed.

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

score_types=("profile")  # counts modisco already done; add "counts" here to redo/extend

mkdir -p "${log_dir}"
for score_type in "${score_types[@]}"; do
    mkdir -p "${averaged_dir}/${dataset}/modisco/${score_type}_report"
done

load_render_modules

activate_env "${CONDA_ENV}"

metadata_start "08.0.run_modisco"
metadata_inputs+=( "contributions=${averaged_dir}/${dataset}" )
metadata_outputs+=( "motifs=${averaged_dir}/${dataset}/modisco/modisco_counts_results.h5" )
metadata_params+=( "dataset=${dataset}" )


for score_type in "${score_types[@]}"; do
    echo "[$(date)] Dataset ${dataset}: running MoDISco on averaged ${score_type} scores..."

    modisco motifs \
        -i "${averaged_dir}/${dataset}/${dataset}_average_shaps.${score_type}.h5" \
        -n 500000 \
        -o "${averaged_dir}/${dataset}/modisco/modisco_${score_type}_results.h5" \
        -w 500 \
        -v

    modisco report \
        -i "${averaged_dir}/${dataset}/modisco/modisco_${score_type}_results.h5" \
        -o "${averaged_dir}/${dataset}/modisco/${score_type}_report" \
        -s "${averaged_dir}/${dataset}/modisco/${score_type}_report" \
        -m "${ref_db_meme}"

    echo "[$(date)] Dataset ${dataset}: ${score_type} MoDISco complete"
done
