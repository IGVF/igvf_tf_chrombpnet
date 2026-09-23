#!/bin/bash
#SBATCH --job-name=modisco_full_model
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# CPU only: TF-MoDISco uses no GPU. Same partition and QOS as 03.3/08.0 --
# the default QOS caps walltime at 2 days; high_p is what gets longer runs.
#SBATCH --time=4-0
#SBATCH --partition=engreitz
#SBATCH --qos=high_p
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 04.5.modisco_full_model.sh
# Purpose: Per-fold interpretation QC of the full model, CPU half: TF-MoDISco
#   on 04.4's profile contribution scores, the motif report and its PDF, and
#   chrombpnet's pipeline-mode HTML report -- where `chrombpnet pipeline`
#   itself ends. Inspect evaluation/chrombpnet_nobias_profile.pdf: the model
#   WITHOUT bias should show TF motifs, and no Tn5 bias motif.
#
# Why its own step: see 04.4. TF-MoDISco is the long pole and needs no GPU.
#
# Array index = fold index.
#
# Input:  04.4's auxiliary/interpret_subsample/chrombpnet_nobias.profile_scores.h5
# Output: auxiliary/interpret_subsample/modisco_results_profile_scores.h5
#         evaluation/modisco_profile/motifs.html, evaluation/chrombpnet_nobias_profile.pdf
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 04.5.modisco_full_model.sh               # all folds
#   sbatch --array=0 04.5.modisco_full_model.sh     # fold 0 only
# Prerequisites: 04.4.qc_full_model_interpret.sh for the fold.

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


metadata_start "04.5.modisco_full_model"
metadata_params+=( "fold=${fold}" "scores=profile" "max_seqlets=50000" "window=500" )
for dataset in "${datasets[@]}"; do
    _dir="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}"
    metadata_inputs+=( "contributions=${_dir}/auxiliary/interpret_subsample/chrombpnet_nobias.profile_scores.h5" )
    metadata_outputs+=( "motifs=${_dir}/auxiliary/interpret_subsample/modisco_results_profile_scores.h5" )
    metadata_outputs+=( "motif_report=${_dir}/evaluation/chrombpnet_nobias_profile.pdf" )
    require_input "${_dir}/auxiliary/interpret_subsample/chrombpnet_nobias.profile_scores.h5" 04.4.qc_full_model_interpret.sh
done
require_input "${genome_fa}" "cli.py download-references"
preflight_check

activate_env "${CONDA_ENV}"

for dataset in "${datasets[@]}"; do
    out_dir="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}"
    echo "[$(date)] [${dataset} fold ${fold}] TF-MoDISco + reports on the full model: ${out_dir}"
    METADATA_RSS_FILE="${out_dir}/.peak_rss_gb_045"   # TF-MoDISco runs as a child; its peak counts
    export METADATA_RSS_FILE
    python "${src_dir}/run_full_model_qc.py" \
        --stage modisco \
        --model-dir "${out_dir}" \
        --genome "${genome_fa}" \
        --data-type "${assay}"
    _rc=$?
    [[ -s "${METADATA_RSS_FILE}" ]] && metadata_metrics+=( "peak_rss_gb=$(<"${METADATA_RSS_FILE}")" )
    _report="${out_dir}/evaluation/chrombpnet_nobias_profile.pdf"
    if [[ ${_rc} -ne 0 || ! -f "${_report}" ]]; then
        echo "ERROR: run_full_model_qc.py --stage modisco failed for ${dataset} fold ${fold}; ${_report} was not written." >&2
        exit 1
    fi
done

echo "[$(date)] Fold ${fold}: done."
