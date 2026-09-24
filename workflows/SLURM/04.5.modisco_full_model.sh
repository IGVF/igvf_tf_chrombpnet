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
# Purpose: Per-fold motif QC of the full model, CPU half: src/motif_qc.py --
#   TF-MoDISco 2.5.2 on a 5000-seqlet budget, its descriptive report and a
#   MEME export -- on 04.4's scores for BOTH heads. The model WITHOUT bias
#   should show TF motifs and no Tn5 bias motif in either head.
#
# Why its own step: see 04.4. None of this needs a GPU.
# Why not chrombpnet's modisco: see the header of src/motif_qc.py. It runs in
#   the `motifs` env (envs/motifs.yml), not the chrombpnet one: modisco 2.5.2
#   needs Python >= 3.9. The report matches against chrombpnet's motif DB,
#   which has Tn5/DNase references next to the TF motifs, so leftover bias
#   is named as such. chrombpnet's PDF and pipeline-mode HTML report are no
#   longer written -- they read the old report's motifs.html; 04.0's
#   train-mode report stays.
#
# Array index = fold index.
#
# Input:  04.4's auxiliary/interpret_subsample/chrombpnet_nobias.{profile,counts}_scores.h5
# Output: evaluation/motif_qc_{profile,counts}/chrombpnet_nobias_{modisco_results.h5,
#         modisco_motifs.meme,motif_qc.json} and report/report.html
# Settings: motif_qc_window (default 400) and motif_qc_max_seqlets (5000) in
#   config.yaml; MOTIF_QC_WINDOW / MOTIF_QC_MAX_SEQLETS override them.
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 04.5.modisco_full_model.sh               # all folds
#   sbatch --array=0 04.5.modisco_full_model.sh     # fold 0 only
# Prerequisites: 04.4.qc_full_model_interpret.sh for the fold, the `motifs`
#   env (MOTIFS_ENV), and `cli.py download-references` (chrombpnet's motif DB).

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
window="${MOTIF_QC_WINDOW:-${motif_qc_window:-400}}"
max_seqlets="${MOTIF_QC_MAX_SEQLETS:-${motif_qc_max_seqlets:-5000}}"
metadata_params+=( "fold=${fold}" "scores=profile,counts" "max_seqlets=${max_seqlets}" "window=${window}" )
for dataset in "${datasets[@]}"; do
    _dir="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}"
    for _head in profile counts; do
        _scores="${_dir}/auxiliary/interpret_subsample/chrombpnet_nobias.${_head}_scores.h5"
        _qc="${_dir}/evaluation/motif_qc_${_head}"
        metadata_inputs+=( "contributions=${_scores}" )
        metadata_outputs+=( "motifs=${_qc}/chrombpnet_nobias_modisco_results.h5" "motif_report=${_qc}/report/report.html" )
        require_input "${_scores}" 04.4.qc_full_model_interpret.sh
    done
done
metadata_inputs+=( "motif_db=${chrombpnet_motifs_meme}" )
require_input "${chrombpnet_motifs_meme}" "cli.py download-references"
preflight_check

activate_env "${motifs_conda}"

for dataset in "${datasets[@]}"; do
    out_dir="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}"
    METADATA_RSS_FILE="${out_dir}/.peak_rss_gb_045"   # TF-MoDISco runs as a child; its peak counts
    export METADATA_RSS_FILE
    for head in profile counts; do
        qc_dir="${out_dir}/evaluation/motif_qc_${head}"
        echo "[$(date)] [${dataset} fold ${fold}] TF-MoDISco on the full model, ${head} head -> ${qc_dir}"
        python "${src_dir}/motif_qc.py" \
            --scores "${out_dir}/auxiliary/interpret_subsample/chrombpnet_nobias.${head}_scores.h5" \
            --motif-db "${chrombpnet_motifs_meme}" \
            --output-dir "${qc_dir}" \
            --prefix chrombpnet_nobias \
            --window "${window}" \
            --max-seqlets "${max_seqlets}" \
            --threads "${SLURM_CPUS_PER_TASK:-4}"
        _rc=$?
        if [[ ${_rc} -ne 0 || ! -f "${qc_dir}/report/report.html" ]]; then
            echo "ERROR: motif_qc.py failed for ${dataset} fold ${fold}, ${head} head; ${qc_dir}/report/report.html was not written." >&2
            exit 1
        fi
    done
    [[ -s "${METADATA_RSS_FILE}" ]] && metadata_metrics+=( "peak_rss_gb=$(<"${METADATA_RSS_FILE}")" )
done

echo "[$(date)] Fold ${fold}: done."
