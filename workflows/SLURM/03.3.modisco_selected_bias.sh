#!/bin/bash
#SBATCH --job-name=modisco_selected_bias
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# CPU only: seqlets, tomtom-lite and TF-MoDISco use no GPU. Same partition
# and QOS as 04.5/08.0 -- the default QOS caps walltime at 2 days.
#SBATCH --time=4-0
#SBATCH --partition=engreitz
#SBATCH --qos=high_p
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 03.3.modisco_selected_bias.sh
# Purpose: Motif QC on only the bias model selected per fold (fold_bias_suffix,
#   populated by 03.1.select_bias.sh), CPU half of the selected-bias QC:
#   src/motif_qc.py -- TF-MoDISco 2.5.2 on a 5000-seqlet budget, its
#   descriptive report and a MEME export -- on 03.2's scores for BOTH heads.
#   This verifies the bias model learned Tn5 and nothing else: the profile
#   head should show Tn5 patterns, and the counts head no TF motifs (on d0 it
#   shows GC-rich positive / AT-rich negative composition patterns).
#   Deliberately not run on the full fold x bias-factor sweep (03.0).
#
# Why not chrombpnet's modisco: see the header of src/motif_qc.py -- a small
#   budget (6 min, not an hour), both heads, and the 2.5.2 report. It runs in
#   the `motifs` env (envs/motifs.yml): modisco 2.5.2 needs Python >= 3.9,
#   which the chrombpnet env cannot have. The report matches against
#   chrombpnet's own motif DB, which, unlike MotifCompendium, has Tn5/DNase
#   references. chrombpnet's *_bias_profile.pdf is no longer written; 03.2's
#   metrics and plots are unchanged.
#
# Array index = fold index (one task per fold, not per bias-factor).
#
# Input:  03.2's auxiliary/interpret_subsample/<prefix>_bias.{profile,counts}_scores.h5
# Output: evaluation/motif_qc_{profile,counts}/<prefix>_bias_{modisco_results.h5,
#         modisco_motifs.meme,motif_qc.json} and report/report.html
# Settings: motif_qc_window (default 400) and motif_qc_max_seqlets (5000) in
#   config.yaml; MOTIF_QC_WINDOW / MOTIF_QC_MAX_SEQLETS override them.
#
# Usage:
#   export DATASET_DIR=/path/to/igvf_tf_collab/<dataset>
#   sbatch 03.3.modisco_selected_bias.sh            # all folds (array 0-4)
#   sbatch --array=0 03.3.modisco_selected_bias.sh  # fold 0 only (quick test)
#
# Prerequisites: 03.2.qc_selected_bias.sh for the fold, the `motifs` env
#   (MOTIFS_ENV), and `cli.py download-references` (chrombpnet's motif DB).

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



metadata_start "03.3.modisco_selected_bias"

file_prefix="${bias_dataset}_${peak_type}_fold_${fold}"
out_dir="${results_path}/bias_models/bias_model${suffix}/${bias_dataset}_${peak_type}_fold_${fold}"
interp="${out_dir}/auxiliary/interpret_subsample"
window="${MOTIF_QC_WINDOW:-${motif_qc_window:-400}}"
max_seqlets="${MOTIF_QC_MAX_SEQLETS:-${motif_qc_max_seqlets:-5000}}"

for _head in profile counts; do
    _qc="${out_dir}/evaluation/motif_qc_${_head}"
    metadata_inputs+=( "contributions=${interp}/${file_prefix}_bias.${_head}_scores.h5" )
    metadata_outputs+=( "motifs=${_qc}/${file_prefix}_bias_modisco_results.h5" "motif_report=${_qc}/report/report.html" )
    require_input "${interp}/${file_prefix}_bias.${_head}_scores.h5" 03.2.qc_selected_bias.sh
done
metadata_inputs+=( "motif_db=${chrombpnet_motifs_meme}" )
require_input "${chrombpnet_motifs_meme}" "cli.py download-references"
preflight_check

activate_env "${chrombpnet_env}"
metadata_params+=( "fold=${fold}" "bias_suffix=${suffix}" "scores=profile,counts" "max_seqlets=${max_seqlets}" "window=${window}" )
echo "[$(date)] [fold ${fold}] Motif QC on selected bias model (suffix ${suffix}): ${out_dir}"

# No `set -e` in this step, so each call and its report are guarded. No GPU
# is used here at all, which is the whole reason this is a separate step from
# 03.2. high_p is what gets the longer walltime: the default QOS caps at 2 days.
METADATA_RSS_FILE="${out_dir}/.peak_rss_gb_033"   # TF-MoDISco runs as a child; its peak counts
export METADATA_RSS_FILE
for head in profile counts; do
    qc_dir="${out_dir}/evaluation/motif_qc_${head}"
    echo "[$(date)] [fold ${fold}] TF-MoDISco, ${head} head -> ${qc_dir}"
    python "${src_dir}/motif_qc.py" \
        --scores "${interp}/${file_prefix}_bias.${head}_scores.h5" \
        --motif-db "${chrombpnet_motifs_meme}" \
        --output-dir "${qc_dir}" \
        --prefix "${file_prefix}_bias" \
        --window "${window}" \
        --max-seqlets "${max_seqlets}" \
        --threads "${SLURM_CPUS_PER_TASK:-4}"
    _rc=$?
    if [[ ${_rc} -ne 0 || ! -f "${qc_dir}/report/report.html" ]]; then
        echo "ERROR: motif_qc.py failed for fold ${fold}, ${head} head; ${qc_dir}/report/report.html was not written." >&2
        exit 1
    fi
done
[[ -s "${METADATA_RSS_FILE}" ]] && metadata_metrics+=( "peak_rss_gb=$(<"${METADATA_RSS_FILE}")" )

echo "[$(date)] [fold ${fold}] Done."
