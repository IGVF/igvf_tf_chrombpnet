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
# Purpose: DeepSHAP contribution scores for BOTH heads (counts and profile) on
#          all of a dataset's filtered peaks, with each fold's bias-corrected
#          chrombpnet_nobias model: `chrombpnet contribs_bw` from chrombpnet
#          2.x. One SLURM array job per fold; each job processes all datasets.
#          These are the analysis-grade scores 06.0 averages across folds
#          (04.4's are a 30K-peak QC subsample).
#
# Why a different --shap-seed per fold (1234 + fold): chrombpnet 2.x seeds the
#   20 dinucleotide-shuffled DeepSHAP references from --shap-seed and each
#   sequence's content. With one seed for every fold, all five models would
#   score a peak against the SAME references, and 06.0's fold average would no
#   longer average out reference noise the way the unseeded 1.x runs did. The
#   seed goes into the run metadata and interpretation.interpret.args.json.
#
# Why require_gpu: contribs_bw has no --device flag (only the training
#   commands do), and JAX falls back to the CPU with at most a warning, so on a
#   node where JAX sees no GPU, DeepSHAP on every peak would run on the CPU
#   until the time limit. require_gpu stops the job before that.
#
# Skip rule: a dataset is skipped when both score h5s AND
#   interpretation.profile_scores.bw exist. contribs_bw writes, in order:
#   interpret.args.json, interpreted_regions.bed, the counts and profile h5s,
#   counts_scores.bw, profile_scores.bw -- so that bigwig is the last file.
#   The h5s are written as *.partial and renamed on completion, so a killed job
#   never leaves one under its final name. The bigwigs are written in place: a
#   job killed while writing profile_scores.bw leaves a truncated file that
#   counts as done, so delete it to redo that dataset. A rerun overwrites every
#   output.
#
# Memory: --mem=128G is a 1.x-era request, sized for 1.x holding every
#   region's scores in memory at once. 2.x streams the scores to disk as it
#   computes them, but still one-hot encodes all regions up front and loads a
#   head's whole projected_shap array to write each bigwig. Re-measure (e.g.
#   sacct MaxRSS) before changing the number.
#
# Array index = fold index.
#
# Input:  ${full_model_dir}/{dataset}_{peak_type}_fold_{fold}/models/chrombpnet_nobias.h5  (04.0)
#         ${peaks_dir}/{dataset}_{peak_type}_peaks_no_blacklist.narrowPeak                (00.1)
# Output per dataset/fold, in ${full_model_dir}/{dataset}_{peak_type}_fold_{fold}/interpretation/:
#   interpretation.{counts,profile}_scores.h5  DeepSHAP scores; 06.0 averages these
#   interpretation.{counts,profile}_scores.bw  the same, per fold, as bigwigs
#   interpretation.interpreted_regions.bed     the peaks actually scored: contribs_bw
#                                              drops any whose 2114 bp window runs off
#                                              a chromosome end; 07.0 and 10.0 read it
#   interpretation.interpret.args.json         arguments, DeepSHAP seed, chrombpnet
#                                              version, JAX backend and devices
#
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 05.0.get_contrib_scores.sh            # all folds (array 0-4)
#   sbatch --array=0 05.0.get_contrib_scores.sh  # fold 0 only
#
# Prerequisites: 04.0.train_full_model.sh for the fold; the chrombpnet 2.x
#   environment (CHROMBPNET_REPO, see lib/bash/common.sh); a GPU that JAX can
#   use (require_gpu says why not, if it cannot).

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

# One DeepSHAP reference seed per fold -- see the header.
shap_seed=$(( 1234 + fold ))

metadata_start "05.0.get_contrib_scores"
metadata_params+=( "fold=${fold}" "shap_seed=${shap_seed}" )
for _ds in "${datasets[@]}"; do
    _dir="${full_model_dir}/${_ds}_${peak_type}_fold_${fold}"
    _prefix="${_dir}/interpretation/interpretation"
    _peaks="${peaks_dir}/${_ds}_${peak_type}_peaks_no_blacklist.narrowPeak"
    metadata_inputs+=( "model=${_dir}/models/chrombpnet_nobias.h5" "peaks=${_peaks}" )
    for _head in counts profile; do
        metadata_outputs+=( "contributions=${_prefix}.${_head}_scores.h5" "contributions=${_prefix}.${_head}_scores.bw" )
    done
    metadata_outputs+=( "peaks=${_prefix}.interpreted_regions.bed" "interpretation_settings=${_prefix}.interpret.args.json" )
    require_input "${_dir}/models/chrombpnet_nobias.h5" 04.0.train_full_model.sh
    require_input "${_peaks}" 00.1.preprocess_peaks.sh
done
unset _ds _dir _prefix _peaks _head
metadata_inputs+=( "genome=${genome_fa}" "chrom_sizes=${chrom_sizes}" )
require_input "${genome_fa}" "cli.py download-references"
require_input "${chrom_sizes}" "cli.py download-references"
preflight_check

activate_env "${chrombpnet_env}"
gpu_env
require_gpu jax

echo "[$(date)] Fold ${fold}: DeepSHAP (counts + profile, --shap-seed ${shap_seed}) for datasets [${datasets[*]}]"
for dataset in "${datasets[@]}"; do
    model_file="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}/models/chrombpnet_nobias.h5"
    interp_dir="${full_model_dir}/${dataset}_${peak_type}_fold_${fold}/interpretation"
    peaks_file="${peaks_dir}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
    prefix="${interp_dir}/interpretation"
    last_output="${prefix}.profile_scores.bw"   # the last file contribs_bw writes

    echo "[$(date)] [${dataset} fold ${fold}] Computing contribution scores"
    echo "  model : ${model_file}"
    echo "  output: ${interp_dir}/"

    if [[ -f "${prefix}.counts_scores.h5" && -f "${prefix}.profile_scores.h5" && -f "${last_output}" ]]; then
        echo "  Already done, skipping."
        continue
    fi

    mkdir -p "${interp_dir}"

    chrombpnet contribs_bw \
        -m "${model_file}" \
        -r "${peaks_file}" \
        -g "${genome_fa}" \
        -c "${chrom_sizes}" \
        -op "${prefix}" \
        --shap-seed "${shap_seed}"
    _rc=$?
    # No `set -e` in this step: guard the call and its last output explicitly,
    # or a failed dataset would fall through to the next and the job exit 0.
    if [[ ${_rc} -ne 0 || ! -f "${last_output}" ]]; then
        echo "ERROR: chrombpnet contribs_bw failed for ${dataset} fold ${fold} (exit ${_rc}); ${last_output} was not written." >&2
        exit 1
    fi

    echo "[$(date)] [${dataset} fold ${fold}] Done."
done

echo "[$(date)] Fold ${fold}: contribution score computation complete."
