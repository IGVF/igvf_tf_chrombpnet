#!/bin/bash
#SBATCH --job-name=finemo_unified
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,owners
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 10.0.run_finemo_unified.sh
# Purpose: Call motif hits using the unified (compendium) MoDISco H5.
#          Uses modisco_compiled.h5 built in step 09 and fold-averaged
#          contribution scores (step 07) so that a single hit set per
#          dataset is produced, enabling direct cross-dataset comparisons.
#
#          One SLURM array job per dataset.
#
# Input per dataset:
#   {averaged_dir}/{dataset}/{dataset}_average_shaps.counts.h5  – averaged DeepLIFT scores (step 07)
#   modisco_compiled.h5                                 – unified MoDISco patterns (step 09)
#
# Output (inside finemo_unified_dir/{dataset}_{peak_type}/):
#   hits.bed.gz + hits.bed.gz.tbi     – tabix-indexed hit calls
#   hits.tsv                          – full hit table
#   finemo_report/                    – HTML report
#
# Usage:
#   sbatch 10.0.run_finemo_unified.sh            # dataset 0
#   sbatch 10.0.run_finemo_unified.sh            # all datasets (array 0-4)
#   sbatch --array=0 10.0.run_finemo_unified.sh   # dataset 0 only (override with --array=0 if needed)
#
# Prerequisites: steps 06 and 09 must have completed.
#   Requires the 'finemo' conda environment.

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

# Use the cross-dataset compendium built by 09.0.cross_dataset_compendium.sh,
# not the per-dataset one in ${modisco_compiled_dir}.
compiled_h5="${REPO_ROOT}/results/compendium/modisco_compiled/modisco_compiled.h5"
if [[ ! -f "${compiled_h5}" ]]; then
    echo "ERROR: ${compiled_h5} not found. Run 09.0.cross_dataset_compendium.sh first." >&2
    exit 1
fi

ml devel
ml system
ml cuda/11.5.0
ml cudnn/8.6.0.163
ml biology samtools


metadata_start "10.0.run_finemo_unified"



counts_h5="${averaged_dir}/${dataset}/${dataset}_average_shaps.counts.h5"

# Use the filtered peak list chrombpnet itself wrote during step 05
# (interpretation.interpreted_regions.bed), not the raw *_peaks_no_blacklist.narrowPeak.
# chrombpnet's contribs_bw silently drops peaks whose input window runs off a
# chromosome end (e.g. chrM), so the averaged H5 in step 06/07 has fewer regions
# than the raw peaks file. All folds filter identically (same peaks + genome), so
# fold 0's interpreted_regions.bed matches the H5 row-for-row.
peaks_file="${full_model_dir_selected}/${dataset}_${peak_type}_fold_${folds[0]}/interpretation/interpretation.interpreted_regions.bed"

if [[ ! -f "${counts_h5}" ]]; then
    echo "ERROR: ${counts_h5} not found. Run 06.0.average_contrib_scores.sh first." >&2
    exit 1
fi

if [[ ! -f "${peaks_file}" ]]; then
    echo "ERROR: ${peaks_file} not found. Run 05.0.get_contrib_scores.sh first." >&2
    exit 1
fi

out_dir="${finemo_unified_dir}/${dataset}_${peak_type}"
hits_file="${out_dir}/hits.bed.gz"


metadata_inputs+=( "contributions=${counts_h5}" "compiled_h5=${compiled_h5}" "peaks=${peaks_file}" )
require_input "${counts_h5}" 06.0.average_contrib_scores.sh
require_input "${compiled_h5}" 09.0.cross_dataset_compendium.sh
require_input "${peaks_file}" 05.0.get_contrib_scores.sh
preflight_check

activate_env "${finemo_conda}"
gpu_env
metadata_outputs+=( "hits=${hits_file}" )
metadata_params+=( "alpha=${finemo_alpha}" "dataset=${dataset}" )
if [[ -f "${hits_file}" ]]; then
    echo "[${dataset}] Hit calls already exist, skipping."
    exit 0
fi

finemo_npz="${out_dir}/intermediate_inputs.npz"
mkdir -p "${out_dir}"

echo "[$(date)] [${dataset}] Extracting regions from averaged scores..."

finemo extract-regions-chrombpnet-h5 \
    --h5s          "${counts_h5}" \
    --peaks        "${peaks_file}" \
    --out-path     "${finemo_npz}" \
    --region-width 1000
if [[ $? -ne 0 ]]; then
    echo "ERROR: [${dataset}] extract-regions-chrombpnet-h5 failed." >&2
    exit 1
fi

echo "[$(date)] [${dataset}] Calling hits (unified modisco)..."

finemo call-hits \
    -r "${finemo_npz}" \
    -m "${compiled_h5}" \
    -l "${finemo_alpha}" \
    -o "${out_dir}" \
    -b 200
if [[ $? -ne 0 ]]; then
    echo "ERROR: [${dataset}] call-hits failed." >&2
    exit 1
fi

if [[ -f "${out_dir}/hits.bed" ]]; then
    echo "[$(date)] [${dataset}] Compressing and indexing hits..."
    bgzip -c "${out_dir}/hits.bed" > "${hits_file}"
    tabix -p bed "${hits_file}"
fi

echo "[$(date)] [${dataset}] Fi-NeMo (unified) complete."
echo "  Hits: ${hits_file}"
