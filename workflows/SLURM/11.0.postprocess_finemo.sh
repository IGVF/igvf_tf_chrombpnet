#!/bin/bash
#SBATCH --job-name=finemo_postprocess
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 11.0.postprocess_finemo.sh
# Purpose: Generate the Fi-NeMo report for a dataset's unified hit calls:
#            finemo report - compute per-motif instance-CWM vs input-CWM
#                             similarity (cwm_similarity in motif_report.tsv)
#
# Input (from step 10):
#   {finemo_unified_dir}/{dataset}_{peak_type}/hits.tsv
#   {finemo_unified_dir}/{dataset}_{peak_type}/intermediate_inputs.npz
#
# Output:
#   {finemo_unified_dir}/{dataset}_{peak_type}/finemo_report/motif_report.tsv
#   (consumed by analysis/0.2.finemo_hit_qc.py)
#
# Usage (DATASET_DIR must be exported):
#   cd workflows/SLURM
#   DATASET_DIR=/path/to/igvf3_cardiomyocyte sbatch 11.0.postprocess_finemo.sh
#
# Prerequisites: 10.0.run_finemo_unified.sh must have completed for this dataset.

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

out_dir="${finemo_unified_dir}/${dataset}_${peak_type}"
hits_tsv="${out_dir}/hits.tsv"
finemo_npz="${out_dir}/intermediate_inputs.npz"
report_dir="${out_dir}/finemo_report"

if [[ -f "${report_dir}/motif_report.tsv" ]]; then
    echo "[${dataset}] Already post-processed, skipping."
    exit 0
fi

if [[ ! -f "${hits_tsv}" ]]; then
    echo "ERROR: ${hits_tsv} not found. Run 10.0.run_finemo_unified.sh first." >&2
    exit 1
fi

if [[ ! -f "${finemo_npz}" ]]; then
    echo "ERROR: ${finemo_npz} not found. Run 10.0.run_finemo_unified.sh first." >&2
    exit 1
fi

ml biology samtools


metadata_start "11.0.postprocess_finemo"
metadata_inputs+=( "hits_tsv=${hits_tsv}" "npz=${finemo_npz}" )
require_input "${hits_tsv}" 10.0.run_finemo_unified.sh
require_input "${finemo_npz}" 10.0.run_finemo_unified.sh
preflight_check

activate_env "${finemo_conda}"
metadata_outputs+=( "motif_report=${report_dir}/motif_report.tsv" )
metadata_params+=( "dataset=${dataset}" )


# Compute cwm_similarity for each motif by comparing the average CWM
# reconstructed from called instances against the input (modisco) CWM.
echo "[$(date)] [${dataset}] Running finemo report..."
finemo report \
    -r "${finemo_npz}" \
    -H "${out_dir}" \
    -o "${report_dir}"

motif_report="${report_dir}/motif_report.tsv"
if [[ ! -f "${motif_report}" ]]; then
    echo "ERROR: ${motif_report} not produced. Check finemo report output above." >&2
    exit 1
fi
echo "[$(date)] [${dataset}] Report done: ${motif_report}"
