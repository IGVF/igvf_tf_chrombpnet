#!/bin/bash
#SBATCH --job-name=qc_datasets
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# qc_datasets.sh
# Purpose: Dataset-level QC across the collaboration's training datasets, via
#          src/qc_datasets.py: total fragments, cells, FRiP, peaks, fragment
#          size distribution, per-chromosome counts and a ChromBPNet
#          training-readiness check.
#
# Cross-dataset: no DATASET or DATASET_DIR. The dataset list comes from
# utils.palettes (DATASET_LABELS_QC) and every path, input and output, is
# built from core_path at the top of src/qc_datasets.py.
#
# Runs in ${preprocess_env}, this repo's preprocess pixi environment:
# qc_datasets.py imports pyranges1 (for FRiP, through utils.intervals), which
# needs Python >= 3.12 and is installed only there.
#
# Input (per dataset, under core_path/<dataset>/):
#   data/fragments/<dataset>_atac_fragments_main_chrs.tsv.gz
#   results/preprocessing/<dataset>_all_peaks_no_blacklist.narrowPeak
#   and core_path/genome/hg38.chrom.sizes
# Output: core_path/results/plots/dataset_qc/ -- summary and per-metric TSVs,
#   plots as .pdf + .png
#
# Usage:
#   cd workflows/SLURM && sbatch qc_datasets.sh
#
# Prerequisites: `pixi install -e preprocess` in this checkout, and read access
#   to core_path.

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

# Cross-dataset step: no DATASET_DIR, so source common.sh directly rather than
# config.sh. (This is what the dummy 'export DATASET_DIR=<any dataset>' used to
# work around.)
# shellcheck source=lib/bash/common.sh
source "${REPO_ROOT}/lib/bash/common.sh" || exit 1

activate_env "${preprocess_env}"

metadata_start "qc_datasets"


python "${src_dir}/qc_datasets.py"
