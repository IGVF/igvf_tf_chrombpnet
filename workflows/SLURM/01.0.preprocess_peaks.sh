#!/bin/bash
#SBATCH --job-name=preprocess_peaks
# bedtools intersect streams the peaks and loads only the small blacklist, so
# memory is tiny and peak-count-independent (~22 MB observed).
#SBATCH --mem=2G
#SBATCH --time=6:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 01.0.preprocess_peaks.sh
# Purpose: Remove blacklisted regions from peak files and reformat to
#          narrowPeak for chrombpnet (summit = midpoint of peak).
#
# The bedtools+awk pipeline this used to run inline now lives in
# src/preprocess_peaks.py (pyranges1), so there is no bedtools dependency and
# the bedtools equivalence is covered by tests/test_intervals.py.

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
set -euo pipefail

mkdir -p "${data_path}"

metadata_start "01.0.preprocess_peaks"


for dataset in "${datasets[@]}"; do
    echo "Processing peaks for ${dataset}..."
    # --blacklist also accepts the ENCODE accession (${blacklist_accession}),
    # which fetches it directly; the local copy is the default because compute
    # nodes may have no outbound network.
    python "${src_dir}/cli.py" preprocess-peaks \
        --peaks        "${regions}" \
        --blacklist    "${blacklist}" \
        --chrom-sizes  "${chrom_sizes}" \
        --input-window "${chrombpnet_input_window}" \
        --out-dir      "${data_path}" \
        --prefix       "${dataset}_${peak_type}"
    metadata_inputs+=( "regions=${regions}" )
    metadata_inputs+=( "blacklist=${blacklist}" )
    metadata_inputs+=( "chrom_sizes=${chrom_sizes}" )
    require_input "${regions}" ""
    require_input "${blacklist}" scripts/bash/download_references.sh
require_input "${chrom_sizes}" scripts/bash/download_references.sh
preflight_check

activate_env "${preprocess_conda}"
    metadata_outputs+=( "narrowpeak=${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak" )
    metadata_outputs+=( "bed=${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.bed" )
    metadata_params+=( "peak_type=${peak_type}" "input_window=${chrombpnet_input_window}" )
done

echo "Done: 01.0.preprocess_peaks.sh"
