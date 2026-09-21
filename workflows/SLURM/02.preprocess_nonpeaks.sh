#!/bin/bash
#SBATCH --job-name=preprocess_nonpeaks
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 02.preprocess_nonpeaks.sh
# Purpose: Generate GC-matched negative (non-peak) regions for each
#          dataset x fold combination using 'chrombpnet prep nonpeaks'.
#

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


metadata_start "02.preprocess_nonpeaks"


set -euo pipefail

for dataset in "${datasets[@]}"; do
    for fold in "${folds[@]}"; do
        out_prefix="${data_path}/${dataset}/output_${peak_type}_fold_${fold}"
        negatives_file="${out_prefix}_negatives.bed"


metadata_inputs+=( "peaks=${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak" "genome=${genome_fa}" "chrom_sizes=${chrom_sizes}" "blacklist=${blacklist}" "fold_json=${folds_dir}/fold_${fold}.json" )
require_input "${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak" 01.preprocess_peaks.sh
require_input "${genome_fa}" scripts/bash/download_references.sh
require_input "${chrom_sizes}" scripts/bash/download_references.sh
require_input "${blacklist}" scripts/bash/download_references.sh
require_input "${folds_dir}/fold_${fold}.json"
preflight_check

activate_env "${CONDA_ENV}"
metadata_tools+=( "$(tool_version chrombpnet chrombpnet --version)" )
metadata_outputs+=( "negatives=${negatives_file}" )
metadata_params+=( "fold=${fold}" "peak_type=${peak_type}" )
        if [[ -f "${negatives_file}" ]]; then
            echo "Found existing negatives for ${dataset} fold_${fold}, skipping..."
            continue
        fi

        echo "Generating negatives for ${dataset} fold_${fold}..."
        chrombpnet prep nonpeaks \
            -g "${genome_fa}" \
            -p "${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak" \
            -c "${chrom_sizes}" \
            -fl "${folds_dir}/fold_${fold}.json" \
            -br "${blacklist}" \
            -o "${out_prefix}"

        echo "  -> written to ${negatives_file}"
    done
done

echo "Done: 02.preprocess_nonpeaks.sh"
