#!/bin/bash
# shellcheck disable=SC2218  # false positive in shellcheck 0.11.0: helpers come from common.sh
#SBATCH --job-name=qc_signal_peaks
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 01.0.qc_signal_peaks.sh
# Purpose: QC the signal and peaks BEFORE committing GPU time to them.
#
# Runs on the two artifacts training will actually use -- the prepared bigwig
# from 00.0 and the filtered narrowPeak from 01.0 -- so the numbers describe
# what ChromBPNet will see, after every filter. Reports:
#
#   TSS enrichment          is this accessibility data at all?
#   profile at peak summits does the signal agree with the peaks?
#   peaks, widths, coverage how much is there
#   signal per peak         quantiles, and how many peaks have NO signal
#   fraction in peaks       signal-to-background for the model's regions
#   total insertions        against the depth ChromBPNet needs
#
# Nothing here fails the pipeline: it is advisory, and the point is to read it
# before 02 and the GPU steps. A high fraction of zero-signal peaks or a TSS
# enrichment near 1 means something is wrong with the inputs, not the model.
#
# Usage:
#   export DATASET=<name>
#   cd workflows/SLURM && sbatch 01.0.qc_signal_peaks.sh
#
# Prerequisites: 00.0.prepare_signal.sh and 00.1.preprocess_peaks.sh.

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

dataset="${datasets[0]}"
signal_bw="${data_path}/signal/data_unstranded.bw"
peaks_np="${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
qc_dir="${results_path}/plots/signal_qc"

metadata_start "01.0.qc_signal_peaks"
metadata_inputs+=( "bigwig=${signal_bw}" "peaks=${peaks_np}" )
metadata_outputs+=( "qc_json=${qc_dir}/${dataset}_signal_qc.json" )
metadata_outputs+=( "qc_tsv=${qc_dir}/${dataset}_signal_qc.tsv" )
metadata_outputs+=( "profile_peaks=${qc_dir}/${dataset}_profile_peaks.pdf" )
metadata_outputs+=( "profile_tss=${qc_dir}/${dataset}_profile_tss.pdf" )
metadata_params+=( "peak_type=${peak_type}" "input_window=${chrombpnet_input_window}" )
[[ -n "${tss_bed}" && -f "${tss_bed}" ]] && metadata_inputs+=( "tss=${tss_bed}" )
require_input "${signal_bw}" 00.0.prepare_signal.sh
require_input "${peaks_np}"  00.1.preprocess_peaks.sh
preflight_check

activate_env "${preprocess_conda}"

# TSS enrichment needs a TSS list. `cli.py download-references` derives one from
# refGene; without it the rest of the QC still runs.
tss_args=()
if [[ -n "${tss_bed}" && -f "${tss_bed}" ]]; then
    tss_args+=( --tss "${tss_bed}" )
else
    echo "[$(date)] no TSS list at ${tss_bed:-<unset>}; skipping TSS enrichment"
    echo "           (re-run: cli.py download-references)"
fi

mkdir -p "${qc_dir}"

python "${src_dir}/cli.py" qc-signal \
    --bigwig       "${signal_bw}" \
    --peaks        "${peaks_np}" \
    "${tss_args[@]+"${tss_args[@]}"}" \
    --input-window "${chrombpnet_input_window}" \
    --out-dir      "${qc_dir}" \
    --prefix       "${dataset}" \
    --metadata-dir "${metadata_dir}"

echo "[$(date)] QC written to ${qc_dir}/ — read it before running 02 and the GPU steps."
