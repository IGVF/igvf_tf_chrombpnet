#!/bin/bash
# shellcheck disable=SC2218  # false positive in shellcheck 0.11.0: these helpers all
# come from lib/bash/common.sh, sourced above.
#SBATCH --job-name=copy_peaks_fragments
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 00.copy_and_prepare_data.sh
# Purpose: stage one dataset's inputs into the paths its config declares, then
#   filter the fragments to the main chromosomes.
#
# OPTIONAL. `regions` and `signal_path` in config/<dataset>/config.yaml can point
# at data wherever it already lives; this step exists only for the case where it
# has to be copied in and filtered first. If your data is already in place, skip
# straight to 01.
#
# This is a TEMPLATE: the two source paths below are the part you edit. Where it
#   writes is NOT editable here -- it comes from the config, so the staged files
#   necessarily match what the rest of the pipeline reads. Shared genome,
#   chrom.sizes, blacklist and motif references are NOT staged here; fetch those
#   once into the lab Data/ folder with:  bash scripts/bash/download_references.sh

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

# Cross-dataset staging template: no DATASET_DIR, so source common.sh directly.
# shellcheck source=lib/bash/common.sh
source "${REPO_ROOT}/lib/bash/common.sh" || exit 1

metadata_start "00.copy_and_prepare_data"

# ── Edit these two for your data ──────────────────────────────────────────────
# Where the unstaged inputs currently live. Everything else is derived from
# config/${DATASET}/config.yaml.
source_peaks="${SOURCE_PEAKS:-/oak/stanford/groups/engreitz/Users/kaybrand/scE2G_preprint/scE2G/results/uniformly_processed/igvf3/h9_cardio_cardiomyocte_d8/Peaks/macs2_peaks.narrowPeak.sorted.candidateRegions.bed}"
source_fragments="${SOURCE_FRAGMENTS:-/oak/stanford/groups/engreitz/Projects/IGVF-E2GPillarProject/QC_pseudobulks/multiome_data/igvf3/h9_cardio_cardiomyocte_d8/atac_fragments_igvf3_h9_cardio_cardiomyocte_d8.tsv.gz}"
# ──────────────────────────────────────────────────────────────────────────────

# Unfiltered fragments land next to the filtered ones the config points at.
staged_fragments="$(dirname "${signal_path}")/${dataset_name}_atac_fragments_unfiltered.tsv.gz"

metadata_inputs+=( "source_peaks=${source_peaks}" "source_fragments=${source_fragments}" )
metadata_outputs+=( "regions=${regions}" "signal=${signal_path}" )
require_input "${source_peaks}"     ""
require_input "${source_fragments}" ""
preflight_check

activate_env "${preprocess_conda}"

mkdir -p "$(dirname "${regions}")" "$(dirname "${signal_path}")"

echo "[$(date)] [${dataset_name}] Staging inputs"
echo "  peaks     : ${source_peaks}"
echo "           -> ${regions}"
cp -f "${source_peaks}" "${regions}"

if [[ -f "${signal_path}" ]]; then
    echo "  fragments : already filtered, skipping (${signal_path})"
else
    echo "  fragments : ${source_fragments}"
    echo "           -> ${staged_fragments}"
    cp -f "${source_fragments}" "${staged_fragments}"

    # Main-chromosome filter: the chromosome set is explicit and tested rather
    # than living in a regex, and the output is bgzip so it can carry a tabix index.
    python "${src_dir}/cli.py" filter-fragments \
        --input        "${staged_fragments}" \
        --output       "${signal_path}" \
        --metadata-dir "${metadata_dir}"
fi

echo "[$(date)] [${dataset_name}] Done."
