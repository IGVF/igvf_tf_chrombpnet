#!/bin/bash
#SBATCH --job-name=copy_peaks_fragments
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 00.copy_and_prepare_data.sh
# Purpose: Stage per-dataset inputs — copy the peak BED and ATAC fragments into
#   each dataset's data/ folder, then filter fragments to main chromosomes
#   (${id}_atac_fragments_main_chrs.tsv.gz, the input consumed by steps 03/04).
#
# NOTE: this is a per-dataset staging TEMPLATE — edit the source paths and the
#   dataset/cluster/target arrays below for your own data. The paths it writes
#   must match `regions` and `signal_path` in config/<dataset>/config.yaml. Shared genome,
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

activate_env "${preprocess_conda}"

metadata_start "00.copy_and_prepare_data"

fragments_in="/oak/stanford/groups/engreitz/Projects/IGVF-E2GPillarProject/QC_pseudobulks/multiome_data";
peaks_in="/oak/stanford/groups/engreitz/Users/kaybrand/scE2G_preprint/scE2G/results/uniformly_processed";

out_path="/oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab";

datasets=( "igvf6" "igvf11" "igvf3" );
clusters=( "definitive_endoderm" "h7" "h9_cardio_cardiomyocte_d8" );
target_ids=( "igvf6_definitive_endoderm" "igvf11_h7_hesc" "igvf3_cardiomyocyte" );

for i in "${!datasets[@]}"
do
    dataset="${datasets[$i]}"
    cluster="${clusters[$i]}"
    target_id="${target_ids[$i]}"

    mkdir -p ${out_path}/${target_id}/data/peaks;
    mkdir -p ${out_path}/${target_id}/data/fragments;

    cp ${peaks_in}/${dataset}/${cluster}/Peaks/macs2_peaks.narrowPeak.sorted.candidateRegions.bed \
        ${out_path}/${target_id}/data/peaks/${target_id}_all_peaks.bed

    cp ${fragments_in}/${dataset}/${cluster}/atac_fragments_${dataset}_${cluster}.tsv.gz \
        ${out_path}/${target_id}/data/fragments/${target_id}_atac_fragments.tsv.gz

    # Main-chromosome filter moved to src/filter_fragments.py: the chromosome set
    # is explicit and tested instead of living in a regex, and the output is
    # bgzip (still valid gzip) so it can carry a tabix index.
    python "${src_dir}/cli.py" filter-fragments \
        --input  ${out_path}/${target_id}/data/fragments/${target_id}_atac_fragments.tsv.gz \
        --output ${out_path}/${target_id}/data/fragments/${target_id}_atac_fragments_main_chrs.tsv.gz

    metadata_inputs+=( "fragments_raw=${out_path}/${target_id}/data/fragments/${target_id}_atac_fragments.tsv.gz" )
    metadata_outputs+=( "fragments_main_chrs=${out_path}/${target_id}/data/fragments/${target_id}_atac_fragments_main_chrs.tsv.gz" )
    metadata_outputs+=( "peaks=${out_path}/${target_id}/data/peaks/${target_id}_all_peaks.bed" )
done
