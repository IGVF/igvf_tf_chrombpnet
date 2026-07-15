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
#   dataset/cluster/target arrays below for your own data. Shared genome,
#   chrom.sizes, blacklist and motif references are NOT staged here; fetch those
#   once into the lab Data/ folder with:  bash scripts/bash/download_references.sh

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

    zcat ${out_path}/${target_id}/data/fragments/${target_id}_atac_fragments.tsv.gz | \
        grep -P '^chr([1-9]|1[0-9]|2[0-2]|X|Y|M)\t' | \
        gzip > ${out_path}/${target_id}/data/fragments/${target_id}_atac_fragments_main_chrs.tsv.gz
done
