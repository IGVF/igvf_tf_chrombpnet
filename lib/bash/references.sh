#!/bin/bash
# shellcheck disable=SC2034  # everything here is consumed by the scripts that source it
# lib/bash/references.sh
#
# The single definition of WHERE the shared reference files live, WHAT they are
# called, and WHERE they came from.
#
# Two kinds of script source this, and that is the whole point:
#   - scripts/bash/download_references.sh  writes these paths
#   - lib/bash/common.sh                   reads them, so every workflow step sees them
# Before this file existed, the root was hardcoded separately in the downloader,
# in each dataset_config.sh and in common.sh, so the writer and the readers could
# drift apart silently.
#
# To point the pipeline at a different copy of the references, either edit
# REFERENCE_ROOT below or override it without touching the repo:
#
#   export REFERENCE_ROOT=/scratch/$USER/Data
#   bash scripts/bash/download_references.sh          # fetch into that root
#   cd workflows/SLURM && sbatch 01.0.preprocess_peaks.sh   # read from it
#
# A single dataset that needs a different genome can still override any variable
# below in its own dataset_config.sh, which config.sh sources afterwards.

# ── Root ──────────────────────────────────────────────────────────────────────
REFERENCE_ROOT="${REFERENCE_ROOT:-/oak/stanford/groups/engreitz/Data}"

genome_build="hg38"
genome_path="${REFERENCE_ROOT}/${genome_build}"
sequence_dir="${genome_path}/Sequence"
chrom_sizes_dir="${sequence_dir}/chrom_sizes"
blacklist_dir="${genome_path}/blacklist"
motif_dir="${REFERENCE_ROOT}/motif"

# ── Genome: IGVF GRCh38 no-alt analysis set ───────────────────────────────────
genome_accession="IGVFFI0653VCGH"
genome_url="https://api.data.igvf.org/reference-files/${genome_accession}/@@download/${genome_accession}.fasta.gz"
genome_metadata_url="https://api.data.igvf.org/reference-files/${genome_accession}/?format=json"
genome_fa_gz="${sequence_dir}/${genome_accession}.fasta.gz"
genome_fa="${sequence_dir}/${genome_accession}.fasta"
# Conventional GCA-style name. The downloader creates it as a symlink to
# ${genome_fa}; it is what dataset_config.sh pointed at before this file existed,
# so anything on the cluster still using that name keeps working.
genome_fa_alias="${sequence_dir}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"

# ── chrom.sizes: IGVF DACC GRCh38 ─────────────────────────────────────────────
# Must match the genome's contigs; the downloader derives it from the .fai when
# the DACC copy is not present.
chrom_sizes="${chrom_sizes_dir}/IGVF.DACC.GRCh38.chrom.sizes.tsv"

# ── Blacklist: ENCODE hg38 ────────────────────────────────────────────────────
blacklist_accession="ENCFF356LFX"
blacklist_url="https://www.encodeproject.org/files/${blacklist_accession}/@@download/${blacklist_accession}.bed.gz"
blacklist_raw="${blacklist_dir}/${blacklist_accession}.bed.gz"
# Symlink to ${blacklist_raw}. Step 02 passes this to `chrombpnet prep nonpeaks -br`.
blacklist="${blacklist_dir}/blacklist.bed.gz"

# ChromBPNet reads a 2114bp window centred on each peak, so the blacklist is
# slopped by half of it: step 01 then drops a peak whenever the *window the model
# would read* touches a blacklist region, not merely when the peak interval does.
# Steps 01 and 02 apply the blacklist at different radii deliberately — peaks get
# the slopped copy, GC-matched negatives get the raw one.
chrombpnet_input_window=2114
blacklist_slop_bp=$(( chrombpnet_input_window / 2 ))
blacklist_slop="${blacklist_dir}/blacklist_slop.bed.gz"

# ── Motif database: canonical kundajelab/MotifCompendium build ────────────────
ref_db_meme="${motif_dir}/MotifCompendium-Database-Human.meme.txt"
ref_db_meme_url="https://raw.githubusercontent.com/kundajelab/MotifCompendium/main/pipeline/data/MotifCompendium-Database-Human.meme.txt"
