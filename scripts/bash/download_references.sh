#!/bin/bash
# download_references.sh
# Fetch + build the shared genome / blacklist / motif references used by this
# pipeline into the lab Data/ folder. Run once per cluster; idempotent — files
# that already exist are verified, not re-downloaded.
#
# Populates the paths referenced by dataset_config.sh / config.sh:
#   $DATA/hg38/Sequence/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna(.gz)(.fai)
#       -> IGVF GRCh38 no-alt analysis set (reference-file IGVFFI0653VCGH)
#   $DATA/hg38/Sequence/chrom_sizes/IGVF.DACC.GRCh38.chrom.sizes.tsv
#   $DATA/hg38/blacklist/{ENCFF356LFX.bed.gz, blacklist.bed.gz -> , blacklist_slop.bed.gz}
#   $DATA/motif/MotifCompendium-Database-Human.meme.txt  (canonical kundajelab build)
#
# Requires curl, samtools and bedtools on PATH, e.g.:
#   ml biology samtools bedtools        # or activate the pipeline's chrombpnet env

set -euo pipefail

DATA="/oak/stanford/groups/engreitz/Data"
SEQ="${DATA}/hg38/Sequence"
CS_DIR="${SEQ}/chrom_sizes"
BL="${DATA}/hg38/blacklist"
MOTIF="${DATA}/motif"

log(){ echo "[$(date '+%F %T')] $*"; }

for t in curl samtools bedtools; do
    command -v "$t" >/dev/null || { echo "ERROR: '$t' not on PATH (try: ml biology samtools bedtools)" >&2; exit 1; }
done

mkdir -p "${SEQ}" "${CS_DIR}" "${BL}" "${MOTIF}"

# --- 1. Genome: IGVF GRCh38 no-alt analysis set (reference-file IGVFFI0653VCGH) ---
FA_GZ="${SEQ}/IGVFFI0653VCGH.fasta.gz"
FA="${SEQ}/IGVFFI0653VCGH.fasta"
FA_URL="https://api.data.igvf.org/reference-files/IGVFFI0653VCGH/@@download/IGVFFI0653VCGH.fasta.gz"
if [[ -s "${FA_GZ}" ]]; then
    log "genome fasta.gz present, skipping download"
else
    log "downloading genome fasta.gz from IGVF"
    curl -fL --retry 3 -o "${FA_GZ}" "${FA_URL}"
fi
# verify against the md5 published in IGVF metadata (best effort)
IGVF_MD5=$(curl -fsL -H 'Accept: application/json' \
    "https://api.data.igvf.org/reference-files/IGVFFI0653VCGH/?format=json" 2>/dev/null \
    | grep -oP '"md5sum"\s*:\s*"\K[0-9a-f]{32}' | head -1 || true)
if [[ -n "${IGVF_MD5}" ]]; then
    if [[ "$(md5sum "${FA_GZ}" | cut -d' ' -f1)" == "${IGVF_MD5}" ]]; then
        log "genome md5 verified against IGVF metadata"
    else
        echo "ERROR: genome fasta.gz md5 does not match IGVF metadata" >&2; exit 1
    fi
fi
[[ -s "${FA}" ]]        || { log "decompressing genome fasta"; zcat "${FA_GZ}" > "${FA}"; }
[[ -s "${FA}.fai" ]]    || { log "indexing genome (samtools faidx)"; samtools faidx "${FA}"; }
# conventional GCA-named symlinks
ln -sf "IGVFFI0653VCGH.fasta"     "${SEQ}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
ln -sf "IGVFFI0653VCGH.fasta.fai" "${SEQ}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.fai"
ln -sf "IGVFFI0653VCGH.fasta.gz"  "${SEQ}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz"

# --- 2. chrom.sizes: IGVF DACC GRCh38 (must match the genome's contigs) ---
CS="${CS_DIR}/IGVF.DACC.GRCh38.chrom.sizes.tsv"
if [[ -s "${CS}" ]]; then
    log "chrom.sizes present"
else
    log "IGVF.DACC chrom.sizes missing — deriving from the genome .fai (identical contigs+lengths)"
    cut -f1,2 "${FA}.fai" > "${CS}"
fi

# --- 3. Blacklist: ENCODE hg38 (ENCFF356LFX) + slop by half the 2114bp window ---
BL_GZ="${BL}/ENCFF356LFX.bed.gz"
BL_URL="https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz"
if [[ -s "${BL_GZ}" ]]; then
    log "blacklist present"
else
    log "downloading ENCODE blacklist (ENCFF356LFX)"
    curl -fL --retry 3 -o "${BL_GZ}" "${BL_URL}"
fi
ln -sf "ENCFF356LFX.bed.gz" "${BL}/blacklist.bed.gz"
SLOP="${BL}/blacklist_slop.bed.gz"
if [[ -s "${SLOP}" ]]; then
    log "blacklist_slop.bed.gz present"
else
    log "generating blacklist_slop.bed.gz (blacklist +/- 1057bp = half the 2114bp input window)"
    bedtools slop -i "${BL_GZ}" -g "${CS}" -b 1057 | gzip > "${SLOP}"
fi

# --- 4. MotifCompendium reference database (canonical, kundajelab/MotifCompendium) ---
MC="${MOTIF}/MotifCompendium-Database-Human.meme.txt"
MC_URL="https://raw.githubusercontent.com/kundajelab/MotifCompendium/main/pipeline/data/MotifCompendium-Database-Human.meme.txt"
if [[ -s "${MC}" ]]; then
    log "MotifCompendium DB present"
else
    log "downloading MotifCompendium reference DB"
    curl -fL --retry 3 -o "${MC}" "${MC_URL}"
fi

log "DONE. Shared references available under ${DATA}"
