#!/bin/bash
# shellcheck disable=SC2218  # false positive in shellcheck 0.11.0: log() is defined below, before its first use
# download_references.sh
# Fetch + build the shared genome / chrom.sizes / blacklist / motif references
# used by this pipeline. Run once per cluster; idempotent — files that already
# exist are verified, not re-downloaded.
#
# Every path, filename, accession and URL below comes from lib/bash/references.sh,
# which the pipeline also reads. This script does not decide where anything goes;
# it only fetches what that file declares. To install somewhere else:
#
#   export REFERENCE_ROOT=/scratch/$USER/Data
#   bash scripts/bash/download_references.sh
#
# Requires curl and samtools on PATH:
#   ml biology samtools
#
# bedtools is no longer needed anywhere: the blacklist slop moved into
# src/preprocess_peaks.py (step 01), whose equivalence to `bedtools slop` and
# `bedtools intersect -v` is pinned by tests/test_intervals.py.

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

# References only — this script needs no dataset and no conda env.
# shellcheck source=../../lib/bash/references.sh
source "${REPO_ROOT}/lib/bash/references.sh" || exit 1

set -euo pipefail

log(){ echo "[$(date '+%F %T')] $*"; }

for t in curl samtools; do
    command -v "$t" >/dev/null || { echo "ERROR: '$t' not on PATH (try: ml biology samtools)" >&2; exit 1; }
done

log "installing references under ${REFERENCE_ROOT}"
mkdir -p "${sequence_dir}" "${chrom_sizes_dir}" "${blacklist_dir}" "${motif_dir}"

# --- 1. Genome: IGVF GRCh38 no-alt analysis set ---
if [[ -s "${genome_fa_gz}" ]]; then
    log "genome fasta.gz present, skipping download"
else
    log "downloading genome fasta.gz from IGVF (${genome_accession})"
    curl -fL --retry 3 -o "${genome_fa_gz}" "${genome_url}"
fi
# verify against the md5 published in IGVF metadata (best effort)
expected_md5=$(curl -fsL -H 'Accept: application/json' "${genome_metadata_url}" 2>/dev/null \
    | grep -oP '"md5sum"\s*:\s*"\K[0-9a-f]{32}' | head -1 || true)
if [[ -n "${expected_md5}" ]]; then
    if [[ "$(md5sum "${genome_fa_gz}" | cut -d' ' -f1)" == "${expected_md5}" ]]; then
        log "genome md5 verified against IGVF metadata"
    else
        echo "ERROR: genome fasta.gz md5 does not match IGVF metadata" >&2; exit 1
    fi
fi
[[ -s "${genome_fa}" ]]     || { log "decompressing genome fasta"; zcat "${genome_fa_gz}" > "${genome_fa}"; }
[[ -s "${genome_fa}.fai" ]] || { log "indexing genome (samtools faidx)"; samtools faidx "${genome_fa}"; }
# Conventional GCA-named symlinks, relative so the tree stays relocatable.
ln -sf "$(basename "${genome_fa}")"        "${genome_fa_alias}"
ln -sf "$(basename "${genome_fa}").fai"    "${genome_fa_alias}.fai"
ln -sf "$(basename "${genome_fa_gz}")"     "${genome_fa_alias}.gz"

# --- 2. chrom.sizes: IGVF DACC GRCh38 (must match the genome's contigs) ---
if [[ -s "${chrom_sizes}" ]]; then
    log "chrom.sizes present"
else
    log "IGVF.DACC chrom.sizes missing — deriving from the genome .fai (identical contigs+lengths)"
    cut -f1,2 "${genome_fa}.fai" > "${chrom_sizes}"
fi

# --- 3. Blacklist: ENCODE hg38 + slop by half the model input window ---
if [[ -s "${blacklist_raw}" ]]; then
    log "blacklist present"
else
    log "downloading ENCODE blacklist (${blacklist_accession})"
    curl -fL --retry 3 -o "${blacklist_raw}" "${blacklist_url}"
fi
ln -sf "$(basename "${blacklist_raw}")" "${blacklist}"
# No slopped copy is built: src/preprocess_peaks.py (step 01) slops in-process
# from chrombpnet_input_window, so the radius has a single definition.

# --- 4. MotifCompendium reference database (canonical, kundajelab/MotifCompendium) ---
if [[ -s "${ref_db_meme}" ]]; then
    log "MotifCompendium DB present"
else
    log "downloading MotifCompendium reference DB"
    curl -fL --retry 3 -o "${ref_db_meme}" "${ref_db_meme_url}"
fi

log "DONE. Shared references available under ${REFERENCE_ROOT}"
