#!/bin/bash
#SBATCH --job-name=preprocess_peaks
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 01.preprocess_peaks.sh
# Purpose: Remove blacklisted regions from peak files and reformat to
#          narrowPeak for chrombpnet (summit = midpoint of peak)

ml biology
ml bedtools/2.30.0

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"
set -euo pipefail

mkdir -p "${data_path}"

for dataset in "${datasets[@]}"; do
    echo "Processing peaks for ${dataset}..."
    # Remove peaks overlapping (slopped) blacklist.
    # Uses ${blacklist_slop} if the dataset_config defines it; otherwise falls back
    # to the legacy location next to the genome for backward compatibility.
    bedtools intersect \
        -v \
        -a "${DATASET_DIR}/data/peaks/${dataset}_${peak_type}_peaks.bed" \
        -b "${blacklist_slop:-${genome_path}/blacklist_slop.bed}" \
        > "${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.bed"

    # Convert to narrowPeak format; summit = midpoint of peak
    awk 'BEGIN{OFS="\t"} {
        summit = int(($3-$2)/2);
        print $1, $2, $3, "peak_"NR, "0", ".", "0", "-1", "-1", summit
    }' "${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.bed" \
        > "${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"

    n=$(wc -l < "${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak")
    echo "  -> ${n} peaks written to ${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
done

echo "Done: 01.preprocess_peaks.sh"
