#!/bin/bash
#SBATCH --job-name=finemo_unified
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,owners
#SBATCH --array=0
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 11.run_finemo_unified.sh
# Purpose: Call motif hits using the unified (compendium) MoDISco H5.
#          Uses modisco_compiled.h5 built in step 11 and fold-averaged
#          contribution scores (step 07) so that a single hit set per
#          dataset is produced, enabling direct cross-dataset comparisons.
#
#          One SLURM array job per dataset.
#
# Input per dataset:
#   {averaged_dir}/{dataset}/{dataset}_average_shaps.counts.h5  – averaged DeepLIFT scores (step 07)
#   modisco_compiled.h5                                 – unified MoDISco patterns (step 11)
#
# Output (inside finemo_unified_dir/{dataset}_{peak_type}/):
#   hits.bed.gz + hits.bed.gz.tbi     – tabix-indexed hit calls
#   hits.tsv                          – full hit table
#   finemo_report/                    – HTML report
#
# Usage:
#   sbatch 11.run_finemo_unified.sh            # dataset 0
#   sbatch 11.run_finemo_unified.sh            # (override with --array=0 if needed)
#
# Prerequisites: steps 06 and 10 must have completed.
#   Requires the 'finemo' conda environment.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

dataset="${datasets[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${dataset}" ]] && { echo "No dataset at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

compiled_h5="${modisco_compiled_dir}/modisco_compiled.h5"
if [[ ! -f "${compiled_h5}" ]]; then
    echo "ERROR: ${compiled_h5} not found. Run 11.motif_compendium.sh first." >&2
    exit 1
fi

ml devel
ml system
ml cuda/11.5.0
ml cudnn/8.6.0.163
ml biology samtools

source "${CONDA_INIT}"
conda activate "${finemo_conda}"

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

counts_h5="${averaged_dir}/${dataset}/${dataset}_average_shaps.counts.h5"
peaks_file="${data_path}/${dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"

if [[ ! -f "${counts_h5}" ]]; then
    echo "ERROR: ${counts_h5} not found. Run 06.average_contrib_scores.sh first." >&2
    exit 1
fi

out_dir="${finemo_unified_dir}/${dataset}_${peak_type}"
hits_file="${out_dir}/hits.bed.gz"

if [[ -f "${hits_file}" ]]; then
    echo "[${dataset}] Hit calls already exist, skipping."
    exit 0
fi

finemo_npz="${out_dir}/intermediate_inputs.npz"
mkdir -p "${out_dir}"

echo "[$(date)] [${dataset}] Extracting regions from averaged scores..."

finemo extract-regions-chrombpnet-h5 \
    --h5s          "${counts_h5}" \
    --peaks        "${peaks_file}" \
    --out-path     "${finemo_npz}" \
    --region-width 1000

echo "[$(date)] [${dataset}] Calling hits (unified modisco)..."

finemo call-hits \
    -r "${finemo_npz}" \
    -m "${compiled_h5}" \
    -l "${finemo_alpha}" \
    -o "${out_dir}" \
    -b 200

if [[ -f "${out_dir}/hits.bed" ]]; then
    echo "[$(date)] [${dataset}] Compressing and indexing hits..."
    bgzip -c "${out_dir}/hits.bed" > "${hits_file}"
    tabix -p bed "${hits_file}"
fi

echo "[$(date)] [${dataset}] Fi-NeMo (unified) complete."
echo "  Hits: ${hits_file}"
