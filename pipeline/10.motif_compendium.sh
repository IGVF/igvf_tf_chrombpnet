#!/bin/bash
#SBATCH --job-name=motif_compendium
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 10.motif_compendium.sh
# Purpose: Build a non-redundant motif compendium from per-dataset averaged MoDISco
#          H5 files using MotifCompendium (Kundaje lab).
#
# Input:  Per-dataset fold-averaged MoDISco H5s (step 08)
#           ${averaged_dir}/{dataset}/modisco/modisco_counts_results.h5
# Output (inside ${modisco_compiled_dir}/):
#   modisco_compiled.h5          – clustered motifs for FiNeMo (step 5.4)
#   modisco_compendium.meme      – MEME format export
#   modisco_compendium.mc        – pickled MotifCompendium object
#   modisco_compendium_meta.tsv  – per-motif TF annotations (MotifCompendium-Database-Human) + cluster IDs
#   modisco_config.tsv           – dataset → H5 path mapping (for reference)
#
# Prerequisites: 08.run_modisco.sh must have completed.
#   Requires the 'motif_compendium' conda environment:
#     mamba create -n motif_compendium python=3.10
#     pip install MotifCompendium
#
# Usage:
#   sbatch 10.motif_compendium.sh

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${modisco_compiled_dir}" "${log_dir}"

# Build config TSV: dataset_name <TAB> modisco_h5_path
# Using the fold-averaged MoDISco results (step 09) for all datasets.
config_tsv="${modisco_compiled_dir}/modisco_config.tsv"
echo "# dataset    modisco_h5" > "${config_tsv}"

for dataset in "${datasets[@]}"; do
    modisco_h5="${averaged_dir}/${dataset}/modisco/modisco_counts_results.h5"
    if [[ -f "${modisco_h5}" ]]; then
        echo -e "${dataset}\t${modisco_h5}" >> "${config_tsv}"
    else
        echo "  [WARN] missing averaged MoDISco H5 for ${dataset}: ${modisco_h5}" >&2
        echo "  Run 09.run_modisco.sh first." >&2
    fi
done

n_found=$(grep -c "^[^#]" "${config_tsv}" || true)
echo "[$(date)] Config TSV: ${config_tsv} (${n_found} datasets)"

if [[ "${n_found}" -eq 0 ]]; then
    echo "ERROR: no MoDISco H5s found. Run 09.run_modisco.sh first." >&2
    exit 1
fi

# Run MotifCompendium clustering + annotation
source "${CONDA_INIT}"
conda activate "${motif_compendium_conda}"

echo "[$(date)] Running MotifCompendium (threshold=${motif_compendium_threshold})..."

python "${SCRIPT_DIR}/motif_compendium.py" \
    --config    "${config_tsv}" \
    --out-dir   "${modisco_compiled_dir}" \
    --ref-db    "${ref_db_meme}" \
    --threshold "${motif_compendium_threshold}" \
    --cpus      "${SLURM_CPUS_PER_TASK:-16}"

if [[ ! -f "${modisco_compiled_dir}/modisco_compiled.h5" ]]; then
    echo "ERROR: modisco_compiled.h5 not produced. Check the log above." >&2
    exit 1
fi

echo "[$(date)] 11.motif_compendium complete."
echo "  Compiled H5  : ${modisco_compiled_dir}/modisco_compiled.h5"
echo "  Annotations  : ${modisco_compiled_dir}/modisco_compendium_meta.tsv"
echo ""
echo "  Next step: sbatch 11.run_finemo_unified.sh"
