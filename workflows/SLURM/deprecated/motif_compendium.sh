#!/bin/bash
# SUPERSEDED — not part of the numbered sequence, and not run by the pipeline.
#
# This builds a PER-DATASET motif compendium. 09.0.cross_dataset_compendium.sh
# replaced it by pooling all datasets, and 10.0.run_finemo_unified.sh reads that
# cross-dataset output. Kept for reference only; the step numbers in the header
# below are the ones it had when it was live and are not updated.
#
# It previously lived at workflows/SLURM/_10.motif_compendium.sh, where the "_"
# was meant to mark it as out-of-sequence but only made it sort above step 00.
#SBATCH --job-name=motif_compendium
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 10.motif_compendium.sh
# Run once after step 08 has completed for all datasets. No DATASET_DIR needed;
# all dataset MoDISco H5s are discovered from the shared results path in config.sh.
#
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
# Prerequisites: 08.0.run_modisco.sh must have completed.
#   Requires the 'motif_compendium' conda environment:
#     mamba create -n motif_compendium python=3.10
#     pip install MotifCompendium
#
# Usage:
#   sbatch 10.motif_compendium.sh

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
activate_env "${motif_compendium_env}"

metadata_start "deprecated/motif_compendium"
metadata_inputs+=( "config_tsv=${config_tsv}" "ref_db=${ref_db_meme}" )
metadata_outputs+=( "compiled_h5=${modisco_compiled_dir}/modisco_compiled.h5" )
metadata_params+=( "threshold=${motif_compendium_threshold}" )


echo "[$(date)] Running MotifCompendium (threshold=${motif_compendium_threshold})..."

python "${src_dir}/motif_compendium.py" \
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
