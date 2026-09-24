#!/bin/bash
# shellcheck disable=SC2218  # false positive in shellcheck 0.11.0: activate_env and
# metadata_start both come from lib/bash/common.sh, sourced above.
#SBATCH --job-name=cross_dataset_compendium
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 09.0.cross_dataset_compendium.sh
# Purpose: Build a single non-redundant motif compendium by pooling MoDISco
#          results across all four datasets in this collaboration:
#            igvf3_cardiomyocyte, igvf6_definitive_endoderm,
#            igvf11_h7_hesc, igvf_endothelial (d3 iPSC-EC)
#
# This script does NOT require DATASET_DIR; it operates at the collaboration
# root and hardcodes the four per-dataset MoDISco H5 paths.
#
# Input:  Per-dataset fold-averaged MoDISco H5s (step 08 for each dataset)
# Output (inside ${REPO_ROOT}/results/compendium/modisco_compiled/):
#   modisco_compiled.h5         - clustered motifs for FiNeMo (cross-dataset)
#   modisco_compendium.meme     - MEME format
#   modisco_compendium.mc       - pickled MotifCompendium object
#   modisco_compendium_meta.tsv - per-motif annotations + cluster IDs
#   modisco_config.tsv          - dataset -> H5 path mapping used for this run
#
# Prerequisites: step 08 must have completed for all four datasets.
#
# Usage:
#   cd workflows/SLURM && sbatch 09.0.cross_dataset_compendium.sh

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

# Cross-dataset step: no DATASET_DIR, so source common.sh directly rather than
# config.sh. CONDA_INIT, motif_compendium_conda, ref_db_meme and
# motif_compendium_threshold used to be re-declared here; they now have one home.
# shellcheck source=lib/bash/common.sh
source "${REPO_ROOT}/lib/bash/common.sh" || exit 1

out_dir="${REPO_ROOT}/results/compendium/modisco_compiled"
log_dir="${REPO_ROOT}/results/logs"
mkdir -p "${out_dir}" "${log_dir}"


# Per-dataset MoDISco H5 paths
declare -A h5_map=(
    [igvf3_cardiomyocyte]="${REPO_ROOT}/igvf3_cardiomyocyte/results/contrib_scores/igvf3_cardiomyocyte/modisco/modisco_counts_results.h5"
    [igvf6_definitive_endoderm]="${REPO_ROOT}/igvf6_definitive_endoderm/results/contrib_scores/igvf6_definitive_endoderm/modisco/modisco_counts_results.h5"
    [igvf11_h7_hesc]="${REPO_ROOT}/igvf11_h7_hesc/results/contrib_scores/igvf11_h7_hesc/modisco/modisco_counts_results.h5"
    [igvf_endothelial]="${REPO_ROOT}/igvf_endothelial/results/contrib_scores/modisco/modisco_counts_results.h5"
)

# Build config TSV
config_tsv="${out_dir}/modisco_config.tsv"
echo "# dataset    modisco_h5" > "${config_tsv}"

for dataset in igvf3_cardiomyocyte igvf6_definitive_endoderm igvf11_h7_hesc igvf_endothelial; do
    h5="${h5_map[$dataset]}"
    if [[ -f "${h5}" ]]; then
        echo -e "${dataset}\t${h5}" >> "${config_tsv}"
        echo "  [OK]   ${dataset}: ${h5}"
    else
        echo "  [WARN] ${dataset}: H5 not found, skipping: ${h5}" >&2
    fi
done

n_found=$(grep -c "^[^#]" "${config_tsv}" || true)
echo "[$(date)] Config TSV: ${config_tsv} (${n_found}/4 datasets)"

if [[ "${n_found}" -eq 0 ]]; then
    echo "ERROR: no MoDISco H5s found. Run 08.0.run_modisco.sh for each dataset first." >&2
    exit 1
fi

# Run MotifCompendium clustering + annotation
activate_env "${motif_compendium_env}"

metadata_start "09.0.cross_dataset_compendium"
metadata_inputs+=( "config=${config_tsv}" "ref_db=${ref_db_meme}" )
metadata_outputs+=( "motifs=${out_dir}/modisco_compiled.h5" "meme=${out_dir}/modisco_compendium.meme" "meta_tsv=${out_dir}/modisco_compendium_meta.tsv" )
metadata_params+=( "threshold=${motif_compendium_threshold}" )


echo "[$(date)] Running MotifCompendium across ${n_found} datasets (threshold=${motif_compendium_threshold})..."

python "${src_dir}/motif_compendium.py" \
    --config    "${config_tsv}" \
    --out-dir   "${out_dir}" \
    --ref-db    "${ref_db_meme}" \
    --threshold "${motif_compendium_threshold}" \
    --cpus      "${SLURM_CPUS_PER_TASK:-16}"

if [[ ! -f "${out_dir}/modisco_compiled.h5" ]]; then
    echo "ERROR: modisco_compiled.h5 not produced. Check the log above." >&2
    exit 1
fi

echo "[$(date)] Cross-dataset compendium complete."
echo "  Compiled H5  : ${out_dir}/modisco_compiled.h5"
echo "  Annotations  : ${out_dir}/modisco_compendium_meta.tsv"
echo "  MEME         : ${out_dir}/modisco_compendium.meme"
