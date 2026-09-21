#!/bin/bash
# Plot combined full-model QC metrics for all four datasets side by side.
# Run once after 04.1.qc_run_full_model.sh has completed for all datasets.
# No DATASET_DIR needed; datasets are hardcoded below.
#
# CORE_PATH is the collaboration root that holds <dataset>/results/... It is
# NOT derived from REPO_ROOT: this step reads results written by 04.1 across
# all four datasets, which on the cluster live outside this checkout. Override
# it at submit time if your results tree is somewhere else.
#SBATCH --job-name=model_qc_combined
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --partition=engreitz,normal
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

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
# config.sh.
# shellcheck source=lib/bash/common.sh
source "${REPO_ROOT}/lib/bash/common.sh" || exit 1

CORE_PATH="${CORE_PATH:-/oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab}"

activate_env "${CONDA_ENV}"

metadata_start "04.2.qc_combined_boxplot"


python "${src_dir}/qc_full_model.py" \
    --combined \
    --core-path "${CORE_PATH}" \
    --datasets igvf11_h7_hesc igvf3_cardiomyocyte igvf6_definitive_endoderm igvf_endothelial \
    --out-dir "${CORE_PATH}/results/plots/full_model_qc_combined"
