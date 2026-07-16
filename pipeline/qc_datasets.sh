#!/bin/bash
#SBATCH --job-name=qc_datasets
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# qc_datasets.sh
# Submit with: sbatch pipeline/qc_datasets.sh
# (No DATASET_DIR needed; datasets are hardcoded in qc_datasets.py)

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CORE_PATH="$(dirname "${SCRIPT_DIR}")"

# config.sh requires DATASET_DIR; any valid dataset works here since we only
# need CONDA_INIT and CONDA_ENV from it.
export DATASET_DIR="${CORE_PATH}/igvf6_definitive_endoderm"
source "${SCRIPT_DIR}/config.sh"

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/qc_datasets.py"
