#!/bin/bash
# config.sh - Pipeline configuration for ChromBPNet
# Source this at the top of every pipeline script: source config.sh
#
# This is a shared pipeline used by multiple datasets. Set DATASET_DIR to the
# root of the dataset you want to run before submitting any job:
#
#   export DATASET_DIR=/path/to/igvf_tf_collab/igvf3_cardiomyocyte
#   sbatch pipeline/05.train_full_model.sh
#
# DATASET_DIR is propagated automatically to SLURM jobs (--export=ALL default).

# Pipeline script directory (auto-detected)
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Validate DATASET_DIR
if [[ -z "${DATASET_DIR}" ]]; then
    echo "ERROR: DATASET_DIR is not set. Export it before sourcing config.sh:" >&2
    echo "  export DATASET_DIR=/path/to/dataset" >&2
    exit 1
fi

# Dataset-specific parameters (datasets, genome, fragments path, bias config)
# results_path and data_path are available to dataset_config.sh
source "${DATASET_DIR}/dataset_config.sh"

# Dataset paths (all data and results are rooted at DATASET_DIR)
results_path="${DATASET_DIR}/results"
data_path="${results_path}/preprocessing"

# Output directories
full_model_dir="${results_path}/full_models"
full_model_dir_selected="${full_model_dir}" # alias kept for script compatibility
predictions_dir="${results_path}/predictions"
averaged_dir="${results_path}/contrib_scores"
compendium_dir="${results_path}/compendium"
log_dir="${results_path}/logs"
modisco_compiled_dir="${compendium_dir}/modisco_compiled"
finemo_unified_dir="${results_path}/finemo_unified"

# MotifCompendium reference database in MEME format (used by step 11)
# Shared lab copy (canonical kundajelab/MotifCompendium build); (re)fetch via scripts/bash/download_references.sh
ref_db_meme="/oak/stanford/groups/engreitz/Data/motif/MotifCompendium-Database-Human.meme.txt"

# Algorithm parameters
finemo_alpha="0.8" # Fi-NeMo hit-calling threshold (lower = more hits)
motif_compendium_threshold="0.95" # Leiden clustering similarity cutoff (step 11)

# Conda environments (recreate from the pinned specs: conda env create -f envs/<name>.yml)
CONDA_INIT="/home/groups/engreitz/Software/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="/home/groups/engreitz/Users/opushkar/.conda/envs/chrombpnet"
finemo_conda="/home/groups/engreitz/Users/opushkar/.conda/envs/finemo"
motif_compendium_conda="/home/groups/engreitz/Users/opushkar/.conda/envs/motif_compendium"
