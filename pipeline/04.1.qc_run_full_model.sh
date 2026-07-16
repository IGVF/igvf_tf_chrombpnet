#!/bin/bash
# Run after step 04.0 (train_full_model.sh) to evaluate full model performance.
#SBATCH --job-name=model_qc
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=engreitz,normal
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/qc_full_model.py" \
    --full-model-dir "${full_model_dir}" \
    --data-path "${data_path}" \
    --datasets "${datasets[@]}" \
    --folds "${folds[@]}" \
    --peak-type "${peak_type}" \
    --out-dir "${results_path}/plots/full_model_qc" \
    --save-plots
