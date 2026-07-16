#!/bin/bash
# Plot combined full-model QC metrics for all four datasets side by side.
# Run after 04.1.qc_run_full_model.sh has been executed for each dataset.
#SBATCH --job-name=model_qc_combined
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --partition=engreitz,normal
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

CORE_PATH="/oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

source /home/groups/engreitz/Software/anaconda3/etc/profile.d/conda.sh
conda activate /home/groups/engreitz/Users/opushkar/.conda/envs/chrombpnet

python "${SCRIPT_DIR}/qc_full_model.py" \
    --combined \
    --core-path "${CORE_PATH}" \
    --datasets igvf11_h7_hesc igvf3_cardiomyocyte igvf6_definitive_endoderm igvf_endothelial \
    --out-dir "${CORE_PATH}/results/plots/full_model_qc_combined"
