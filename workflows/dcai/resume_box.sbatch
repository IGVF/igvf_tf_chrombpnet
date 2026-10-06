#!/bin/bash
#SBATCH --job-name=chrombpnet_resume_box
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=216
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# resume_box.sbatch
# Purpose: one whole DCAI node running workflows/dcai/resume_box.sh -- the bias
#   QC and the full chains another box did not finish. Submit it with
#   --dependency=afterany:<that box> so the two never run at once. The job ends
#   when every chain does; the time limit is a generous cap (defq has none,
#   billing is the node-time used), and a later resume picks up from there.
#
# Usage (from the checkout, so the job inherits REPO_ROOT and the paths):
#   source workflows/dcai/env.sh
#   RESUME_CONFIGS=<list> sbatch --dependency=afterany:<box> \
#       --chdir "${DATASET_ROOT}/chrombpnet" workflows/dcai/resume_box.sbatch

# sbatch copies this file to a spool directory, so REPO_ROOT must come from
# the submitting shell (env.sh exports it).
: "${REPO_ROOT:?submit after: source workflows/dcai/env.sh}"
exec bash "${REPO_ROOT}/workflows/dcai/resume_box.sh"
