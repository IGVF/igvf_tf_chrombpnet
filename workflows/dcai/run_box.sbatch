#!/bin/bash
#SBATCH --job-name=chrombpnet_prep_box
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=216
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# run_box.sbatch
# Purpose: one whole DCAI node (the cluster's unit of allocation: 216 cores,
#   ~2 TB, 8x H100) running workflows/dcai/run_box.sh. --exclusive because the
#   partition oversubscribes by default; --mem=0 is all of the node's memory.
#   The time limit is explicit because defq's default is 8 h; the run resumes
#   if it is cut short.
#
# Usage (from the checkout, so the job inherits REPO_ROOT and the paths):
#   source workflows/dcai/env.sh
#   sbatch --chdir "${DATASET_ROOT}/chrombpnet" workflows/dcai/run_box.sbatch
#   BOX_FIRST_N=10 BOX_STAGES=prep,bias BOX_BACKFILL=prep sbatch ...   # see run_box.sh

# sbatch copies this file to a spool directory, so REPO_ROOT must come from
# the submitting shell (env.sh exports it).
: "${REPO_ROOT:?submit after: source workflows/dcai/env.sh}"
exec bash "${REPO_ROOT}/workflows/dcai/run_box.sh"
