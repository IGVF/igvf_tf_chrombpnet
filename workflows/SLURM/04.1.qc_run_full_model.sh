#!/bin/bash
#SBATCH --job-name=model_qc
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=engreitz,normal
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 04.1.qc_run_full_model.sh
# Purpose: Full-model performance QC for one dataset, across its folds: the
#   metrics chrombpnet wrote in 04.0 (counts Pearson/Spearman, profile median
#   JSD, Tn5 max-bias response of the no-bias model) as one table and box
#   plots, plus a predicted-vs-observed log-count scatter per fold. CPU only.
#
# The observed side of the scatter is 00.0's prepared bigwig: training reads
#   it in place (chrombpnet -bw), so the model directory holds no copy under
#   auxiliary/. src/qc_full_model.py sums it over each prediction row's own
#   window from evaluation/chrombpnet_predictions.h5, so predicted and
#   observed are the same regions by construction.
#
# Input:  ${full_model_dir}/<dataset>_<peak_type>_fold_<fold>/evaluation/
#           chrombpnet_metrics.json, chrombpnet_nobias_max_bias_response.txt,
#           chrombpnet_predictions.h5 (04.0; a fold without metrics is skipped)
#         ${data_path}/signal/data_unstranded.bw (00.0)
# Output: ${results_path}/plots/full_model_qc/model_metrics.tsv,
#           performance_boxplot.{pdf,png}, tn5_response.{pdf,png},
#           <dataset>_fold<fold>_scatter.{pdf,png} and _scatter_data.tsv
#
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   sbatch 04.1.qc_run_full_model.sh
#
# Prerequisites: 04.0.train_full_model.sh for the folds to report, and
#   00.0.prepare_signal.sh. 04.2.qc_combined_boxplot.sh reads this step's
#   model_metrics.tsv.

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

metadata_start "04.1.qc_run_full_model"
out_dir="${results_path}/plots/full_model_qc"
# The observed signal. Training reads it in place (-bw), so it is not under
# the model's auxiliary/.
prepared_bw="${data_path}/signal/data_unstranded.bw"
metadata_inputs+=( "model=${full_model_dir}" "signal=${prepared_bw}" )
metadata_outputs+=( "metrics=${out_dir}/model_metrics.tsv" )
metadata_params+=( "peak_type=${peak_type}" )
require_input "${prepared_bw}" 00.0.prepare_signal.sh
preflight_check

activate_env "${chrombpnet_env}"

python "${src_dir}/qc_full_model.py" \
    --full-model-dir "${full_model_dir}" \
    --bigwig "${prepared_bw}" \
    --datasets "${datasets[@]}" \
    --folds "${folds[@]}" \
    --peak-type "${peak_type}" \
    --out-dir "${out_dir}" \
    --save-plots
# No `set -e` in this step: without the guard a failed QC exits 0 and 04.2
# finds no table, or a stale one.
if [[ $? -ne 0 || ! -f "${out_dir}/model_metrics.tsv" ]]; then
    echo "ERROR: qc_full_model.py failed; ${out_dir}/model_metrics.tsv was not written." >&2
    exit 1
fi
echo "[$(date)] Done. Full-model QC in ${out_dir}"
