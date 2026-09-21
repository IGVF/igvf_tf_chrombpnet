#!/bin/bash
# 03.1.select_bias.sh
# Select the best Tn5 bias model per fold from the sweep trained in 03.0.
# Config-driven: reads the dataset name, results path and bias sweep from
# dataset_config.sh (via config.sh). Set DATASET_DIR before running.
#
# Usage:
#   export DATASET_DIR=/path/to/dataset
#   bash 03.1.select_bias.sh
#
# Produces QC plots + tables under ${results_path}/plots/bias_model_selection/.
# Afterward, copy the per-fold winners from selected_bias_per_fold.tsv into
# fold_bias_suffix in dataset_config.sh, then run 04.0.
#
# If fold_bias_suffix is already populated in dataset_config.sh (e.g. for
# re-running plots), the selections are overlaid on the plots via --fold-bias.

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
activate_env "${CONDA_ENV}"

metadata_start "03.1.select_bias"


set -euo pipefail

# Bias labels to evaluate, derived from the sweep in dataset_config
# (bias_suffixes_sweep entries like "_05" -> "05").
biases=()

metadata_params+=( "biases=${biases[*]}" )
for s in "${bias_suffixes_sweep[@]}"; do
    biases+=( "${s#_}" )
done

out_dir="${results_path}/plots/bias_model_selection/${bias_dataset}"

echo "[$(date)] Bias-model selection for ${bias_dataset}"
echo "  bias_models : ${results_path}/bias_models"
echo "  biases      : ${biases[*]}"
echo "  folds       : ${folds[*]}"
echo "  out_dir     : ${out_dir}"

# Build --fold-bias args from fold_bias_suffix if already set in dataset_config.sh.
# Produces e.g.: "0:_08 1:_06 2:_08 3:_08 4:_07"
fold_bias_args=()
if declare -p fold_bias_suffix &>/dev/null 2>&1; then
    for fold in "${!fold_bias_suffix[@]}"; do
        fold_bias_args+=( "${fold}:${fold_bias_suffix[$fold]}" )
    done
fi

extra_args=()
if [[ ${#fold_bias_args[@]} -gt 0 ]]; then
    extra_args+=( --fold-bias "${fold_bias_args[@]}" )
fi

python "${src_dir}/select_bias_model.py" \
    --bias-models-dir "${results_path}/bias_models" \
    --dataset "${bias_dataset}" \
    --peak-type "${peak_type}" \
    --biases "${biases[@]}" \
    --folds "${folds[@]}" \
    --out-dir "${out_dir}" \
    "${extra_args[@]}"

echo "[$(date)] Done. Review ${out_dir}/ then set fold_bias_suffix in dataset_config.sh."
