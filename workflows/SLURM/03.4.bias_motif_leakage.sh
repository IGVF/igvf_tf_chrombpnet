#!/bin/bash
# 03.4.bias_motif_leakage.sh
# Purpose: has each fold's selected bias model learned TF motifs -- and, once
#   04.5 has run, did the full models trained on leaky bias models learn fewer?
#
# A bias model should learn the enzyme's cut preference and base composition,
#   not TF motifs, and 03.1 cannot tell the two apart: its score (the profile
#   head's normalised JSD) rewards a bias model that has absorbed TF signal as
#   readily as one that has not, and it leans to the top of the factor sweep,
#   where that happens most. 03.3 runs TF-MoDISco on every fold's selected bias
#   model; src/bias_motif_leakage.py reads those results and reports, per fold
#   and head, the share of positive seqlets in patterns whose consensus carries
#   a real TF site (AP-1, NFI, CEBP, CTCF, TEAD), on both strands. A clean bias
#   model is at 0. It reads the sequence, not 03.3's report labels: tomtom-lite
#   names GC-rich stretches, Alu fragments and poly-A runs after TFs, so a label
#   screen makes most counts heads look contaminated.
#
# Whether it matters: with 04.5's results present, the same is read for the
#   full models, and the summary compares their TF-site shares on folds whose
#   bias model is leaky (counts head at 10% or more) with folds whose bias model
#   is clean. Similar ranges mean the leakage did not reach the full models; a
#   site clearly lower on leaky folds means it did, and those folds' bias models
#   want retraining at a lower bias factor (03.0), then 03.2-03.4 and their
#   full models. Advisory, like 02.0: it never fails on what it finds.
#
# Input:  per fold, from fold_bias_suffix in config.yaml:
#         ${results_path}/bias_models/bias_model<suffix>/<prefix>/evaluation/
#             motif_qc_{counts,profile}/<prefix>_bias_modisco_results.h5   (03.3)
#         ${full_model_dir}/<dataset>_<peak_type>_fold_<fold>/evaluation/
#             motif_qc_{counts,profile}/chrombpnet_nobias_modisco_results.h5   (04.5, optional)
# Output: ${results_path}/plots/bias_motif_leakage/
#             bias_motif_leakage.tsv          fold x head x model: seqlets, TF-site share, sites
#             bias_motif_leakage_summary.txt  leaky folds, and the full-model comparison
#
# Usage:
#   export DATASET=<name>        # or DATASET_CONFIG=/path/to/config.yaml
#   bash 03.4.bias_motif_leakage.sh   # after 03.3; again after 04.5 for the comparison
#
# Prerequisites: 03.3.modisco_selected_bias.sh for every fold, with
#   fold_bias_suffix set in the dataset config.yaml.

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

out_dir="${results_path}/plots/bias_motif_leakage"

metadata_start "03.4.bias_motif_leakage"
fold_bias_args=()
for fold in "${folds[@]}"; do
    suffix="${fold_bias_suffix[${fold}]:-}"
    if [[ -z "${suffix}" ]]; then
        echo "ERROR: no bias suffix for fold ${fold} in fold_bias_suffix (config.yaml); run 03.1 and set it." >&2
        exit 1
    fi
    fold_bias_args+=( "${fold}:${suffix}" )
    prefix="${bias_dataset}_${peak_type}_fold_${fold}"
    for head in counts profile; do
        motifs="${results_path}/bias_models/bias_model${suffix}/${prefix}/evaluation/motif_qc_${head}/${prefix}_bias_modisco_results.h5"
        metadata_inputs+=( "motifs=${motifs}" )
        require_input "${motifs}" 03.3.modisco_selected_bias.sh
    done
done
metadata_params+=( "fold_bias=${fold_bias_args[*]}" )
metadata_outputs+=( "motif_leakage=${out_dir}/bias_motif_leakage.tsv" "motif_leakage_summary=${out_dir}/bias_motif_leakage_summary.txt" )
preflight_check

activate_env "${chrombpnet_env}"

echo "[$(date)] TF motifs in the selected bias models (fold:suffix ${fold_bias_args[*]})"
python "${src_dir}/bias_motif_leakage.py" \
    --bias-models-dir "${results_path}/bias_models" \
    --full-model-dir "${full_model_dir}" \
    --dataset "${datasets[0]}" \
    --bias-dataset "${bias_dataset}" \
    --peak-type "${peak_type}" \
    --folds "${folds[@]}" \
    --fold-bias "${fold_bias_args[@]}" \
    --out-dir "${out_dir}"
_rc=$?
# No `set -e` in this step: guard the call and its output explicitly.
if [[ ${_rc} -ne 0 || ! -f "${out_dir}/bias_motif_leakage.tsv" ]]; then
    echo "ERROR: bias_motif_leakage.py failed (exit ${_rc}); ${out_dir}/bias_motif_leakage.tsv was not written." >&2
    exit 1
fi
echo "[$(date)] Done: ${out_dir}/bias_motif_leakage_summary.txt"
