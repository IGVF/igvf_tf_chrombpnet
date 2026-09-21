#!/bin/bash
# status.sh — what has run for this dataset, and what to run next.
#
#   export DATASET=igvf3_cardiomyocyte
#   bash workflows/SLURM/status.sh
#
# Reads the same config the steps do and checks for each step's marker output.
# It inspects the filesystem, not the SLURM queue: a step shown as PENDING may
# be running right now. Run metadata (results/metadata/) is the record of what
# actually executed -- see queries.sql.

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

fold0="${folds[0]}"
d="${datasets[0]}"

printf '\n  dataset : %s\n' "${DATASET}"
printf '  config  : %s\n'   "${dataset_config}"
printf '  data    : %s\n'   "${dataset_dir}"
printf '  refs    : %s\n\n' "${REFERENCE_ROOT}"

pending=()

# report <step> <marker> [marker...] — DONE if every marker exists.
report() {
    local step="$1"; shift
    local missing=0 m
    for m in "$@"; do [[ -e "${m}" ]] || missing=$(( missing + 1 )); done
    if (( missing == 0 )); then
        printf '  [ DONE ] %s\n' "${step}"
    else
        printf '  [  --  ] %s  (%d/%d output(s) missing)\n' "${step}" "${missing}" "$#"
        pending+=( "${step}" )
    fi
}

report 00.0.prepare_signal         "${data_path}/signal/data_unstranded.bw"
report 01.0.preprocess_peaks         "${data_path}/${d}_${peak_type}_peaks_no_blacklist.narrowPeak"
report 02.0.preprocess_nonpeaks      "${data_path}/${d}/output_${peak_type}_fold_${fold0}_negatives.bed"
report 03.0.train_bias_model       "${results_path}/bias_models"
report 03.1.select_bias            "${results_path}/plots/bias_model_selection/${bias_dataset}/selected_bias_per_fold.tsv"
report 03.2.qc_selected_bias       "${results_path}/bias_models"
report 04.0.train_full_model       "${full_model_dir}/${d}_${peak_type}_fold_${fold0}/models/chrombpnet_nobias.h5"
report 04.1.qc_run_full_model      "${results_path}/plots/full_model_qc/model_metrics.tsv"
report 04.3.generate_predictions   "${predictions_dir}/${d}_${peak_type}"
report 05.0.get_contrib_scores       "${full_model_dir}/${d}_${peak_type}_fold_${fold0}/interpretation/interpretation.counts_scores.h5"
report 06.0.average_contrib_scores   "${averaged_dir}/${d}/${d}_average_shaps.counts.h5"
report 07.0.contribs_to_bigwig       "${averaged_dir}/${d}/${d}_average_shaps.counts.bw"
report 08.0.run_modisco              "${averaged_dir}/${d}/modisco/modisco_counts_results.h5"
report 09.0.cross_dataset_compendium "${REPO_ROOT}/results/compendium/modisco_compiled/modisco_compiled.h5"
report 10.0.run_finemo_unified       "${finemo_unified_dir}/${d}_${peak_type}/hits.bed.gz"
report 11.0.postprocess_finemo       "${finemo_unified_dir}/${d}_${peak_type}/finemo_report/motif_report.tsv"

# 04.2 (combined QC) and 09 span datasets, so they are not per-dataset state and
# are reported above only where a single shared artifact exists.

echo ""
if (( ${#pending[@]} == 0 )); then
    echo "  All steps complete for ${DATASET}."
else
    printf '  Next:  cd %s/workflows/SLURM && DATASET=%s sbatch %s.sh\n' \
        "${REPO_ROOT}" "${DATASET}" "${pending[0]}"
    (( ${#pending[@]} > 1 )) && printf '  Then:  %s\n' "${pending[*]:1}"
fi
echo ""
