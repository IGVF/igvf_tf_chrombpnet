#!/bin/bash
# run_all.sh
# Purpose: Steps 00.0 -> 08.0 for one dataset, in order, on a molab box. This
#   is the literal command sequence, written out rather than generated, so you
#   can read it, copy any single line to a terminal, or comment lines out and
#   re-run.
#
# The step scripts themselves live in workflows/SLURM/ and are deliberately NOT
# copied here: that directory is the one definition of what each step does, and
# molab only changes how it is launched. `run_step.sh --list` shows them all
# with the environment each one runs in.
#
# Resumable: run_step.sh skips every array index whose run-metadata record
# says it finished with its outputs intact, so re-running after a failure
# costs only the steps that have not finished.
#
# STOPS at 03.1 by design. Bias-model selection is a manual hand-off: 03.1
# writes selected_bias_per_fold.tsv and the plots are meant to be reviewed
# before the winners go into fold_bias_suffix. Re-run with --after-bias once
# that is done.
#
# Ends at 08.0. Left out on purpose, because they compare or pool datasets
# rather than process this one: 04.2 (full-model QC side by side across every
# dataset config it finds; it writes under ${DATASET_ROOT}/results, outside
# this dataset's output_dir, so neither the skip check nor the bucket sync sees
# it), and 09.0-11.0 (the cross-dataset motif compendium, then Fi-NeMo against
# it). Run those by hand with run_step.sh once every dataset has reached 08.0.
#
# Input:  a validated config (cli.py config validate --dataset "$DATASET")
# Output: everything under output_dir from the config
# Usage:
#   source workflows/molab/env.sh
#   bash workflows/molab/run_all.sh                 # 00.0 .. 03.1, then stop
#   bash workflows/molab/run_all.sh --after-bias    # 03.2 .. 08.0
#   bash workflows/molab/run_all.sh --dry-run
# Prerequisites: setup_molab.sh has run; env.sh is sourced.

# shellcheck disable=SC2218  # 0.11.0 false positive (see CLAUDE.md): molab_config comes from env.sh, sourced before use

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workflows/molab/env.sh
source "${SCRIPT_DIR}/env.sh"

RUN="bash ${SCRIPT_DIR}/run_step.sh"
PHASE="pre-bias"
DRY=""
for a in "$@"; do
    case "$a" in
        --after-bias) PHASE="post-bias" ;;
        --dry-run)    DRY="--dry-run" ;;
        -h|--help)    sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown option $a" >&2; exit 1 ;;
    esac
done

# Array ranges, derived by lib/bash/config.sh exactly as the steps derive them.
# Two different lists: 03.0 indexes fold_idx * n_factors + factor_idx over
# bias_sweep_folds, while the per-fold steps (03.2 .. 05.0) index folds -- and
# a config may sweep the bias on a subset of its folds.
# shellcheck disable=SC2016  # expanded by molab_config's child bash
mapfile -t _cfg < <(molab_config '${#bias_sweep_folds[@]}' '${#folds[@]}')
n_sweep_folds="${_cfg[0]:-0}"
n_folds="${_cfg[1]:-0}"
unset _cfg
if [[ "${n_folds}" -lt 1 || "${n_sweep_folds}" -lt 1 ]]; then
    echo "ERROR: could not resolve folds from ${DATASET_CONFIG:-config/${DATASET}/config.yaml}" >&2
    exit 1
fi
fold_max=$(( n_folds - 1 ))

# sweep_range — set sweep_max for 03.0 from the factors load_bias_sweep
# returns, the function 03.0 and 03.1 call: 02.0's scan when it exists,
# otherwise the config's bias_factors. Called right before 03.0, not up here,
# because 02.0 writes the scan: a range counted from the config before that
# would not match the jobs 03.0 then derives from the scan.
sweep_range() {
    local cfg=() n_factors scan from="the config's bias_factors"
    # shellcheck disable=SC2016  # expanded by molab_config's child bash
    mapfile -t cfg < <(molab_config \
        '$(load_bias_sweep >/dev/null 2>&1; echo "${#bias_factors[@]}")' '${bias_scan_file}')
    n_factors="${cfg[0]:-0}"
    scan="${cfg[1]:-}"
    [[ -s "${scan}" ]] && from="02.0's scan"
    if [[ "${n_factors}" -lt 1 ]]; then
        echo "ERROR: no bias factors to sweep: no scan at ${scan} and none in the config." >&2
        echo "  02.0.qc_training_data.sh writes the scan." >&2
        return 1
    fi
    sweep_max=$(( n_sweep_folds * n_factors - 1 ))
    echo "=== 03.0: ${n_sweep_folds} fold(s) x ${n_factors} bias factor(s), from ${from} -> --array 0-${sweep_max}"
}

echo "=== dataset=${DATASET}  folds=${n_folds}  bias-sweep folds=${n_sweep_folds}"
echo

if [[ "${PHASE}" == "pre-bias" ]]; then
    # --- signal, peaks, background ------------------------------------------
    ${RUN} ${DRY} 00.0.prepare_signal.sh        || exit 1   # preprocess: Tn5 shift + cut-site bigwig
    ${RUN} ${DRY} 00.1.preprocess_peaks.sh      || exit 1   # preprocess: blacklist filter -> narrowPeak
    ${RUN} ${DRY} 01.0.preprocess_nonpeaks.sh   || exit 1   # chrombpnet: GC-matched negatives

    # QC comes AFTER the negatives: the comparative half scores peaks against
    # them. Advisory, never fails the pipeline.
    ${RUN} ${DRY} 02.0.qc_training_data.sh       || exit 1   # preprocess: TSS enrichment, AUROC

    # --- bias sweep ---------------------------------------------------------
    sweep_range || exit 1
    ${RUN} ${DRY} --array "0-${sweep_max}" 03.0.train_bias_model.sh || exit 1   # chrombpnet, GPU
    ${RUN} ${DRY} 03.1.select_bias.sh           || exit 1   # chrombpnet: pick a winner per fold

    # shellcheck disable=SC2016  # expanded by molab_config's child bash
    selection_dir="$(molab_config '${results_path}/plots/bias_model_selection/${bias_dataset}')"
    cat <<MSG

=== Stopping here, on purpose.

  Review   ${selection_dir}/
  then copy the winners from selected_bias_per_fold.tsv into fold_bias_suffix
  in the dataset config (${DATASET_CONFIG:-config/${DATASET}/config.yaml}).

  03.1 also flags a winner sitting at an END of the swept range (sweep_edge):
  select_best answers "best of what we tried" and cannot see past the grid it
  was given. If it fires, widen bias_factors before continuing.

  Then:  bash workflows/molab/run_all.sh --after-bias
MSG
    exit 0
fi

# --- selected bias model: QC --------------------------------------------------
${RUN} ${DRY} --array "0-${fold_max}" 03.2.qc_selected_bias.sh      || exit 1   # chrombpnet, GPU: predict + DeepSHAP
${RUN} ${DRY} --array "0-${fold_max}" 03.3.modisco_selected_bias.sh || exit 1   # chrombpnet, CPU: TF-MoDISco

# --- full model ---------------------------------------------------------------
${RUN} ${DRY} --array "0-${fold_max}" 04.0.train_full_model.sh      || exit 1   # chrombpnet, GPU
${RUN} ${DRY} 04.1.qc_run_full_model.sh                             || exit 1   # chrombpnet: metrics across folds
# Array index = dataset, and a config names one: index 0, run_step's default.
${RUN} ${DRY} 04.3.generate_predictions.sh                          || exit 1   # chrombpnet, GPU
${RUN} ${DRY} --array "0-${fold_max}" 04.4.qc_full_model_interpret.sh || exit 1 # chrombpnet, GPU: DeepSHAP, per fold
${RUN} ${DRY} --array "0-${fold_max}" 04.5.modisco_full_model.sh    || exit 1   # chrombpnet, CPU: TF-MoDISco, per fold

# --- contribution scores and motifs -------------------------------------------
${RUN} ${DRY} --array "0-${fold_max}" 05.0.get_contrib_scores.sh    || exit 1   # chrombpnet, GPU: all peaks, per fold
${RUN} ${DRY} 06.0.average_contrib_scores.sh                        || exit 1   # chrombpnet: average over folds
${RUN} ${DRY} 07.0.contribs_to_bigwig.sh                            || exit 1   # chrombpnet: averaged scores -> bigwig
${RUN} ${DRY} 08.0.run_modisco.sh                                   || exit 1   # chrombpnet, CPU: TF-MoDISco on the average

echo
echo "=== 00.0 -> 08.0 complete for ${DATASET}."
