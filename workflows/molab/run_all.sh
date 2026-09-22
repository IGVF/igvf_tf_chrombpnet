#!/bin/bash
# run_all.sh
# Purpose: Steps 00.0 -> 04.1 in order, on a molab box. This is the literal
#   command sequence, written out rather than generated, so you can read it,
#   copy any single line to a terminal, or comment lines out and re-run.
#
# The step scripts themselves live in workflows/SLURM/ and are deliberately NOT
# copied here: that directory is the one definition of what each step does, and
# molab only changes how it is launched. `run_step.sh --list` shows them all
# with the environment each one runs in.
#
# Resumable: every step skips work whose output already exists, so re-running
# after a failure costs only the steps that have not finished.
#
# STOPS at 03.1 by design. Bias-model selection is a manual hand-off: 03.1
# writes selected_bias_per_fold.tsv and the plots are meant to be reviewed
# before the winners go into fold_bias_suffix. Re-run with --after-bias once
# that is done.
#
# Input:  a validated config (cli.py config validate --dataset "$DATASET")
# Output: everything under output_dir from the config
# Usage:
#   source workflows/molab/env.sh
#   bash workflows/molab/run_all.sh                 # 00.0 .. 03.1, then stop
#   bash workflows/molab/run_all.sh --after-bias    # 03.2 .. 04.1
#   bash workflows/molab/run_all.sh --dry-run
# Prerequisites: setup_molab.sh has run; env.sh is sourced.

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
        -h|--help)    sed -n '2,29p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown option $a" >&2; exit 1 ;;
    esac
done

# The 03.0 array is (folds x bias factors) - 1. Both come from the config, so
# derive it rather than hardcoding sbatch's 0-19 default, which assumes 5x4.
n_folds=$(python3 "${REPO_ROOT}/lib/python/utils/config.py" export \
    "${REPO_ROOT}/config/${DATASET}/config.yaml" \
    | sed -n 's/^bias_sweep_folds=( \(.*\) )$/\1/p' | wc -w)
n_factors=$(python3 "${REPO_ROOT}/lib/python/utils/config.py" export \
    "${REPO_ROOT}/config/${DATASET}/config.yaml" \
    | sed -n 's/^bias_factors=( \(.*\) )$/\1/p' | wc -w)
sweep_max=$(( n_folds * n_factors - 1 ))
fold_max=$(( n_folds - 1 ))

echo "=== dataset=${DATASET}  folds=${n_folds}  bias factors=${n_factors}  03.0 array=0-${sweep_max}"
echo

if [[ "${PHASE}" == "pre-bias" ]]; then
    # --- signal, peaks, background ------------------------------------------
    ${RUN} ${DRY} 00.0.prepare_signal.sh        || exit 1   # pixi: Tn5 shift + cut-site bigwig
    ${RUN} ${DRY} 00.1.preprocess_peaks.sh      || exit 1   # pixi: blacklist filter -> narrowPeak
    ${RUN} ${DRY} 01.0.preprocess_nonpeaks.sh   || exit 1   # container: GC-matched negatives

    # QC comes AFTER the negatives: the comparative half scores peaks against
    # them. Advisory, never fails the pipeline.
    ${RUN} ${DRY} 02.0.qc_signal_peaks.sh       || exit 1   # pixi: TSS enrichment, AUROC

    # --- bias sweep ---------------------------------------------------------
    ${RUN} ${DRY} --array "0-${sweep_max}" 03.0.train_bias_model.sh || exit 1   # GPU
    ${RUN} ${DRY} 03.1.select_bias.sh           || exit 1   # container: pick a winner per fold

    cat <<MSG

=== Stopping here, on purpose.

  Review   ${REPO_ROOT:-.}/<output_dir>/plots/bias_model_selection/${DATASET}/
  then copy the winners from selected_bias_per_fold.tsv into fold_bias_suffix
  in config/${DATASET}/config.yaml.

  03.1 also flags a winner sitting at an END of the swept range (sweep_edge):
  select_best answers "best of what we tried" and cannot see past the grid it
  was given. If it fires, widen bias_factors before continuing.

  Then:  bash workflows/molab/run_all.sh --after-bias
MSG
    exit 0
fi

# --- full model -------------------------------------------------------------
${RUN} ${DRY} --array "0-${fold_max}" 03.2.qc_selected_bias.sh      || exit 1   # GPU: predict + DeepLIFT
${RUN} ${DRY} --array "0-${fold_max}" 03.3.modisco_selected_bias.sh || exit 1   # CPU: TF-MoDISco
${RUN} ${DRY} --array "0-${fold_max}" 04.0.train_full_model.sh  || exit 1   # GPU
${RUN} ${DRY} 04.1.qc_run_full_model.sh                         || exit 1

echo
echo "=== 00.0 -> 04.1 complete for ${DATASET}."
