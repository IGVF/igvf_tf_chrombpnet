#!/bin/bash
#SBATCH --job-name=bias_sweep
# Measured, not guessed: the July 2026 sweep on this dataset peaked at 23.2 GB
# (mean 18.6 GB over 11 recorded steps) against a 128 GB request. 48 GB keeps
# ~2x headroom and matters for scheduling -- 128 GB at 4 CPUs is exactly the
# `gpu` partition's 32 GB/core ceiling, so the job could only land on a node
# with 128 GB free and never backfilled into a smaller gap.
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Two limits at once, an upper and a lower.
#   upper: the loaded cuda/11.5 cannot drive Ada (GPU_CC 8.9, L40S) or Hopper
#          (9.0, H100/H200), so those are excluded for correctness. Lifting
#          that needs the env off tensorflow==2.8/CUDA 11, not a flag.
#   lower: 7.0/7.5 (V100, TITAN_V, RTX_2080Ti) are ELIGIBLE for cuda 11.5 but
#          are excluded on purpose, to avoid the slowest silicon. The July 2026
#          bias_sweep runs, by node: TITAN_V (7.0) 20:51 and 22:24; RTX_2080Ti
#          (7.5) 12:09 and 05:45; A100_SXM4 (8.0) 08:31. So CC 7.0 is clearly
#          the slow tier and worth excluding. Note the evidence does NOT show
#          8.0 beating 7.5 -- the single fastest run was a 2080Ti -- and the
#          runs differ by fold and bias factor, so this is a coarse signal, not
#          a benchmark. If the 8.0|8.6 queue is deep, adding GPU_CC:7.5 back at
#          submit time is defensible.
# Leaves GPU_CC 8.0 (A100_SXM4/A100_PCIE) and 8.6 (A40, RTX_3090).
#SBATCH --constraint="GPU_CC:8.0|GPU_CC:8.6"
#SBATCH --time=2-0
#SBATCH --partition=gpu,owners
#SBATCH --array=0-19
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 03.0.train_bias_model.sh
# Purpose: Train Tn5 bias models for a sweep of bias threshold factors on
#   bias_dataset (defined in dataset_config.sh), then compute fast QC metrics
#   (counts/profile Pearson r, JSD) for each - the metrics select_bias_model.py
#   (03.1) needs to pick a winner per fold. Deliberately skips the expensive
#   interpretation + TF-MoDISco QC (that only runs on the selected bias model,
#   in 03.2.qc_selected_bias.sh) - running it on every fold x factor combo is
#   what made this step slow before.
#
# Cost: this array is len(bias_sweep_folds) x len(bias_factors) GPU jobs — 20 by
#   default. Set bias_sweep_folds in config.yaml to pilot on fewer folds first;
#   note that per-fold winners genuinely differ in practice (igvf3 selected
#   _08/_08/_06/_08/_07), so a pilot informs the shortlist rather than replacing
#   the per-fold choice. `cli.py config validate` prints the required --array.
#
#   Array index maps to fold x bias_factor: task_id = fold_idx * n_factors + factor_idx
#     e.g. with 4 bias_factors: tasks 0-3 = fold 0, tasks 4-7 = fold 1, etc.
#   --array default (0-19) assumes 5 folds x 4 factors; override for datasets
#   with a different bias_factors length (e.g. igvf11_h7_hesc has 6 -> 0-29).
#
# Usage:
#   export DATASET_DIR=/path/to/igvf_tf_collab/<dataset>
#   sbatch 03.0.train_bias_model.sh              # all folds x factors
#   sbatch --array=0 03.0.train_bias_model.sh    # fold 0, first bias factor only (quick test)
#   BIAS_BATCH_SIZE=128 sbatch --export=ALL --array=2 03.0.train_bias_model.sh
#                                                # one factor at batch 128 -> bias_model<sfx>_bs128/
#
# After all jobs complete, run 03.1.select_bias.sh, then 03.2.qc_selected_bias.sh,
# then 04.0.train_full_model.sh.

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

# Which factors to sweep: 02.0's scan when it exists, else the config.
load_bias_sweep

n_factors=${#bias_factors[@]}
fold_idx=$(( SLURM_ARRAY_TASK_ID / n_factors ))
factor_idx=$(( SLURM_ARRAY_TASK_ID % n_factors ))
# bias_sweep_folds defaults to folds; set it in config.yaml to pilot the sweep
# on a subset (see "Cost" in the header).
fold="${bias_sweep_folds[$fold_idx]}"
bf="${bias_factors[$factor_idx]}"
suffix="${bias_suffixes_sweep[$factor_idx]}"
[[ -z "${fold}" || -z "${bf}" ]] && { echo "Invalid array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

# Even when the factors came from the config, refuse one the scan has already
# shown cannot work. The failure is otherwise an IndexError deep inside the
# one-hot encoder, minutes into a GPU allocation.
if [[ -s "${bias_scan_file}" ]]; then
    _v=$(awk -F'\t' -v f="${bf}" '$1==f {print $7}' "${bias_scan_file}")
    if [[ "${_v}" == fail* ]]; then
        echo "ERROR: bias_threshold_factor ${bf} cannot train on this dataset." >&2
        echo "  ${bias_scan_file} says: ${_v}" >&2
        echo "  Pick a factor marked ok there, or re-run 02.0 if the data changed." >&2
        exit 1
    fi
fi



metadata_start "03.0.train_bias_model"



set_signal_args
signal_file="${signal_path}"
peaks_file="${peaks_dir}/${bias_dataset}_${peak_type}_peaks_no_blacklist.narrowPeak"
negatives_file="${data_path}/${bias_dataset}/output_${peak_type}_fold_${fold}_negatives.bed"
fold_json="${folds_dir}/fold_${fold}.json"
file_prefix="${bias_dataset}_${peak_type}_fold_${fold}"
# Batch size: chrombpnet's default (64) unless the config sets bias_batch_size
# or the environment sets BIAS_BATCH_SIZE (the environment wins, so a one-off
# test needs no config edit). A different batch size trains a different model,
# so it gets its own directory -- `_bs<N>` after the factor suffix -- and can
# never overwrite, or be skipped as, the default-size model. 03.1 selects among
# the default-size directories only; the `_bs<N>` runs are for comparison.
batch_size="${BIAS_BATCH_SIZE:-${bias_batch_size:-}}"
batch_args=()
batch_tag=""
if [[ -n "${batch_size}" ]]; then
    [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: batch size must be a positive integer, got '${batch_size}'" >&2; exit 1; }
    batch_args=( -bs "${batch_size}" )
    batch_tag="_bs${batch_size}"
fi
out_dir="${results_path}/bias_models/bias_model${suffix}${batch_tag}/${bias_dataset}_${peak_type}_fold_${fold}"
model_file="${out_dir}/models/${file_prefix}_bias.h5"


metadata_inputs+=( "${signal_type}=${signal_file}" "peaks=${peaks_file}" "negatives=${negatives_file}" "fold_json=${fold_json}" )
require_input "${signal_file}" 00.0.prepare_signal.sh
require_input "${peaks_file}" 00.1.preprocess_peaks.sh
require_input "${negatives_file}" 01.0.preprocess_nonpeaks.sh
require_input "${fold_json}"
require_input "${genome_fa}"   "cli.py download-references"
require_input "${chrom_sizes}" "cli.py download-references"
preflight_check

load_gpu_modules
activate_env "${CONDA_ENV}"
gpu_env
metadata_outputs+=( "bias_model=${model_file}" )
# The metrics JSON is a real output of this step and the ONLY thing 03.1 reads
# from it, so it belongs in the record too -- otherwise a run that trains a
# model but fails to score it looks complete in the metadata.
metrics_json="${out_dir}/evaluation/${file_prefix}_bias_metrics.json"
metadata_outputs+=( "bias_metrics=${metrics_json}" )
metadata_params+=( "fold=${fold}" "bias_factor=${bf}" "bias_suffix=${suffix}" "batch_size=${batch_size:-64}" )
echo "[$(date)] [fold ${fold} bias=${bf}] Training bias model"
echo "  output dir : ${out_dir}"

# Require BOTH markers before skipping, the same rule 04.0 uses, and for the
# same reason. ${model_file} alone does NOT mean training finished: chrombpnet
# hands it to a Keras ModelCheckpoint with save_best_only=True
# (chrombpnet/training/train.py), so it appears as soon as val_loss first
# improves -- after epoch 1. This array runs on `owners`, where preemption
# mid-training is routine, and the old check kept that epoch-1 model forever
# and scored it as if it were the trained one. ${metrics_json} is only written
# once training has returned and the scoring stage has run.
#
# The cost: if scoring fails for its own reasons, a complete model is retrained.
# That is the trade 04.0 already makes deliberately, and scoring failures are
# now loud (see the guard below) rather than silent.
if [[ -f "${model_file}" && -f "${metrics_json}" ]]; then
    echo "  Model trained and scored already, nothing to do."
    exit 0
elif [[ -f "${model_file}" ]]; then
    echo "  Found ${model_file} but no ${metrics_json}." >&2
    echo "  That model may be a mid-training checkpoint, so it is discarded and retrained." >&2
fi

for f in "${signal_file}" "${peaks_file}" "${negatives_file}" "${fold_json}"; do
    [[ -f "${f}" ]] || { echo "  Missing input: ${f}" >&2; exit 1; }
done

rm -rf "${out_dir}"
mkdir -p "${out_dir}"

# Peak RSS has to be measured inside the process that allocates the training
# arrays; the metadata trap runs in a sibling and would otherwise record its
# own ~26 MB as this step's footprint. See docs/resource-measurements.md.
METADATA_RSS_FILE="${out_dir}/.peak_rss_gb"
export METADATA_RSS_FILE

python "${src_dir}/chrombpnet_train.py" \
    --prepared-bigwig "${data_path}/signal" \
    ${prepared_args[@]+"${prepared_args[@]}"} -- \
    bias train \
    "${signal_args[@]}" \
    -d "${assay}" \
    -g "${genome_fa}" \
    -c "${chrom_sizes}" \
    -p "${peaks_file}" \
    -n "${negatives_file}" \
    -fl "${fold_json}" \
    -b "${bf}" \
    -o "${out_dir}" \
    -fp "${file_prefix}" \
    ${batch_args[@]+"${batch_args[@]}"}
_train_status=$?
if [[ -s "${METADATA_RSS_FILE}" ]]; then
    metadata_metrics+=( "peak_rss_gb=$(<"${METADATA_RSS_FILE}")" )
fi
if [[ ${_train_status} -ne 0 || ! -f "${model_file}" ]]; then
    echo "ERROR: chrombpnet bias train failed for fold ${fold} bias=${bf} (bias threshold factor may be too low/high for this fold - see stdout above)." >&2
    exit 1
fi

echo "[$(date)] [fold ${fold} bias=${bf}] Computing fast QC metrics."

# This guard is the difference between a failed sweep cell and a WRONG bias
# selection. There is no `set -e` here, and this is the last real command, so
# without it a failure exits 0 and SLURM records COMPLETED. 03.1 then reads the
# sweep through select_bias_model.load_metrics, which only logs
# "missing metrics file" and `continue`s -- so the fold x factor cell silently
# vanishes and a per-fold winner gets chosen from an incomplete grid.
python "${src_dir}/predict_bias_metrics.py" \
    --bias-model "${model_file}" \
    --output-dir "${out_dir}" \
    --file-prefix "${file_prefix}" \
    --genome "${genome_fa}" \
    --fold-json "${fold_json}"
if [[ $? -ne 0 || ! -f "${metrics_json}" ]]; then
    echo "ERROR: predict_bias_metrics.py failed for fold ${fold} bias=${bf}; ${metrics_json} was not written." >&2
    echo "       03.1 would silently drop this cell from the sweep rather than fail." >&2
    exit 1
fi

echo "[$(date)] [fold ${fold} bias=${bf}] Done."
