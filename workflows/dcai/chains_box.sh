#!/bin/bash
# chains_box.sh
# Purpose: run full_chain.sh (04.0 -> 08.0) for many datasets inside a running
#   box, one dataset after another as their bias models are QC'd and the GPUs
#   have room.
#
# A dataset starts once every fold's selected bias model exists and 03.3's two
# motif-QC summaries are written for it (so the QC is there to review), or once
# bias_qc_box.py is no longer running (a fold whose QC failed does not block it
# for ever). Datasets start in the order given.
#
# GPU room: a full-model GPU step (04.0, 05.0, 04.4, 04.3) keeps a GPU close to
# fully busy at batch 64 (the fork's K562 profiling), so more than a couple per
# GPU only slows every one of them. A new dataset (five fold chains) starts
# only while fewer than CHAINS_MAX_GPU_STEPS - 5 such steps run (default 16:
# two per GPU on eight GPUs). Each chain's folds go to five GPUs, rotating.
#
# Keeping the box: holds it with <box>/.hold.chains while datasets are left,
# and releases it on exit like full_chain.sh (.chains_done back when no other
# .hold* is left and bias_qc_box.py is not running).
#
# Input:  $@ = the datasets' config.yaml files, in order; BOX_DIR; the
#         environment the box ran 03.0 with (full_chain.sh's).
#         CHAINS_MAX_GPU_STEPS (default 16), CHAINS_GPUS (default 8).
# Output: full_chain.sh's, per dataset; <box>/full_chain.<dataset>.log.
# Usage:  srun --jobid <box> --overlap -N1 -n1 bash workflows/dcai/chains_box.sh <config.yaml>...
#         CHAINS_DRY_RUN=1 ... prints which datasets are ready and starts nothing.

set -uo pipefail
: "${BOX_DIR:?set BOX_DIR to the directory of the running box}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
max_steps="${CHAINS_MAX_GPU_STEPS:-16}"
n_gpus="${CHAINS_GPUS:-8}"

yaml() { sed -n "s/^$1: *\"\{0,1\}\([^\"]*\)\"\{0,1\} *\$/\1/p" "$2"; }

# ready <config> -- every fold's bias model exists, and its QC is done or no
# longer coming.
ready() {
    local c="$1" name out f suffix prefix d qc_running=0
    name="$(yaml dataset_name "${c}")"; out="$(yaml output_dir "${c}")"
    pgrep -f "bias_qc_box.py --box-dir ${BOX_DIR}" > /dev/null && qc_running=1
    for f in 0 1 2 3 4; do
        suffix="$(sed -n "/^fold_bias_suffix:/,/^[^ ]/s/^ *\"${f}\": *\"\([^\"]*\)\"/\1/p" "${c}")"
        [[ -n "${suffix}" ]] || return 1
        prefix="${name}_all_fold_${f}"
        d="${out}/bias_models/bias_model${suffix}/${prefix}/evaluation"
        [[ -f "${d}/${prefix}_bias_metrics.json" ]] || return 1
        if (( qc_running )); then
            [[ -f "${d}/motif_qc_counts/${prefix}_bias_motif_qc.json" && -f "${d}/motif_qc_profile/${prefix}_bias_motif_qc.json" ]] || return 1
        fi
    done
}

gpu_steps() {
    pgrep -fc "SLURM/(04\.0\.train_full_model|05\.0\.get_contrib_scores|04\.4\.qc_full_model_interpret|04\.3\.generate_predictions)\.sh" || true
}

if [[ "${CHAINS_DRY_RUN:-0}" == "1" ]]; then
    echo "full-model GPU steps running: $(gpu_steps)"
    for c in "$@"; do
        if ready "${c}"; then echo "ready    ${c}"; else echo "waiting  ${c}"; fi
    done
    exit 0
fi

hold="${BOX_DIR}/.hold.chains"
touch "${hold}"
rm -f "${BOX_DIR}/.chains_done"
release() {
    rm -f "${hold}"
    if ! compgen -G "${BOX_DIR}/.hold*" > /dev/null && ! pgrep -f "bias_qc_box.py --box-dir ${BOX_DIR}" > /dev/null; then
        touch "${BOX_DIR}/.chains_done"
        echo "[$(date)] no other holder: released the box (.chains_done)"
    fi
}
trap release EXIT

pending=( "$@" )
k=0
while (( ${#pending[@]} )); do
    left=()
    for c in "${pending[@]}"; do
        if (( $(gpu_steps) + 5 <= max_steps )) && ready "${c}"; then
            gpus="$(for i in 0 1 2 3 4; do echo $(( (5 * k + i) % n_gpus )); done | paste -sd,)"
            name="$(yaml dataset_name "${c}")"
            echo "[$(date)] starting ${name} on GPUs ${gpus}"
            bash "${REPO_ROOT}/workflows/dcai/full_chain.sh" "${c}" "${gpus}" > "${BOX_DIR}/full_chain.${name}.log" 2>&1 &
            k=$(( k + 1 ))
            sleep 300   # let its five 04.0 runs show up in gpu_steps before counting again
        else
            left+=( "${c}" )
        fi
    done
    pending=( "${left[@]+"${left[@]}"}" )
    (( ${#pending[@]} )) && sleep 120
done
echo "[$(date)] every dataset started; waiting for the chains"
wait
echo "[$(date)] all chains finished"
