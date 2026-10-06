#!/bin/bash
# box_rates.sh
# Purpose: training throughput of a running box, from its 03.0 task logs: the
#   latest step time of every training that logged in the last few minutes,
#   summed per GPU and, when the box runs BOX_MPS=half, per MPS group -- the
#   A/B the setting exists for.
#
# Reads only each log's first line (run_box.sh writes "[box] gpu=N mps=on|off"
# there) and its last 4 KB, where Keras' progress bar keeps the current
# "NNms/step". Light enough for the login node, which must not run anything
# heavy (it has 15 GB and the Claude session lives on it).
#
# Input:  $1 = the box directory, ${DATASET_ROOT}/chrombpnet/box/<job id>
#         RATES_ACTIVE_MIN: a log counts as training if written within this many
#         minutes (default 3)
# Output: per GPU: trainings, mean ms/step, it/s summed; per MPS group: it/s per GPU
# Usage:  bash workflows/dcai/box_rates.sh /dcai/projects/iu_0109/datasets/amsc/chrombpnet/box/517784

set -uo pipefail
box="${1:?usage: box_rates.sh <box dir>}"

find "${box}/logs" -name '03.0.*.log' -mmin "-${RATES_ACTIVE_MIN:-3}" 2>/dev/null | while read -r f; do
    where="$(head -c 200 "${f}" | head -n 1 | sed -n 's/^\[box\] gpu=\([0-9]*\) mps=\([a-z]*\).*/\1 \2/p')"
    ms="$(tail -c 4000 "${f}" | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' \
        | grep -o '[0-9.]*ms/step' | tail -n 1 | sed 's/ms\/step//')"
    [[ -n "${ms}" ]] && echo "${where:-? ?} ${ms}"
done | awk '
    $3 > 0 {
        g = $1; m[g] = $2; n[g]++; ms[g] += $3; it[g] += 1000 / $3
    }
    END {
        printf "%-4s %-4s %10s %12s %14s\n", "GPU", "MPS", "trainings", "mean ms/step", "it/s summed"
        for (i = -1; i < 64; i++) {
            g = (i < 0 ? "?" : i "")
            if (!(g in n)) continue
            printf "%-4s %-4s %10d %12.1f %14.0f\n", g, m[g], n[g], ms[g] / n[g], it[g]
            grp_it[m[g]] += it[g]; grp_gpus[m[g]]++; grp_n[m[g]] += n[g]
        }
        for (k in grp_it)
            printf "MPS %-3s: %d GPU(s), %d trainings, %.0f it/s per GPU\n", k, grp_gpus[k], grp_n[k], grp_it[k] / grp_gpus[k]
    }'
