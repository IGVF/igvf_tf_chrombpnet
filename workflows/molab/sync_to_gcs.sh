#!/bin/bash
# sync_to_gcs.sh
# Purpose: Copy everything this box would lose to the molab bucket. molab has
#   no persistent disk -- when the session ends the machine is gone, and a
#   session that dies comes back with only part of /marimo -- so anything not
#   copied out is lost. Run it after every step.
#
# Backs up two things, and the SECOND is the one that matters most:
#
#   results/   outputs, metadata, logs, plots. Large, but regenerable from the
#              inputs and the code.
#   repo.bundle  a git bundle of every branch. This is NOT regenerable.
#              The commits live only on this box: the pipeline is cloned from
#              IGVF/igvf_tf_chrombpnet, which we cannot push to. A bundle is a
#              single file holding the full history, restored with
#              `git clone repo.bundle` or `git fetch repo.bundle <branch>`.
#              It is a few hundred KB against ~1 GB of results, so there is no
#              reason not to write it every time.
#
# Also copies the dataset config, so a restore has the exact parameters used.
#
# Why not gcloud: there is none on the box. gcs.sh mints a read_write token
# from GCP_SA_JSON with openssl and uploads with curl -- the same helpers
# setup_molab.sh downloads with. Each file is skipped when the bucket already
# holds identical bytes (size + md5), so only what a step produced moves, and
# nothing remote is ever deleted: a partial local tree must never remove a
# good remote copy.
#
# Input:  ${output_dir} from the dataset config, and this git checkout
# Output: gs://${GCP_BUCKET}/${MOLAB_GCS_PREFIX}/{results/,repo.bundle,config.yaml,MANIFEST.txt}
# Usage:
#   source workflows/molab/env.sh
#   bash workflows/molab/sync_to_gcs.sh                 # sync everything
#   bash workflows/molab/sync_to_gcs.sh --bundle-only   # just the commits (fast)
#   bash workflows/molab/sync_to_gcs.sh --dry-run       # list, upload nothing
#   bash workflows/molab/sync_to_gcs.sh --restore       # bucket -> box, the reverse
#
# --restore fetches results/ back into ${output_dir}, but ONLY files missing
# locally. A local file that differs from the bucket is newer work not synced
# yet (a step re-ran), so it is never overwritten; downloads are md5-checked.
# run_step.sh calls it before running anything, so a fresh box sees what a
# dead one already finished and skips it.
# Prerequisites: GCP_BUCKET + GCP_SA_JSON in the .env env.sh sources, for a
#   service account allowed to write objects under the prefix.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workflows/molab/env.sh
source "${SCRIPT_DIR}/env.sh"

PREFIX="${MOLAB_GCS_PREFIX:-chrombpnet/test_data_d0}"
BUNDLE_ONLY=0
DRY=0
RESTORE=0
for a in "$@"; do
    case "$a" in
        --bundle-only) BUNDLE_ONLY=1 ;;
        --dry-run)     DRY=1 ;;
        --restore)     RESTORE=1 ;;
        -h|--help)     sed -n '2,44p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown option $a" >&2; exit 1 ;;
    esac
done

log() { echo "[$(date '+%F %T')] $*"; }

[[ -n "${GCP_BUCKET:-}" ]]  || { echo "ERROR: GCP_BUCKET is not set (see workflows/molab/.env.example)" >&2; exit 1; }
[[ -n "${GCP_SA_JSON:-}" ]] || { echo "ERROR: GCP_SA_JSON is not set (see workflows/molab/.env.example)" >&2; exit 1; }

if [[ "${RESTORE}" == "1" ]]; then GCS_SCOPE=read_only; else GCS_SCOPE=read_write; fi
# shellcheck source=workflows/molab/gcs.sh
source "${SCRIPT_DIR}/gcs.sh"
GCS_TOKEN=$(gcs_token) || { echo "ERROR: could not mint a GCS token" >&2; exit 1; }
DEST="gs://${GCP_BUCKET}/${PREFIX}"

config_file="${DATASET_CONFIG:-${REPO_ROOT}/config/${DATASET}/config.yaml}"
# As config.sh derives it, like every step: a sed over the YAML export left an
# output_dir of "${DATASET_ROOT}/..." unexpanded and synced the wrong tree.
# shellcheck disable=SC2016  # expanded by molab_config's child bash
output_dir="$(DATASET_CONFIG="${config_file}" molab_config '${results_path}')"
[[ -n "${output_dir}" ]] || { echo "ERROR: could not read output_dir from ${config_file}" >&2; exit 1; }

if [[ "${RESTORE}" == "1" ]]; then
    log "restoring ${DEST}/results/ -> ${output_dir} (missing files only)"
    n_got=0; n_have=0; n_fail=0; n_linked=0
    # Paths the sync recorded as hard links are not downloaded (older syncs
    # may have uploaded them as separate objects): they are re-linked below.
    declare -A as_link=()
    links="${output_dir}/hardlinks.tsv"
    if [[ ! -e "${links}" && "${DRY}" == "0" ]]; then
        gcs_fetch "${PREFIX}/results/hardlinks.tsv" "${links}" 2>/dev/null || rm -f "${links}"
    fi
    if [[ -f "${links}" ]]; then
        while IFS=$'\t' read -r link_rel _; do as_link["${link_rel}"]=1; done < "${links}"
    fi
    # fd 3, not stdin: gcs_fetch runs commands that read stdin.
    while read -r obj <&3; do
        rel="${obj#"${PREFIX}/results/"}"
        dest="${output_dir}/${rel}"
        if [[ -n "${as_link[${rel}]:-}" ]]; then
            continue
        elif [[ -e "${dest}" ]]; then
            n_have=$((n_have + 1))
        elif [[ "${DRY}" == "1" ]]; then
            echo "would restore: ${dest}"
        elif gcs_fetch "${obj}" "${dest}"; then
            n_got=$((n_got + 1))
        else
            n_fail=$((n_fail + 1))
        fi
    done 3< <(gcs_list "${PREFIX}/results/")
    if [[ -f "${links}" && "${DRY}" == "0" ]]; then
        while IFS=$'\t' read -r link_rel target_rel; do
            [[ -e "${output_dir}/${link_rel}" || ! -f "${output_dir}/${target_rel}" ]] && continue
            mkdir -p "$(dirname "${output_dir}/${link_rel}")"
            ln "${output_dir}/${target_rel}" "${output_dir}/${link_rel}" && n_linked=$((n_linked + 1))
        done < "${links}"
    fi
    log "restore done: ${n_got} fetched, ${n_linked} re-linked, ${n_have} already here, ${n_fail} failed"
    (( n_fail == 0 ))
    exit
fi

n_up=0; n_same=0; n_fail=0
put() {
    if [[ "${DRY}" == "1" ]]; then
        echo "would upload: $1 -> ${DEST}/$2"
        return 0
    fi
    if gcs_put "$1" "${PREFIX}/$2"; then
        [[ "${GCS_PUT}" == "uploaded" ]] && { n_up=$((n_up + 1)); log "uploaded: ${DEST}/$2"; } || n_same=$((n_same + 1))
    else
        n_fail=$((n_fail + 1))
    fi
}

staging=$(mktemp -d)
trap 'rm -rf "${staging}"' EXIT

# --- the commits ------------------------------------------------------------
branch=$(git -C "${REPO_ROOT}" branch --show-current)
log "bundling ${branch} ($(git -C "${REPO_ROOT}" rev-list --count HEAD) commits)"
git -C "${REPO_ROOT}" bundle create "${staging}/repo.bundle" --all >/dev/null 2>&1 \
    || { echo "ERROR: git bundle failed" >&2; exit 1; }

{
    echo "branch:       ${branch}"
    echo "head:         $(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "dataset:      ${DATASET}"
    echo "output_dir:   ${output_dir}"
    echo "dirty:        $(git -C "${REPO_ROOT}" status --porcelain | wc -l) uncommitted path(s)"
    echo
    echo "restore the code with:"
    echo "  download ${DEST}/repo.bundle, then"
    echo "  git clone repo.bundle igvf_tf_chrombpnet && cd \$_ && git checkout ${branch}"
} > "${staging}/MANIFEST.txt"

log "-> ${DEST}/ (bundle, manifest, config)"
put "${staging}/repo.bundle" repo.bundle
put "${staging}/MANIFEST.txt" MANIFEST.txt
put "${config_file}" config.yaml

if [[ "${BUNDLE_ONLY}" == "0" ]]; then
    # --- the results --------------------------------------------------------
    log "syncing ${output_dir} -> ${DEST}/results/"
    # Hard links upload once. 03.0 links the prepared bigwig (~200 MB) into
    # every bias model's auxiliary/, where 03.0's scoring and 03.2 read it, and
    # a bucket has no links: each would be md5'd and uploaded as its own copy,
    # and restored as one. The SHORTEST path of each inode is uploaded -- the
    # original (preprocessing/signal/...), not a per-model link under a
    # directory 03.0 deletes on every retrain -- and the rest go to
    # results/hardlinks.tsv, which --restore turns back into links.
    links_file="${staging}/hardlinks.tsv"
    : > "${links_file}"
    declare -A first_path=()
    # fd 3, not stdin: gcs_put runs commands that read stdin.
    while IFS= read -r -d '' rec <&3; do
        ino="${rec%% *}"; rest="${rec#* }"; nlink="${rest%% *}"; f="${rest#* }"
        rel="${f#"${output_dir}/"}"
        [[ "${rel}" == "hardlinks.tsv" ]] && continue  # regenerated below, never uploaded from the tree
        if (( nlink > 1 )); then
            if [[ -n "${first_path[${ino}]:-}" ]]; then
                printf '%s\t%s\n' "${rel}" "${first_path[${ino}]}" >> "${links_file}"
                continue
            fi
            first_path["${ino}"]="${rel}"
        fi
        put "${f}" "results/${rel}"
    done 3< <(find "${output_dir}" -type f -printf '%i %n %p\0' \
                | awk 'BEGIN { RS = ORS = "\0" } { print length($0) "\t" $0 }' | sort -z -n -k1,1 -k2 | cut -z -f2-)
    if [[ -s "${links_file}" ]]; then
        put "${links_file}" "results/hardlinks.tsv"
        log "$(wc -l < "${links_file}") hard link(s) recorded in results/hardlinks.tsv, not uploaded"
    fi
fi

log "done: ${n_up} uploaded, ${n_same} unchanged, ${n_fail} failed -> ${DEST}/"
(( n_fail == 0 ))
