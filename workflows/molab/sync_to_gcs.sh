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
output_dir=$(python3 "${REPO_ROOT}/lib/python/utils/config.py" export "${config_file}" \
    | sed -n 's/^output_dir=//p' | tr -d '"')
[[ -n "${output_dir}" ]] || { echo "ERROR: could not read output_dir from ${config_file}" >&2; exit 1; }

if [[ "${RESTORE}" == "1" ]]; then
    log "restoring ${DEST}/results/ -> ${output_dir} (missing files only)"
    n_got=0; n_have=0; n_fail=0
    # fd 3, not stdin: gcs_fetch runs commands that read stdin.
    while read -r obj <&3; do
        dest="${output_dir}/${obj#"${PREFIX}/results/"}"
        if [[ -e "${dest}" ]]; then
            n_have=$((n_have + 1))
        elif [[ "${DRY}" == "1" ]]; then
            echo "would restore: ${dest}"
        elif gcs_fetch "${obj}" "${dest}"; then
            n_got=$((n_got + 1))
        else
            n_fail=$((n_fail + 1))
        fi
    done 3< <(gcs_list "${PREFIX}/results/")
    log "restore done: ${n_got} fetched, ${n_have} already here, ${n_fail} failed"
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
    # fd 3, not stdin: gcs_put runs commands that read stdin.
    while IFS= read -r -d '' f <&3; do
        put "${f}" "results/${f#"${output_dir}/"}"
    done 3< <(find "${output_dir}" -type f -print0 | sort -z)
fi

log "done: ${n_up} uploaded, ${n_same} unchanged, ${n_fail} failed -> ${DEST}/"
(( n_fail == 0 ))
