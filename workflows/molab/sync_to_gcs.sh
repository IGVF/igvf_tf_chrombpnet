#!/bin/bash
# sync_to_gcs.sh
# Purpose: Copy everything this box would lose to GCS. molab has no persistent
#   disk -- when the session ends the machine is gone -- so anything not copied
#   out is lost.
#
# Backs up two things, and the SECOND is the one that matters most:
#
#   results/   outputs, metadata, logs, plots. Large, but regenerable from the
#              inputs and the code.
#   repo.bundle  a git bundle of the working branch. This is NOT regenerable.
#              The commits live only on this box: the pipeline is cloned from
#              IGVF/igvf_tf_chrombpnet, which we cannot push to. A bundle is a
#              single file holding the full history, restored with
#              `git clone repo.bundle` or `git fetch repo.bundle <branch>`.
#              It is a few hundred KB against ~1 GB of results, so there is no
#              reason not to write it every time.
#
# Also copies the dataset config, so a restore has the exact parameters used.
#
# Input:  ${output_dir} from the dataset config, and this git checkout
# Output: gs://<bucket>/<prefix>/{results/,repo.bundle,config.yaml,MANIFEST.txt}
# Usage:
#   source workflows/molab/env.sh
#   bash workflows/molab/sync_to_gcs.sh                 # sync everything
#   bash workflows/molab/sync_to_gcs.sh --bundle-only   # just the commits (fast)
#   bash workflows/molab/sync_to_gcs.sh --dry-run
# Prerequisites: `gcloud auth login` -- the bucket is public to READ but not to
#   write, so an unauthenticated run fails with HTTP 401.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workflows/molab/env.sh
source "${SCRIPT_DIR}/env.sh"

DEST="${MOLAB_GCS_DEST:-gs://broad-buenrostro-pipeline-genome-annotations/chrombpnet/test_data_d0}"
BUNDLE_ONLY=0
DRY=""
for a in "$@"; do
    case "$a" in
        --bundle-only) BUNDLE_ONLY=1 ;;
        --dry-run)     DRY="--dry-run" ;;
        -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown option $a" >&2; exit 1 ;;
    esac
done

command -v gcloud >/dev/null 2>&1 || { echo "ERROR: gcloud not on PATH" >&2; exit 1; }
if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q .; then
    echo "ERROR: no authenticated gcloud account." >&2
    echo "  The bucket is public to read but NOT to write; anonymous upload returns 401." >&2
    echo "  Run:  gcloud auth login --no-browser" >&2
    exit 1
fi

log() { echo "[$(date '+%F %T')] $*"; }

output_dir=$(python3 "${REPO_ROOT}/lib/python/utils/config.py" export \
    "${DATASET_CONFIG:-${REPO_ROOT}/config/${DATASET}/config.yaml}" \
    | sed -n 's/^output_dir=//p' | tr -d '"')
[[ -n "${output_dir}" ]] || { echo "ERROR: could not read output_dir from the config" >&2; exit 1; }

staging=$(mktemp -d)
trap 'rm -rf "${staging}"' EXIT

# --- the commits ------------------------------------------------------------
branch=$(git -C "${REPO_ROOT}" branch --show-current)
log "bundling ${branch} ($(git -C "${REPO_ROOT}" rev-list --count HEAD) commits)"
git -C "${REPO_ROOT}" bundle create "${staging}/repo.bundle" --all >/dev/null 2>&1 \
    || { echo "ERROR: git bundle failed" >&2; exit 1; }

{
    echo "created:      $(date -Is)"
    echo "branch:       ${branch}"
    echo "head:         $(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "dataset:      ${DATASET}"
    echo "output_dir:   ${output_dir}"
    echo "dirty:        $(git -C "${REPO_ROOT}" status --porcelain | wc -l) uncommitted path(s)"
    echo
    echo "restore the code with:"
    echo "  gcloud storage cp ${DEST}/repo.bundle ."
    echo "  git clone repo.bundle igvf_tf_chrombpnet && cd \$_ && git checkout ${branch}"
} > "${staging}/MANIFEST.txt"

cp "${DATASET_CONFIG:-${REPO_ROOT}/config/${DATASET}/config.yaml}" "${staging}/config.yaml" 2>/dev/null

log "-> ${DEST}/ (bundle, manifest, config)"
gcloud storage cp ${DRY:+--dry-run} \
    "${staging}/repo.bundle" "${staging}/MANIFEST.txt" "${staging}/config.yaml" \
    "${DEST}/" || exit 1

if [[ "${BUNDLE_ONLY}" == "1" ]]; then
    log "done (--bundle-only)"
    exit 0
fi

# --- the results ------------------------------------------------------------
# rsync, not cp: this is re-run after every step, and only the new files should
# move. No --delete-unmatched-destination-objects: a partial local tree must
# never remove a good remote copy.
log "syncing ${output_dir} -> ${DEST}/results/"
gcloud storage rsync ${DRY:+--dry-run} --recursive \
    "${output_dir}" "${DEST}/results" || exit 1

log "done. $(du -sh "${output_dir}" | cut -f1) under ${DEST}/results/"
