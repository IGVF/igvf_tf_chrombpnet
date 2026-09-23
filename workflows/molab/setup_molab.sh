#!/bin/bash
# setup_molab.sh
# Purpose: Prepare a molab (marimo cloud) box to run this pipeline end to end.
#   Updates apt, installs Apptainer and its userspace dependencies, fetches the
#   chrombpnet container from the private molab bucket and unpacks it, bakes
#   the low-memory one-hot encoder into it, deletes the image once the unpacked
#   copy is verified, installs pixi's `qc` environment, and downloads the d0
#   test data and the shared references. Idempotent: everything already
#   present is verified and skipped.
#
# Why a container at all: the cluster runs chrombpnet from `envs/chrombpnet.yml`,
# which pins tensorflow 2.8 / numpy 1.23 against CUDA 11 wheels. Those wheels do
# not exist for this hardware, so molab uses a prebuilt image instead (built
# from gs://${GCP_BUCKET}/chrombpnet/build_files/chrombpnet.Dockerfile). pixi's
# `qc` environment stands in for `envs/preprocess.yml` (it is a superset:
# python >= 3.12, pyranges1, pybigtools, pysam, pyfaidx, bedtools).
#
# Why the container is UNPACKED to a directory rather than run as a .sif: molab
# runs under gVisor, which has no loop devices and no kernel squashfs, and
# fuse-overlayfs fails there ("cannot read lower dirs: Function not
# implemented"). Apptainer therefore cannot mount a .sif at all. Unsquashing it
# to a plain directory sidesteps every one of those: Apptainer execs the
# directory using underlay bind mounts only. `unsquashfs` exits 2 because it
# cannot mknod device nodes under gVisor; that is harmless and expected, but it
# makes `apptainer build --sandbox` delete its own output, which is why this
# script calls unsquashfs directly.
#
# Why the .sif is DELETED as soon as it is unpacked: the box reports no disk
# quota (`df` says 8.0E), so nothing here can see how close it is to a real
# one, and the image is dead weight once unpacked. Its md5 is checked against
# the bucket and its entry count taken BEFORE extraction, so nothing reads it
# after unsquashfs returns; a failed count means re-downloading, which is what
# gcs_fetch's resume path is for. Peak: ~24 GB at the end of the unpack (image
# + sandbox), ~16 GB from then on. Two markers under `.igvf_molab/`:
# `unpacked` (sandbox matches the image) and `complete` (encoder verified), so
# a rerun never touches the image again and a failed bake re-bakes only.
#
# Why output goes to ${MOLAB_SETUP_LOG} and unsquashfs runs -no-progress: runs
# from marimo's web terminal repeatedly killed the whole session during the
# unpack, while the same script with output going to a file got through the
# same 24 GB peak. The cause is not proven, but the ~400k progress-bar redraws
# are the obvious difference, and if a session dies anyway the log in /marimo
# shows how far it got. Each line carries the space used on /: the memory
# figures gVisor exposes stay flat while files are written, so that is the
# only counter that moves.
#
# Why the one-hot encoder is baked in here but monkey-patched on the cluster:
# the cluster runs the stock .sif, so `src/chrombpnet_train.py` calls
# `utils.onehot.install()` at runtime. The sandbox is ours to edit, so here
# `lib/python/utils/onehot.py` is copied into chrombpnet's package and rebinds
# `one_hot.dna_to_one_hot` -- which reaches every entry point (interpretation,
# bigwig helpers), not only training. The stock file is kept as
# `one_hot.py.upstream`, the patch is re-applied from it on every run so edits
# to onehot.py propagate, and it is checked byte-for-byte against upstream
# inside the container before anything is deleted.
#
# Input:  none (everything is fetched)
# Output: ${MOLAB_SANDBOX}, the pixi `qc` env, ${MOLAB_DATA_DIR}, ${REFERENCE_ROOT}
# Usage:  bash workflows/molab/setup_molab.sh
#         bash workflows/molab/setup_molab.sh --skip-references
# Prerequisites: root (for apt), outbound HTTPS, and GCP_BUCKET + GCP_SA_JSON
#   (the service-account key as inline JSON) in the .env env.sh sources. Disk:
#   ~24 GB at the end of the unpack (image + sandbox), ~17 GB after it.

# shellcheck disable=SC2218  # 0.11.0 false positive (see CLAUDE.md): log() is defined before use

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workflows/molab/env.sh
source "${SCRIPT_DIR}/env.sh"

SIF_OBJECT="chrombpnet/containers/chrombpnet.sif"
TEST_DATA_PREFIX="chrombpnet/test_data_d0/"
MOLAB_DATA_DIR="${MOLAB_DATA_DIR:-/marimo/data/test_data_d0}"
APPTAINER_VERSION="1.5.3"
APT=(env DEBIAN_FRONTEND=noninteractive apt-get -y -qq
     -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

SKIP_REFERENCES=0
[[ "${1:-}" == "--skip-references" ]] && SKIP_REFERENCES=1

MOLAB_SETUP_LOG="${MOLAB_SETUP_LOG:-/marimo/setup_molab.log}"
mkdir -p "$(dirname "${MOLAB_SETUP_LOG}")"
exec > >(tee -a "${MOLAB_SETUP_LOG}") 2>&1

log() { echo "[$(date '+%F %T')] [/ used: $(df -h --output=used / | tail -1 | tr -d ' ')] $*"; }
log "setup_molab.sh started (log: ${MOLAB_SETUP_LOG})"

[[ -n "${GCP_BUCKET:-}" ]]  || { echo "ERROR: GCP_BUCKET is not set (see workflows/molab/.env.example)" >&2; exit 1; }
[[ -n "${GCP_SA_JSON:-}" ]] || { echo "ERROR: GCP_SA_JSON is not set (see workflows/molab/.env.example)" >&2; exit 1; }

# ── 0. apt ────────────────────────────────────────────────────────────────────
# The image ships with empty package lists, so every install below fails
# without this. No `upgrade`: nothing here needs newer packages, and on a box
# whose every written byte counts against a hidden limit it is pure cost.
log "apt: update"
"${APT[@]}" update

# ── 1. locale ─────────────────────────────────────────────────────────────────
# The image sets LC_ALL=en_US.UTF-8 but ships no generated locale, so every
# bash invocation prints a setlocale warning that buries real output.
if ! locale -a 2>/dev/null | grep -qi '^en_US\.utf-\?8$'; then
    log "generating en_US.UTF-8 locale"
    "${APT[@]}" install locales >/dev/null
    sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
    grep -q '^en_US.UTF-8' /etc/locale.gen || echo 'en_US.UTF-8 UTF-8' >> /etc/locale.gen
    locale-gen >/dev/null
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
fi

# ── 2. Apptainer + userspace deps ────────────────────────────────────────────
# squashfuse/fuse2fs/fuse-overlayfs are Apptainer's unprivileged mount helpers;
# squashfs-tools provides the unsquashfs used below. Debian trixie has no
# apptainer package, so take the upstream trixie build.
if ! command -v apptainer >/dev/null 2>&1; then
    log "installing apptainer ${APPTAINER_VERSION} and dependencies"
    "${APT[@]}" install \
        squashfs-tools squashfuse fuse2fs fuse3 uidmap fuse-overlayfs \
        libseccomp2 cryptsetup-bin ca-certificates curl openssl >/dev/null
    deb="$(mktemp -d)/apptainer.deb"
    curl -fsSL -o "${deb}" \
        "https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}-trixie%2B_amd64.deb"
    "${APT[@]}" install "${deb}" >/dev/null
    rm -f "${deb}"
    "${APT[@]}" clean
fi
log "apptainer: $(apptainer --version)"

# Overlay is unusable under gVisor; underlay (plain bind mounts) is not.
if grep -q '^enable overlay = yes' /etc/apptainer/apptainer.conf 2>/dev/null; then
    log "apptainer.conf: disabling overlay, enabling underlay (gVisor has no working overlayfs)"
    cp /etc/apptainer/apptainer.conf /etc/apptainer/apptainer.conf.orig
    sed -i 's/^enable overlay = .*/enable overlay = no/'  /etc/apptainer/apptainer.conf
    sed -i 's/^enable underlay = .*/enable underlay = yes/' /etc/apptainer/apptainer.conf
fi

# ── 3. GCS access ────────────────────────────────────────────────────────────
# The bucket is private and there is no gcloud on the box, so a read-only
# OAuth token is minted from the service-account key with openssl. The key
# reaches openssl through a pipe (process substitution) and never touches
# disk; the token reaches curl the same way, so neither shows in `ps`.
sa_field() {
    printf '%s' "${GCP_SA_JSON}" \
        | python3 -c 'import json, sys; sys.stdout.write(json.load(sys.stdin)[sys.argv[1]])' "$1"
}
b64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }

gcs_token() {
    local now header claims unsigned sig
    now=$(date +%s)
    header=$(printf '{"alg":"RS256","typ":"JWT"}' | b64url)
    claims=$(printf '{"iss":"%s","scope":"https://www.googleapis.com/auth/devstorage.read_only","aud":"https://oauth2.googleapis.com/token","iat":%d,"exp":%d}' \
        "$(sa_field client_email)" "${now}" "$((now + 3600))" | b64url)
    unsigned="${header}.${claims}"
    sig=$(printf '%s' "${unsigned}" | openssl dgst -sha256 -sign <(sa_field private_key) | b64url)
    curl -fsS https://oauth2.googleapis.com/token \
        --data-urlencode "grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer" \
        --data-urlencode "assertion@"<(printf '%s' "${unsigned}.${sig}") \
        | python3 -c 'import json, sys; sys.stdout.write(json.load(sys.stdin)["access_token"])'
}

GCS_TOKEN=""
auth_header() { printf 'Authorization: Bearer %s\n' "${GCS_TOKEN}"; }
urlenc() { python3 -c 'import sys, urllib.parse; sys.stdout.write(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }
gcs_api() { curl -fsS -H @<(auth_header) "https://storage.googleapis.com/storage/v1/b/${GCP_BUCKET}/o$1"; }

# "<size> <md5>" of one object; md5 is "-" for composite objects, which have none.
gcs_meta() {
    gcs_api "/$(urlenc "$1")?fields=size,md5Hash" \
        | python3 -c 'import json, sys; j = json.load(sys.stdin); print(j["size"], j.get("md5Hash") or "-")'
}

# Every object name under a prefix, one per line, directory placeholders skipped.
gcs_list() {
    local page="" out
    while :; do
        out=$(gcs_api "?prefix=$(urlenc "$1")&fields=items(name),nextPageToken${page:+&pageToken=$(urlenc "${page}")}")
        printf '%s' "${out}" | python3 -c 'import json, sys; [print(i["name"]) for i in json.load(sys.stdin).get("items", []) if not i["name"].endswith("/")]'
        page=$(printf '%s' "${out}" | python3 -c 'import json, sys; sys.stdout.write(json.load(sys.stdin).get("nextPageToken", ""))')
        [[ -n "${page}" ]] || break
    done
}

md5_b64() { openssl dgst -md5 -binary "$1" | base64; }

# Does <file> match the bucket's <size> and <md5>? md5 "-" means size only.
matches() {
    [[ -f "$1" ]] && [[ "$(stat -c %s "$1")" == "$2" ]] || return 1
    [[ "$3" == "-" ]] || [[ "$(md5_b64 "$1")" == "$3" ]]
}

# Download <object> to <dest> unless it is already there and intact. Resumes a
# partial <dest>.part; mints a fresh token between attempts because each lasts
# an hour, which a slow 8 GB download can outlive.
gcs_fetch() {
    local obj="$1" dest="$2" size md5 attempt
    read -r size md5 < <(gcs_meta "${obj}")
    if matches "${dest}" "${size}" "${md5}"; then
        log "present: ${dest}"
        return 0
    fi
    rm -f "${dest}"
    mkdir -p "$(dirname "${dest}")"
    [[ -f "${dest}.part" ]] && (( $(stat -c %s "${dest}.part") > size )) && rm -f "${dest}.part"
    log "downloading gs://${GCP_BUCKET}/${obj} ($(numfmt --to=iec "${size}"))"
    for attempt in 1 2 3; do
        (( attempt == 1 )) || GCS_TOKEN=$(gcs_token)
        curl -fL --retry 5 --retry-all-errors -sS -C - -H @<(auth_header) -o "${dest}.part" \
            "https://storage.googleapis.com/storage/v1/b/${GCP_BUCKET}/o/$(urlenc "${obj}")?alt=media" && break
        (( attempt < 3 )) || { echo "ERROR: download of ${obj} failed" >&2; return 1; }
    done
    matches "${dest}.part" "${size}" "${md5}" \
        || { echo "ERROR: ${dest}.part does not match gs://${GCP_BUCKET}/${obj} (size ${size}, md5 ${md5})" >&2; rm -f "${dest}.part"; return 1; }
    mv "${dest}.part" "${dest}"
}

GCS_TOKEN=$(gcs_token)
log "authenticated to gs://${GCP_BUCKET} as $(sa_field client_email)"

# ── 4. container: image -> sandbox ──────────────────────────────────────────
# Two markers: `unpacked` once the sandbox matches the image (the image is
# gone from then on), `complete` once the baked encoder has verified too. A
# rerun after a failed bake therefore re-bakes without re-downloading.
UNPACKED="${MOLAB_SANDBOX}/.igvf_molab/unpacked"
MARKER="${MOLAB_SANDBOX}/.igvf_molab/complete"
# `complete` alone is a sandbox from before `unpacked` existed; it was verified.
if [[ -f "${UNPACKED}" || -f "${MARKER}" ]]; then
    log "sandbox present and matches its image: ${MOLAB_SANDBOX} (image not needed)"
else
    # Whatever is here without the marker is a killed unpack: start clean,
    # before the download, so the two never add up on disk.
    rm -rf "${MOLAB_SANDBOX}"
    mkdir -p "$(dirname "${MOLAB_SIF}")"
    gcs_fetch "${SIF_OBJECT}" "${MOLAB_SIF}"

    # The squashfs partition's byte offset inside the SIF, from its header.
    offset=$(apptainer sif list "${MOLAB_SIF}" | awk -F'|' '/Squashfs/ {split($4,a,"-"); gsub(/ /,"",a[1]); print a[1]}')
    [[ -n "${offset}" ]] || { echo "ERROR: could not find the squashfs offset in ${MOLAB_SIF}" >&2; exit 1; }

    # Every entry in the image, less its root and the device nodes gVisor
    # cannot create, must exist in the sandbox. Counted BEFORE extraction, so
    # the image can be deleted the moment unsquashfs returns. Streamed.
    want=$(unsquashfs -o "${offset}" -lls "${MOLAB_SIF}" 2>/dev/null \
        | awk '/^[-dlps][rwxsStT-]{9}/ {n++} END {print n - 1}')
    [[ "${want}" -gt 0 ]] || { echo "ERROR: could not list ${MOLAB_SIF}" >&2; exit 1; }

    log "unpacking container to ${MOLAB_SANDBOX} (squashfs at offset ${offset}, ${want} entries)"
    # -no-exit-code: gVisor forbids mknod, so device nodes fail and
    # unsquashfs would exit 2 on an otherwise complete extraction.
    # -no-progress: the bar redraws ~400k times, which buries the log and is
    # a lot to push through marimo's web terminal.
    unsquashfs -no-exit-code -ignore-errors -no-progress -p "${MOLAB_CPUS}" \
        -d "${MOLAB_SANDBOX}" -o "${offset}" "${MOLAB_SIF}"
    log "removing ${MOLAB_SIF} (md5-checked and listed; not read again)"
    rm -f "${MOLAB_SIF}" "${MOLAB_SIF}.part"

    have=$(find "${MOLAB_SANDBOX}" -mindepth 1 -not -path "${MOLAB_SANDBOX}/.igvf_molab*" | wc -l)
    if [[ "${want}" != "${have}" ]]; then
        echo "ERROR: sandbox has ${have} entries, image has ${want}; removing it, rerun to re-download" >&2
        rm -rf "${MOLAB_SANDBOX}"
        exit 1
    fi
    mkdir -p "$(dirname "${UNPACKED}")"
    date -Is > "${UNPACKED}"
    log "sandbox matches image: ${have} entries"
fi

# ── 5. bake the low-memory one-hot encoder into the sandbox ─────────────────
utils_dir="${MOLAB_SANDBOX}/opt/chrombpnet/chrombpnet/training/utils"
[[ -f "${utils_dir}/one_hot.py" ]] || { echo "ERROR: ${utils_dir}/one_hot.py not found" >&2; exit 1; }
[[ -f "${utils_dir}/one_hot.py.upstream" ]] || cp -p "${utils_dir}/one_hot.py" "${utils_dir}/one_hot.py.upstream"
cp "${REPO_ROOT}/lib/python/utils/onehot.py" "${utils_dir}/igvf_onehot.py"
{
    cat "${utils_dir}/one_hot.py.upstream"
    echo
    echo "# --- igvf_tf_chrombpnet: low-memory encoder, baked in by workflows/molab/setup_molab.sh ---"
    echo "_upstream_dna_to_one_hot = dna_to_one_hot"
    echo "from chrombpnet.training.utils.igvf_onehot import dna_to_one_hot  # noqa: E402,F401"
} > "${utils_dir}/one_hot.py"
rm -f "${utils_dir}"/__pycache__/one_hot.*.pyc "${utils_dir}"/__pycache__/igvf_onehot.*.pyc

log "verifying the baked encoder against upstream inside the container"
# --cleanenv: the check needs nothing from this shell, least of all the key.
# 5000 x 2114 crosses the 4096-sequence chunk boundary at ~0.2 GB peak.
if ! apptainer exec --cleanenv "${MOLAB_SANDBOX}" python3 - <<'PY'
import numpy as np
from chrombpnet.training.utils import one_hot

new, old = one_hot.dna_to_one_hot, one_hot._upstream_dna_to_one_hot
assert getattr(new, "_igvf_low_memory", False), "baked encoder is not the igvf one"
probe = ["ACGTACGT", "acgtACGT", "NNNNACGT", "ACGTNRYK", "TTTTTTTT"]
assert np.array_equal(old(probe), new(probe)), "probe differs"
rng = np.random.default_rng(0)
codes = rng.choice(np.frombuffer(b"ACGTACGTACGTacgtNNRYKn", dtype=np.uint8), size=(5000, 2114))
seqs = [row.tobytes().decode("ascii") for row in codes]
a, b = old(seqs), new(seqs)
assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), "random set differs"
print("baked encoder byte-identical to upstream on %d x %d" % a.shape[:2])
PY
then
    cp -p "${utils_dir}/one_hot.py.upstream" "${utils_dir}/one_hot.py"
    rm -f "${utils_dir}/igvf_onehot.py"
    echo "ERROR: baked encoder failed verification; upstream one_hot.py restored" >&2
    exit 1
fi

# ── 6. keep only what is needed ─────────────────────────────────────────────
[[ -f "${MARKER}" ]] || date -Is > "${MARKER}"
# The image went right after the unpack; this only catches a stray one.
rm -f "${MOLAB_SIF}" "${MOLAB_SIF}.part"
log "sandbox: $(du -sh "${MOLAB_SANDBOX}" | cut -f1)"

mkdir -p "${MOLAB_CUDA_CACHE}" "${MOLAB_MPLCONFIG}"

# ── 7. pixi qc environment (stands in for envs/preprocess.yml) ───────────────
if ! command -v pixi >/dev/null 2>&1; then
    log "installing pixi"
    curl -fsSL https://pixi.sh/install.sh | bash >/dev/null
    ln -sf "${HOME}/.pixi/bin/pixi" /usr/local/bin/pixi
fi
log "pixi: $(pixi --version)"
log "installing the pixi qc environment"
# Every pixi call runs from REPO_ROOT: pixi resolves a manifest from the cwd
# upwards even for `clean cache`, and /marimo (the usual cwd here) holds
# marimo's own pyproject.toml with no [tool.pixi], which pixi refuses.
# The env is hardlinked out of the package cache, so clearing the cache frees
# the downloaded archives (~1.5 GB) without touching the installed env.
(cd "${REPO_ROOT}" && pixi install -e qc && pixi clean cache --yes >/dev/null)

# ── 8. d0 test data ──────────────────────────────────────────────────────────
log "fetching test data into ${MOLAB_DATA_DIR}"
# fd 3, not stdin: gcs_fetch runs commands that read stdin.
while read -r obj <&3; do
    gcs_fetch "${obj}" "${MOLAB_DATA_DIR}/${obj#"${TEST_DATA_PREFIX}"}"
done 3< <(gcs_list "${TEST_DATA_PREFIX}")

# ── 9. shared references ─────────────────────────────────────────────────────
if [[ "${SKIP_REFERENCES}" == "1" ]]; then
    log "skipping references (--skip-references)"
else
    log "fetching references into ${REFERENCE_ROOT}"
    (cd "${REPO_ROOT}" && pixi run -e qc python src/cli.py download-references --path "${DATASET_CONFIG}")
fi

log "setup complete."
echo
echo "Next:"
echo "  source workflows/molab/env.sh"
echo "  bash workflows/molab/run_step.sh 00.0.prepare_signal.sh"
