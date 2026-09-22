#!/bin/bash
# setup_molab.sh
# Purpose: Prepare a molab (marimo cloud) box to run this pipeline end to end.
#   Installs Apptainer and its userspace dependencies, fetches and unpacks the
#   chrombpnet container, installs pixi's `qc` environment, and downloads the
#   shared references. Idempotent: everything already present is verified and
#   skipped.
#
# Why a container at all: the cluster runs chrombpnet from `envs/chrombpnet.yml`,
# which pins tensorflow 2.8 / numpy 1.23 against CUDA 11 wheels. Those wheels do
# not exist for this hardware, so molab uses a prebuilt image instead. pixi's
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
# Input:  none (everything is fetched)
# Output: ${MOLAB_SIF}, ${MOLAB_SANDBOX}, the pixi `qc` env, ${REFERENCE_ROOT}
# Usage:  bash workflows/molab/setup_molab.sh
#         bash workflows/molab/setup_molab.sh --skip-references
# Prerequisites: root (for apt), ~40 GB free disk, outbound HTTPS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workflows/molab/env.sh
source "${SCRIPT_DIR}/env.sh"

SIF_URL="https://storage.googleapis.com/broad-buenrostro-pipeline-genome-annotations/chrombpnet/containers/chrombpnet.sif"
SIF_BYTES=8389165056
SIF_MD5_B64="L8UAYNYWLlTu6MjUXkCwvw=="
APPTAINER_VERSION="1.5.3"

SKIP_REFERENCES=0
[[ "${1:-}" == "--skip-references" ]] && SKIP_REFERENCES=1

log() { echo "[$(date '+%F %T')] $*"; }

# ── 1. locale ─────────────────────────────────────────────────────────────────
# The image sets LC_ALL=en_US.UTF-8 but ships no generated locale, so every
# bash invocation prints a setlocale warning that buries real output.
if ! locale -a 2>/dev/null | grep -qi '^en_US\.utf-\?8$'; then
    log "generating en_US.UTF-8 locale"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq locales >/dev/null
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
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        squashfs-tools squashfuse fuse2fs fuse3 uidmap fuse-overlayfs \
        libseccomp2 cryptsetup-bin ca-certificates curl >/dev/null
    deb="$(mktemp -d)/apptainer.deb"
    curl -fsSL -o "${deb}" \
        "https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}-trixie%2B_amd64.deb"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${deb}" >/dev/null
    rm -f "${deb}"
fi
log "apptainer: $(apptainer --version)"

# Overlay is unusable under gVisor; underlay (plain bind mounts) is not.
if grep -q '^enable overlay = yes' /etc/apptainer/apptainer.conf 2>/dev/null; then
    log "apptainer.conf: disabling overlay, enabling underlay (gVisor has no working overlayfs)"
    cp /etc/apptainer/apptainer.conf /etc/apptainer/apptainer.conf.orig
    sed -i 's/^enable overlay = .*/enable overlay = no/'  /etc/apptainer/apptainer.conf
    sed -i 's/^enable underlay = .*/enable underlay = yes/' /etc/apptainer/apptainer.conf
fi

# ── 3. container image ───────────────────────────────────────────────────────
mkdir -p "$(dirname "${MOLAB_SIF}")"
if [[ -s "${MOLAB_SIF}" ]] && [[ "$(stat -c %s "${MOLAB_SIF}")" == "${SIF_BYTES}" ]]; then
    log "container image present ($(numfmt --to=iec "${SIF_BYTES}"))"
else
    log "downloading container image ($(numfmt --to=iec "${SIF_BYTES}"))"
    curl -fL --retry 5 --retry-all-errors -C - -o "${MOLAB_SIF}" "${SIF_URL}"
fi

got_size=$(stat -c %s "${MOLAB_SIF}")
[[ "${got_size}" == "${SIF_BYTES}" ]] || { echo "ERROR: ${MOLAB_SIF} is ${got_size} bytes, expected ${SIF_BYTES}" >&2; exit 1; }
log "verifying md5"
got_md5=$(openssl dgst -md5 -binary "${MOLAB_SIF}" | base64)
[[ "${got_md5}" == "${SIF_MD5_B64}" ]] || { echo "ERROR: md5 ${got_md5} != ${SIF_MD5_B64}" >&2; exit 1; }
log "md5 verified: ${got_md5}"

# ── 4. unpack to a sandbox directory ─────────────────────────────────────────
if [[ -x "${MOLAB_SANDBOX}/bin/sh" ]]; then
    log "sandbox present: ${MOLAB_SANDBOX}"
else
    # The squashfs partition's byte offset inside the SIF, from its header.
    offset=$(apptainer sif list "${MOLAB_SIF}" | awk -F'|' '/Squashfs/ {split($4,a,"-"); gsub(/ /,"",a[1]); print a[1]}')
    [[ -n "${offset}" ]] || { echo "ERROR: could not find the squashfs offset in ${MOLAB_SIF}" >&2; exit 1; }
    log "unpacking container to ${MOLAB_SANDBOX} (squashfs at offset ${offset})"
    rm -rf "${MOLAB_SANDBOX}"
    # -no-exit-code: gVisor forbids mknod, so device nodes fail and unsquashfs
    # would exit 2 on an otherwise complete extraction.
    unsquashfs -no-exit-code -ignore-errors -p "${MOLAB_CPUS}" \
        -d "${MOLAB_SANDBOX}" -o "${offset}" "${MOLAB_SIF}"
    [[ -x "${MOLAB_SANDBOX}/bin/sh" ]] || { echo "ERROR: extraction produced no ${MOLAB_SANDBOX}/bin/sh" >&2; exit 1; }
fi
log "sandbox: $(du -sh "${MOLAB_SANDBOX}" | cut -f1)"

mkdir -p "${MOLAB_CUDA_CACHE}" "${MOLAB_MPLCONFIG}"

# ── 5. pixi qc environment (stands in for envs/preprocess.yml) ───────────────
if ! command -v pixi >/dev/null 2>&1; then
    log "installing pixi"
    curl -fsSL https://pixi.sh/install.sh | bash >/dev/null
    ln -sf "${HOME}/.pixi/bin/pixi" /usr/local/bin/pixi
fi
log "pixi: $(pixi --version)"
log "installing the pixi qc environment"
(cd "${REPO_ROOT}" && pixi install -e qc)

# ── 6. shared references ─────────────────────────────────────────────────────
if [[ "${SKIP_REFERENCES}" == "1" ]]; then
    log "skipping references (--skip-references)"
else
    log "fetching references into ${REFERENCE_ROOT}"
    (cd "${REPO_ROOT}" && pixi run -e qc python src/cli.py download-references --dataset "${DATASET}")
fi

log "setup complete."
echo
echo "Next:"
echo "  source workflows/molab/env.sh"
echo "  bash workflows/molab/run_step.sh 00.0.prepare_signal.sh"
