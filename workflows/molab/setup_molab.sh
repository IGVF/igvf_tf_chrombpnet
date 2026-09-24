#!/bin/bash
# setup_molab.sh
# Purpose: Prepare a molab (marimo cloud) box to run this pipeline end to end.
#   Installs the few system tools it needs (git, curl, openssl), a pinned pixi,
#   the chrombpnet 2.x checkout at the pinned commit with its pixi environment,
#   and this repo's pixi environments; checks the chrombpnet tools and the GPU;
#   then downloads the d0 test data and the shared references. Idempotent:
#   everything already present is checked and skipped, so re-run it after
#   every new session.
#
# Why pixi and nothing else: chrombpnet 2.x (Keras 3 on JAX) brings its CUDA
# libraries as pip wheels inside its own environment, so the box needs no
# system CUDA, no conda and no container -- only an NVIDIA driver new enough
# for the wheels (>= 580 for cuda13). Every environment is installed with
# --locked, from a lock file: chrombpnet's own pixi.lock, the one the port was
# validated with, and this repo's pixi.lock for preprocess / qc / finemo /
# motif-compendium. A lock file that no longer matches its manifest stops the
# install instead of being re-solved on the box.
#
# Why a separate chrombpnet checkout (${CHROMBPNET_REPO}, default
# /marimo/chrombpnet-igvf): setup moves its HEAD to CHROMBPNET_REV, the commit
# lib/bash/common.sh pins, with `git checkout --detach`. It refuses to do that
# to a checkout that is on a branch or has local changes, so it cannot move a
# development checkout (such as /marimo/chrombpnet) by accident.
#
# Why pixi is pinned: a box image may ship no pixi, or an older one than the
# lock files were written with. setup then puts pixi ${PIXI_VERSION} in
# ${MOLAB_SCRATCH}/bin, which env.sh puts first on PATH -- under /marimo, so a
# new session keeps it, as it keeps the package caches beside it.
#
# Why the GPU check only warns: steps 00.0-02.0 need no GPU, and a recreated
# molab session can come back without one (no /dev/nvidia*, JAX quietly on the
# CPU). Saying so here, loudly, beats a GPU step finding out hours in.
#
# Why output goes to ${MOLAB_SETUP_LOG}: a session can end mid-setup, and the
# log in /marimo then shows how far it got. Each line carries the space used
# on /, because the box shows no disk quota (df reports 8.0E) and that is the
# only disk counter that moves.
#
# Input:  none (everything is fetched)
# Output: pixi in ${MOLAB_SCRATCH}/bin (when needed), caches under
#         ${MOLAB_SCRATCH}/cache, ${CHROMBPNET_REPO} and its .pixi/envs/,
#         this repo's .pixi/envs/, ${MOLAB_DATA_DIR}, ${REFERENCE_ROOT}
# Usage:  bash workflows/molab/setup_molab.sh
#         bash workflows/molab/setup_molab.sh --skip-references
# Exit:   0 ready (a missing GPU is a warning only); non-zero when something
#         failed: an install, a download, or the chrombpnet tool check.
# Prerequisites: root (for apt), outbound HTTPS, and GCP_BUCKET + GCP_SA_JSON
#   (the service-account key as inline JSON) in the .env env.sh sources.

# shellcheck disable=SC2218  # 0.11.0 false positive (see CLAUDE.md): log() is defined before use

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workflows/molab/env.sh
source "${SCRIPT_DIR}/env.sh"

TEST_DATA_PREFIX="chrombpnet/test_data_d0/"
MOLAB_DATA_DIR="${MOLAB_DATA_DIR:-/marimo/data/test_data_d0}"
PIXI_VERSION="${PIXI_VERSION:-0.81.0}"
CHROMBPNET_URL="${CHROMBPNET_URL:-https://github.com/NNFC-GMD/chrombpnet}"
# This repo's environments (pixi.toml). finemo-cu126 is left out: pixi.toml
# keeps it for drivers below 580, and its torch cannot drive Blackwell.
LOCAL_ENVS=( preprocess qc finemo motif-compendium )
APT=(env DEBIAN_FRONTEND=noninteractive apt-get -y -qq
     -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

SKIP_REFERENCES=0
for a in "$@"; do
    case "$a" in
        --skip-references) SKIP_REFERENCES=1 ;;
        -h|--help) sed -n '2,49p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown option $a" >&2; exit 1 ;;
    esac
done

MOLAB_SETUP_LOG="${MOLAB_SETUP_LOG:-/marimo/setup_molab.log}"
mkdir -p "$(dirname "${MOLAB_SETUP_LOG}")"
exec > >(tee -a "${MOLAB_SETUP_LOG}") 2>&1

log() { echo "[$(date '+%F %T')] [/ used: $(df -h --output=used / | tail -1 | tr -d ' ')] $*"; }
log "setup_molab.sh started (log: ${MOLAB_SETUP_LOG})"

[[ -n "${GCP_BUCKET:-}" ]]  || { echo "ERROR: GCP_BUCKET is not set (see workflows/molab/.env.example)" >&2; exit 1; }
[[ -n "${GCP_SA_JSON:-}" ]] || { echo "ERROR: GCP_SA_JSON is not set (see workflows/molab/.env.example)" >&2; exit 1; }

# Nothing from the notebook's shell may reach the installs or the checks
# below: its PYTHONPATH/venv would leak packages into every environment's
# python, and a CUDA on LD_LIBRARY_PATH would shadow JAX's own CUDA wheels.
# (activate_env drops the same variables for every step.)
unset PYTHONPATH PYTHONHOME PYTHONSAFEPATH VIRTUAL_ENV LD_LIBRARY_PATH
# The CUDA environments declare a __cuda virtual package. The override lets
# them install and run even when the session came back without its GPU;
# gpu-check below is what says whether the GPU actually works.
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.0}"

# CHROMBPNET_REV is defined once, in lib/bash/common.sh (or overridden from the
# .env). Read it from there rather than repeat the SHA; a child bash, so
# nothing else common.sh sets leaks into this script.
# shellcheck disable=SC2016  # expanded by the child bash
CHROMBPNET_REV="$(bash -c 'source "$1/lib/bash/common.sh" >/dev/null 2>&1; printf "%s" "${CHROMBPNET_REV:-}"' _ "${REPO_ROOT}")"
[[ "${CHROMBPNET_REV}" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "ERROR: could not read a full CHROMBPNET_REV from lib/bash/common.sh (got '${CHROMBPNET_REV}')" >&2; exit 1; }

# ── 0. apt ────────────────────────────────────────────────────────────────────
# The image ships with empty package lists, so any install fails without an
# update -- but a rerun that installs nothing does not need one. No `upgrade`:
# nothing here needs newer packages, and on a box whose every written byte
# counts against a hidden limit it is pure cost.
apt_updated=0
apt_install() {
    if [[ "${apt_updated}" == "0" ]]; then
        log "apt: update"
        "${APT[@]}" update
        apt_updated=1
    fi
    log "apt: install $*"
    "${APT[@]}" install "$@" >/dev/null
}

# ── 1. locale ─────────────────────────────────────────────────────────────────
# The image sets LC_ALL=en_US.UTF-8 but ships no generated locale, so every
# bash invocation prints a setlocale warning that buries real output.
if ! locale -a 2>/dev/null | grep -qi '^en_US\.utf-\?8$'; then
    log "generating en_US.UTF-8 locale"
    apt_install locales
    sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
    grep -q '^en_US.UTF-8' /etc/locale.gen || echo 'en_US.UTF-8 UTF-8' >> /etc/locale.gen
    locale-gen >/dev/null
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
fi

# ── 2. system tools ───────────────────────────────────────────────────────────
# curl + openssl + CA certificates for gcs.sh and the pixi download, git for
# the chrombpnet checkout. Everything else comes from pixi environments.
missing=()
for tool in curl openssl git; do
    command -v "${tool}" >/dev/null 2>&1 || missing+=( "${tool}" )
done
[[ -s /etc/ssl/certs/ca-certificates.crt ]] || missing+=( ca-certificates )
if [[ ${#missing[@]} -gt 0 ]]; then
    apt_install "${missing[@]}"
    "${APT[@]}" clean
fi

# ── 3. GCS access ─────────────────────────────────────────────────────────────
# The bucket is private and there is no gcloud on the box; gcs.sh mints an
# OAuth token from the service-account key with openssl (read-only here).
# Checked now, so a bad key fails in seconds rather than after the installs.
# shellcheck source=workflows/molab/gcs.sh
source "${SCRIPT_DIR}/gcs.sh"
GCS_TOKEN=$(gcs_token)
log "authenticated to gs://${GCP_BUCKET} as $(sa_field client_email)"

# ── 4. pixi ───────────────────────────────────────────────────────────────────
mkdir -p "${MOLAB_SCRATCH}/bin" "${PIXI_CACHE_DIR}" "${UV_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${KERAS_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
have_pixi=""
if command -v pixi >/dev/null 2>&1; then
    have_pixi="$(pixi --version 2>/dev/null | awk '{print $2}')" || have_pixi=""
fi
if [[ -z "${have_pixi}" \
      || "$(printf '%s\n' "${PIXI_VERSION}" "${have_pixi}" | sort -V | head -n1)" != "${PIXI_VERSION}" ]]; then
    log "pixi ${have_pixi:-not found}: installing ${PIXI_VERSION} into ${MOLAB_SCRATCH}/bin"
    curl -fsSL -o "${MOLAB_SCRATCH}/bin/pixi.tmp" \
        "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl"
    chmod 0755 "${MOLAB_SCRATCH}/bin/pixi.tmp"
    mv "${MOLAB_SCRATCH}/bin/pixi.tmp" "${MOLAB_SCRATCH}/bin/pixi"
    [[ ":${PATH}:" == *":${MOLAB_SCRATCH}/bin:"* ]] || export PATH="${MOLAB_SCRATCH}/bin:${PATH}"
fi
log "$(pixi --version) ($(command -v pixi))"

# ── 5. chrombpnet checkout at CHROMBPNET_REV ─────────────────────────────────
cloned=0
if [[ ! -e "${CHROMBPNET_REPO}" ]]; then
    log "cloning ${CHROMBPNET_URL} into ${CHROMBPNET_REPO}"
    git clone --quiet "${CHROMBPNET_URL}" "${CHROMBPNET_REPO}"
    cloned=1
elif [[ ! -e "${CHROMBPNET_REPO}/.git" ]]; then
    echo "ERROR: ${CHROMBPNET_REPO} exists but is not a git checkout; move it, or set CHROMBPNET_REPO" >&2
    exit 1
fi

head_rev="$(git -C "${CHROMBPNET_REPO}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${head_rev}" != "${CHROMBPNET_REV}" && "${cloned}" == "0" ]]; then
    # Only ever move a checkout this script made: detached and clean. A branch
    # or local edits mean someone works in it.
    if branch="$(git -C "${CHROMBPNET_REPO}" symbolic-ref -q --short HEAD)"; then
        echo "ERROR: ${CHROMBPNET_REPO} is on branch '${branch}', not at CHROMBPNET_REV=${CHROMBPNET_REV}." >&2
        echo "  setup only moves a detached checkout, so it will not touch a working copy." >&2
        echo "  Point CHROMBPNET_REPO at a dedicated checkout (default /marimo/chrombpnet-igvf)," >&2
        echo "  or detach this one yourself: git -C ${CHROMBPNET_REPO} checkout --detach ${CHROMBPNET_REV}" >&2
        exit 1
    fi
    if [[ -n "$(git -C "${CHROMBPNET_REPO}" status --porcelain --untracked-files=no)" ]]; then
        echo "ERROR: ${CHROMBPNET_REPO} has local changes; not moving it to ${CHROMBPNET_REV}." >&2
        git -C "${CHROMBPNET_REPO}" status --short --untracked-files=no >&2
        exit 1
    fi
fi

if [[ "${cloned}" == "0" ]]; then
    log "fetching ${CHROMBPNET_REPO}"
    git -C "${CHROMBPNET_REPO}" fetch --quiet origin \
        || log "WARNING: git fetch failed; carrying on if ${CHROMBPNET_REV:0:7} is already here"
fi
if ! git -C "${CHROMBPNET_REPO}" cat-file -e "${CHROMBPNET_REV}^{commit}" 2>/dev/null; then
    # Not on any branch head (a rewritten branch, say): ask for it by SHA.
    git -C "${CHROMBPNET_REPO}" fetch --quiet origin "${CHROMBPNET_REV}" \
        || { echo "ERROR: commit ${CHROMBPNET_REV} not found in ${CHROMBPNET_URL}" >&2; exit 1; }
fi
if [[ "${head_rev}" != "${CHROMBPNET_REV}" ]]; then
    log "checking out ${CHROMBPNET_REV:0:7} (was ${head_rev:0:7})"
    git -C "${CHROMBPNET_REPO}" checkout --quiet --detach "${CHROMBPNET_REV}"
fi
log "chrombpnet checkout: ${CHROMBPNET_REPO} @ $(git -C "${CHROMBPNET_REPO}" rev-parse --short HEAD)"

# ── 6. pixi environments ──────────────────────────────────────────────────────
# Every call names its manifest, so none depends on the cwd -- /marimo, the
# usual one here, holds marimo's own pyproject.toml with no [tool.pixi], which
# pixi refuses. --locked: install exactly the lock file, never re-solve.
CHROMBPNET_MANIFEST="${CHROMBPNET_REPO}/pyproject.toml"
log "installing chrombpnet 2.x: pixi ${CHROMBPNET_PIXI_ENV} from ${CHROMBPNET_MANIFEST}"
pixi install --locked --manifest-path "${CHROMBPNET_MANIFEST}" -e "${CHROMBPNET_PIXI_ENV}" \
    || { echo "ERROR: pixi install of ${CHROMBPNET_PIXI_ENV} failed (see above)" >&2; exit 1; }
for env in "${LOCAL_ENVS[@]}"; do
    log "installing pixi ${env} from ${REPO_ROOT}/pixi.toml"
    pixi install --locked --manifest-path "${REPO_ROOT}/pixi.toml" -e "${env}" \
        || { echo "ERROR: pixi install of ${env} failed (see above)" >&2; exit 1; }
done

# ── 7. checks ─────────────────────────────────────────────────────────────────
# The command-line tools the chrombpnet steps shell out to, all from the
# chrombpnet environment: bedtools (01.0 negatives), samtools, MEME's tomtom
# (modisco report-simple) and modisco itself (03.3, 04.5, 08.0).
status=0
log "chrombpnet env: versions"
pixi run --frozen --manifest-path "${CHROMBPNET_MANIFEST}" -e "${CHROMBPNET_PIXI_ENV}" versions || status=1
log "chrombpnet env: tools"
# shellcheck disable=SC2016  # expanded by the inner bash
pixi run --frozen --manifest-path "${CHROMBPNET_MANIFEST}" -e "${CHROMBPNET_PIXI_ENV}" bash -c \
    'set -e; bedtools --version; samtools --version | head -n1; printf "tomtom "; tomtom -version; modisco --help >/dev/null; echo "modisco ok"' \
    || status=1
[[ "${status}" == "0" ]] || log "ERROR: the chrombpnet environment is missing a tool (see above)"

if gpu="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null)" \
        && [[ -n "${gpu}" ]]; then
    log "GPU (name, compute capability, driver): ${gpu}"
    driver="${gpu%%$'\n'*}"; driver="${driver##*, }"; driver_major="${driver%%.*}"
    if [[ "${CHROMBPNET_PIXI_ENV}" == cuda13* && "${driver_major}" =~ ^[0-9]+$ && "${driver_major}" -lt 580 ]]; then
        log "WARNING: driver ${driver} is older than 580, which CUDA 13 needs: set CHROMBPNET_PIXI_ENV=cuda12"
    fi
else
    log "WARNING: nvidia-smi lists no GPU."
fi
log "chrombpnet env: gpu-check"
if pixi run --frozen --manifest-path "${CHROMBPNET_MANIFEST}" -e "${CHROMBPNET_PIXI_ENV}" gpu-check; then
    log "GPU: JAX in the chrombpnet environment runs on it"
else
    log "WARNING: ================================================================"
    log "WARNING: JAX in the chrombpnet environment sees NO GPU. The GPU steps"
    log "WARNING: (#SBATCH --gres=gpu:1) would stop, or run on the CPU for days."
    log "WARNING: Restart the molab session on a GPU machine. Steps 00.0-02.0"
    log "WARNING: need no GPU and can run meanwhile."
    log "WARNING: ================================================================"
fi

# ── 8. d0 test data ───────────────────────────────────────────────────────────
log "fetching test data into ${MOLAB_DATA_DIR}"
# A token lasts an hour, which the installs above can outlast; gcs_list does
# not retry, so mint a fresh one first.
GCS_TOKEN=$(gcs_token)
# Only config/ and inputs/: sync_to_gcs.sh writes results/, repo.bundle and
# the manifest under the same prefix, and none of that is setup's to fetch.
# fd 3, not stdin: gcs_fetch runs commands that read stdin.
for sub in config/ inputs/; do
    while read -r obj <&3; do
        gcs_fetch "${obj}" "${MOLAB_DATA_DIR}/${obj#"${TEST_DATA_PREFIX}"}"
    done 3< <(gcs_list "${TEST_DATA_PREFIX}${sub}")
done

# ── 9. shared references ──────────────────────────────────────────────────────
if [[ "${SKIP_REFERENCES}" == "1" ]]; then
    log "skipping references (--skip-references)"
else
    # reference_root comes from the dataset config. DATASET_CONFIG may name a
    # copy that does not exist yet -- the one a 2.x run makes from the d0
    # config fetched just above -- so fall back to that fetched config.
    refs_cfg="${DATASET_CONFIG}"
    if [[ ! -f "${refs_cfg}" ]]; then
        refs_cfg="${MOLAB_DATA_DIR}/config/config.yaml"
        log "no ${DATASET_CONFIG} yet; reading reference_root from ${refs_cfg}"
    fi
    log "fetching references into ${REFERENCE_ROOT}"
    pixi run --frozen --manifest-path "${REPO_ROOT}/pixi.toml" -e preprocess \
        python "${REPO_ROOT}/src/cli.py" download-references --path "${refs_cfg}"
fi

# ── 10. disk ──────────────────────────────────────────────────────────────────
# du counts a hardlinked file once, under the first argument that reaches it,
# so the caches listed after the environments show only what is NOT linked
# into one (mostly downloaded archives; `pixi clean cache` reclaims them).
log "disk used by the environments and caches:"
du -sh "${CHROMBPNET_REPO}/.pixi/envs/${CHROMBPNET_PIXI_ENV}" \
    "${LOCAL_ENVS[@]/#/${REPO_ROOT}/.pixi/envs/}" \
    "${PIXI_CACHE_DIR}" "${UV_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" 2>/dev/null || true
if [[ -d /marimo/containers ]]; then
    log "NOTE: /marimo/containers (the chrombpnet 1.x container) is no longer used; rm -rf it to free the space."
fi

if [[ "${status}" != "0" ]]; then
    log "setup finished WITH ERRORS (see above)."
    exit 1
fi
log "setup complete."
echo
echo "Next:"
echo "  source workflows/molab/env.sh"
echo "  bash workflows/molab/run_step.sh 00.0.prepare_signal.sh"
