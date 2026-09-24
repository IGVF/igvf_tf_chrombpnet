#!/bin/bash
# env.sh
# Purpose: Every environment variable the pipeline needs on a molab box, in one
#   place. Source it before running any step:  source workflows/molab/env.sh
#
# Why so little is set here: every step enters its own environment through
# activate_env (lib/bash/common.sh), and every default there is a pixi
# environment -- this repo's pixi.toml for preprocess / finemo /
# motif-compendium, and a pinned chrombpnet 2.x checkout for chrombpnet. So
# CHROMBPNET_ENV, PREPROCESS_ENV, FINEMO_ENV and MOTIF_COMPENDIUM_ENV are
# deliberately left UNSET: common.sh's pixi defaults are what molab runs. Only
# the box-specific values are set below: where the chrombpnet checkout lives,
# where pixi and the caches live (under /marimo, which survives a new
# session; $HOME does not), the real CPU count, and the data locations.
#
# BOOTSTRAP_PYTHON is left unset as well. The container era pointed it at the
# qc env so config.sh had a python inside the container; the side effect was
# that every step's run metadata recorded the qc env's packages instead of the
# step's own. common.sh's bootstrap_python now finds one on PATH.
#
# Input:  an optional .env (see .env.example)
# Output: exported variables, and the molab_config function
# Usage:  source workflows/molab/env.sh
# Prerequisites: run setup_molab.sh once first.

# Repo root, from this file's location (this is sourced, never sbatch'd, so
# BASH_SOURCE is reliable here -- unlike in the SLURM steps).
MOLAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${MOLAB_DIR}/../.." && pwd)}"

# ── Local overrides and credentials ──────────────────────────────────────────
# A .env holds anything machine- or account-specific: GITHUB_TOKEN,
# GCP_BUCKET, GCP_SA_JSON, MOLAB_CPUS, DATASET. Sourced FIRST so every default
# below can be overridden from it. See .env.example. The first that exists
# wins: ${MOLAB_ENV_FILE}, workflows/molab/.env (gitignored), then the .env one
# level above the checkout (/marimo/.env on molab), which lives outside git
# entirely.
for _molab_env in "${MOLAB_ENV_FILE:-}" "${MOLAB_DIR}/.env" "${REPO_ROOT}/../.env"; do
    if [[ -n "${_molab_env}" && -f "${_molab_env}" ]]; then
        set -a
        # shellcheck disable=SC1090  # untracked, machine-local
        source "${_molab_env}"
        set +a
        break
    fi
done
unset _molab_env

# A GitHub token in the environment is enough to push: this credential helper
# feeds it to git without writing it to disk or into .git/config. Configured
# per-repo, and only when a token is actually present.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    git -C "${REPO_ROOT}" config credential.helper \
        '!f() { echo username=x-access-token; echo "password=${GITHUB_TOKEN}"; }; f'
fi

# ── Which dataset ────────────────────────────────────────────────────────────
export DATASET="${DATASET:-d0}"

# The d0 config lives WITH its inputs, not in config/<dataset>/, so the test
# dataset is self-contained: setup_molab.sh mirrors
# gs://${GCP_BUCKET}/chrombpnet/test_data_d0/{config,inputs}/ to
# /marimo/data/test_data_d0/, fragments, peaks and the parameters that describe
# them together. config/README.md supports this -- DATASET_CONFIG
# points at a config anywhere. Note the trade-off: a config outside the
# checkout is not version-controlled, which is fine for a local test dataset
# and would NOT be for a real one (see config/HEP3B/config.yaml, tracked).
export DATASET_CONFIG="${DATASET_CONFIG:-/marimo/data/test_data_d0/config/config.yaml}"

# ── Software environments ────────────────────────────────────────────────────
# chrombpnet 2.x comes from its own checkout, installed by setup_molab.sh from
# that repo's lock file. NOT /marimo/chrombpnet: that is a development
# checkout, and setup moves this one's HEAD to the pinned commit. The commit
# itself, CHROMBPNET_REV, is defined once in lib/bash/common.sh; set it in the
# .env only to try another one on purpose (activate_env warns on a mismatch).
export CHROMBPNET_REPO="${CHROMBPNET_REPO:-/marimo/chrombpnet-igvf}"
# cuda13 needs NVIDIA driver >= 580; cuda12 is chrombpnet's fallback.
export CHROMBPNET_PIXI_ENV="${CHROMBPNET_PIXI_ENV:-cuda13}"

# No conda on the box. activate_env only reads CONDA_INIT for an environment
# given as a conda prefix, which none of the defaults is; empty keeps a stray
# prefix from reaching for a conda that does not exist.
export CONDA_INIT=""

# A shell that sourced the container-era env.sh still exports its values, and
# common.sh would honour them: the PREPROCESS_ENV directory would be taken for
# a conda prefix and, with CONDA_INIT empty, silently not entered. Drop
# exactly those values; anything else set on purpose is kept.
[[ "${CHROMBPNET_ENV:-}" != apptainer:* ]] || unset CHROMBPNET_ENV
[[ "${FINEMO_ENV:-}" != unconfigured:* ]] || unset FINEMO_ENV
[[ "${MOTIF_COMPENDIUM_ENV:-}" != unconfigured:* ]] || unset MOTIF_COMPENDIUM_ENV
[[ "${PREPROCESS_ENV:-}" != "${REPO_ROOT}/.pixi/envs/preprocess" ]] || unset PREPROCESS_ENV
[[ "${BOOTSTRAP_PYTHON:-}" != "${REPO_ROOT}/.pixi/envs/qc/bin/python" ]] || unset BOOTSTRAP_PYTHON

# Pinned pixi, caches and anything else setup writes outside the checkouts.
# Under /marimo so a new session keeps them, next to the checkouts' .pixi/
# directories so pixi can hardlink packages out of its cache instead of
# copying them. One PIXI_CACHE_DIR serves both checkouts (this repo's
# environments and chrombpnet's), so a package they share is stored once.
export MOLAB_SCRATCH="${MOLAB_SCRATCH:-/marimo/igvf-scratch}"
# setup_molab.sh puts a pinned pixi here when the box has none, or an older one.
if [[ -x "${MOLAB_SCRATCH}/bin/pixi" && ":${PATH}:" != *":${MOLAB_SCRATCH}/bin:"* ]]; then
    export PATH="${MOLAB_SCRATCH}/bin:${PATH}"
fi
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${MOLAB_SCRATCH}/cache/rattler}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${MOLAB_SCRATCH}/cache/uv}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${MOLAB_SCRATCH}/cache/jax}"
export KERAS_HOME="${KERAS_HOME:-${MOLAB_SCRATCH}/cache/keras}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${MOLAB_SCRATCH}/cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${MOLAB_SCRATCH}/cache/numba}"

# ── GPU memory ───────────────────────────────────────────────────────────────
# JAX allocates on demand instead of grabbing most of the card at start-up.
# chrombpnet sets the same default when it is imported; exporting it covers
# everything that imports jax first.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# XLA_PYTHON_CLIENT_MEM_FRACTION is deliberately NOT defaulted. Unset, a step
# may use the whole card, which is what a box running only this pipeline
# wants. Set it (e.g. 0.25 in the .env) when the GPU is shared with other
# jobs: it caps this process at that fraction of the card's TOTAL memory, and
# DeepSHAP sizes its batches from what is free.

# ── Data locations ───────────────────────────────────────────────────────────
export REFERENCE_ROOT="${REFERENCE_ROOT:-/marimo/data/references}"
export DATASET_ROOT="${DATASET_ROOT:-/marimo/data}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# molab_config <expr>... — print each shell expression, one per line, as
# lib/bash/config.sh evaluates it for the selected dataset, e.g.
#   molab_config '${metadata_dir}' '${#folds[@]}'
# config.sh runs in a child bash, the same derivation every step makes, so the
# launchers' paths cannot drift from the steps' (a sed over the YAML export
# used to leave an output_dir of "${DATASET_ROOT}/..." unexpanded). Fails,
# printing nothing, when the config does not resolve.
molab_config() {
    # shellcheck disable=SC2016  # expanded by the child bash, after config.sh
    bash -c 'REPO_ROOT="$1"; source "$1/lib/bash/config.sh" >/dev/null 2>&1 || exit 1; shift
             for _e in "$@"; do eval "printf \"%s\\n\" \"${_e}\""; done' _ "${REPO_ROOT}" "$@"
}

# Launcher logs. Default to the pipeline's OWN log_dir (lib/bash/config.sh:
# log_dir="${results_path}/logs") rather than inventing a second location.
# Override to put them elsewhere.
if [[ -z "${MOLAB_LOG_DIR:-}" ]]; then
    # shellcheck disable=SC2016  # expanded by molab_config's child bash
    MOLAB_LOG_DIR="$(molab_config '${log_dir}' 2>/dev/null)" || MOLAB_LOG_DIR=""
    export MOLAB_LOG_DIR="${MOLAB_LOG_DIR:-${REPO_ROOT}/results/logs}"
fi

# ── Real resources ───────────────────────────────────────────────────────────
# The box reports the host's CPUs and RAM (20+ cores, 160 GB), not the slice we
# actually get. Left unpinned, OpenMP, BLAS and numba size their thread pools
# to the phantom count and thrash; run_step.sh caps them all at this. Set it to
# the REAL core count.
export MOLAB_CPUS="${MOLAB_CPUS:-4}"
