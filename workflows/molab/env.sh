#!/bin/bash
# env.sh
# Purpose: Every environment variable the pipeline needs on a molab box, in one
#   place. Source it before running any step:  source workflows/molab/env.sh
#
# Why these values: lib/bash/common.sh defaults each of the four conda
# environments to a path under another user's Sherlock home. None of those
# exist here and none should be referenced, so every one is overridden below.
# On molab there is no conda at all:
#
#   * steps 00.0 / 00.1 / 02.0 run in the pixi `preprocess` environment, which
#     reproduces envs/preprocess.yml (see pixi.toml);
#   * steps 01.0 / 03.x / 04.x run inside the chrombpnet container, which
#     already has chrombpnet, tensorflow and bedtools on PATH.
#
# In both cases the tools are on PATH before the step starts, so CONDA_INIT is
# set EMPTY, which makes activate_env a no-op (lib/bash/common.sh). Empty is
# opt-in and meaningful; unset would fall back to the Sherlock default and the
# step would die with "conda init script not found".
#
# Input:  none
# Output: exported variables only
# Usage:  source workflows/molab/env.sh
# Prerequisites: run setup_molab.sh once first.

# Repo root, from this file's location (this is sourced, never sbatch'd, so
# BASH_SOURCE is reliable here -- unlike in the SLURM steps).
MOLAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${MOLAB_DIR}/../.." && pwd)}"

# ── Local overrides and credentials ──────────────────────────────────────────
# workflows/molab/.env is gitignored and holds anything machine- or
# account-specific: GITHUB_TOKEN, MOLAB_CPUS, DATASET. Sourced FIRST so every
# default below can be overridden from it. See .env.example.
if [[ -f "${MOLAB_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091  # untracked, machine-local
    source "${MOLAB_DIR}/.env"
    set +a
fi

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
# dataset is self-contained: fragments, peaks and the parameters that describe
# them in one directory. config/README.md supports this -- DATASET_CONFIG
# points at a config anywhere. Note the trade-off: a config outside the
# checkout is not version-controlled, which is fine for a local test dataset
# and would NOT be for a real one (see config/HEP3B/config.yaml, tracked).
export DATASET_CONFIG="${DATASET_CONFIG:-/marimo/data/test_data_d0/inputs/config.yaml}"

# ── No conda anywhere ────────────────────────────────────────────────────────
# molab uses pixi for everything reproducible and Apptainer for chrombpnet.
# There is no conda on the box and none is installed. Empty CONDA_INIT is the
# opt-in that makes activate_env a no-op instead of a hard error; see the
# header and lib/bash/common.sh.
export CONDA_INIT=""

# lib/bash/common.sh names these four after conda because that is what the
# cluster uses. On molab only the first is a real environment:
#
#   PREPROCESS_ENV         a pixi environment, defined in pixi.toml and
#                          mirroring envs/preprocess.yml. Steps 00.0/00.1/02.0.
#   CHROMBPNET_ENV         NOT an environment path here -- chrombpnet, its
#                          TensorFlow and bedtools come from the Apptainer
#                          container, which run_step.sh execs directly. It
#                          cannot be a pixi env: chrombpnet 1.0.1 pins
#                          tensorflow==2.8.0 / numpy==1.23.4 against CUDA 11
#                          wheels that do not exist for this GPU.
#   FINEMO_ENV             steps 10/11, not wired up on molab yet.
#   MOTIF_COMPENDIUM_ENV   step 09, not wired up on molab yet.
#
# The last three are never dereferenced (activate_env returns before it looks
# at the path), but they are set to a self-describing sentinel rather than left
# unset, because unset would fall back to a Sherlock home directory.
export PREPROCESS_ENV="${REPO_ROOT}/.pixi/envs/preprocess"
export CHROMBPNET_ENV="apptainer:${MOLAB_SANDBOX:-/marimo/containers/chrombpnet_sandbox}"
export FINEMO_ENV="unconfigured:molab"
export MOTIF_COMPENDIUM_ENV="unconfigured:molab"

# ── Data locations ───────────────────────────────────────────────────────────
export REFERENCE_ROOT="${REFERENCE_ROOT:-/marimo/data/references}"
export DATASET_ROOT="${DATASET_ROOT:-/marimo/data}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ── Container ────────────────────────────────────────────────────────────────
export MOLAB_SIF="${MOLAB_SIF:-/marimo/containers/chrombpnet.sif}"
export MOLAB_SANDBOX="${MOLAB_SANDBOX:-/marimo/containers/chrombpnet_sandbox}"
export MOLAB_CUDA_CACHE="${MOLAB_CUDA_CACHE:-/marimo/containers/cuda_jit_cache}"
export MOLAB_MPLCONFIG="${MOLAB_MPLCONFIG:-/marimo/containers/mplconfig}"
# Launcher logs. Default to the pipeline's OWN log_dir (lib/bash/config.sh:
# log_dir="${results_path}/logs"), derived from this dataset's output_dir,
# rather than inventing a second location. Override to put them elsewhere.
if [[ -z "${MOLAB_LOG_DIR:-}" ]]; then
    _molab_cfg="${DATASET_CONFIG:-${REPO_ROOT}/config/${DATASET}/config.yaml}"
    _molab_out=""
    if [[ -f "${_molab_cfg}" ]]; then
        _molab_out=$(python3 "${REPO_ROOT}/lib/python/utils/config.py" export "${_molab_cfg}" 2>/dev/null \
            | sed -n 's/^output_dir=//p' | tr -d '"')
    fi
    export MOLAB_LOG_DIR="${_molab_out:-${REPO_ROOT}/results}/logs"
    unset _molab_cfg _molab_out
fi

# Bind-mount the tree the data and results live under, into the container at
# the same path, so every absolute path in config.yaml resolves identically
# inside and outside.
export MOLAB_BIND="${MOLAB_BIND:-/marimo}"

# ── Real resources ───────────────────────────────────────────────────────────
# The box reports the host's CPUs and RAM (20+ cores, 160 GB), not the slice we
# actually get. Left unpinned, TensorFlow and OpenMP size their thread pools to
# the phantom count and thrash. Set this to the REAL core count.
export MOLAB_CPUS="${MOLAB_CPUS:-4}"
