#!/bin/bash
#SBATCH --job-name=prepare_signal
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 00.0.prepare_signal.sh
# Purpose: prepare this dataset's signal on CPU, once — everything the GPU
#          training tasks would otherwise each redo inside a GPU allocation.
#
# Nothing is copied. `signal_path` and `regions` in the config point at your data
# wherever it already lives; this step only writes DERIVED files, all under
# ${output_dir}/preprocessing/signal/. (An earlier 00.0 copied inputs into a
# <dataset>/data/ convention; that convention is gone, so the copy was pure
# duplication of a multi-GB file.)
#
# What it does:
#   1. filter reads to the main chromosomes AND count Tn5 cut sites, in a
#      single pass over the file
#   2. write the bigwig the training steps reuse
#
# It imports no chrombpnet: the pileup is numpy + pybigtools.
#
# The genome FASTA is needed ONLY to auto-detect the Tn5 shift, which reads the
# sequence around each cut site and compares it to reference Tn5 bias matrices
# -- it cannot be done without sequence. The pileup itself never reads any.
# So:
#   plus_shift/minus_shift set in config.yaml -> no detection, no FASTA read
#   either of them unset                      -> detection runs, ${genome_fa}
#                                                is required
# Detection is a one-off: run it once, write the answer into config.yaml.
#
# chrombpnet train / pipeline / bias train all begin with:
#   enzyme-shift auto-detection (samples reads, compares to a reference motif)
#   awk shift | sort | bedtools genomecov -bg -5 | sort | bedGraphToBigWig
# unconditionally — there is no existence check to short-circuit. Step 03.0 is a
# 5-fold x 4-factor array, so that identical conversion runs 20 times on GPU
# nodes, and 04.0 adds 5 more.
#
# This step runs it once here, on CPU. 03.0 and 04.0 then pass the result to
# src/chrombpnet_train.py, which installs it and skips chrombpnet's conversion.
# Reuse is refused (and chrombpnet converts normally) if the signal file, its
# md5, the assay or the chrombpnet version no longer match the sidecar.
#
# Also normalises a BAM signal to tagAlign, which is what chrombpnet converts it
# to internally anyway — after which set signal_type: tagalign in the config so
# the remaining steps read a text stream instead of decoding the BAM.
#
# Output (inside ${output_dir}/preprocessing/signal/):
#   data_unstranded.bw        the prepared bigwig
#   prepared_bigwig.json      sidecar: signal path, md5, assay, chrombpnet version
#
# Usage:
#   export DATASET=<name>
#   cd workflows/SLURM && sbatch 00.0.prepare_signal.sh
#
# Prerequisites: signal_path in config.yaml must exist, and references must be
#   installed (scripts/bash/download_references.sh). This step is the first
#   thing to run for a dataset.

# --- bootstrap: locate the repo root (identical block in every workflow step) --
# sbatch copies the submitted script to a node-local spool dir, so BASH_SOURCE
# does not point into the repo; SLURM_SUBMIT_DIR (the cwd at submit time) does.
# Walk up from whichever is usable until lib/bash/common.sh turns up. Override
# with REPO_ROOT if you submit from outside the checkout.
# An inherited REPO_ROOT is only trusted if it really is this repo -- the name
# is generic enough that another project may have exported it.
[[ -n "${REPO_ROOT}" && -f "${REPO_ROOT}/lib/bash/common.sh" ]] || REPO_ROOT=""
if [[ -z "${REPO_ROOT}" ]]; then
    for _d in "${SLURM_SUBMIT_DIR}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
        while [[ -n "${_d}" && "${_d}" != "/" ]]; do
            [[ -f "${_d}/lib/bash/common.sh" ]] && { REPO_ROOT="${_d}"; break 2; }
            _d="$(dirname "${_d}")"
        done
    done
fi
[[ -n "${REPO_ROOT}" ]] || {
    echo "ERROR: cannot locate the repo root. Submit from inside the checkout, or:" >&2
    echo "  export REPO_ROOT=/path/to/this/checkout" >&2
    exit 1
}
export REPO_ROOT
# --- end bootstrap -------------------------------------------------------------

# shellcheck source=lib/bash/config.sh
source "${REPO_ROOT}/lib/bash/config.sh" || exit 1

prepared_dir="${data_path}/signal"

metadata_start "00.0.prepare_signal"
metadata_inputs+=( "signal=${signal_path}" "genome=${genome_fa}" "chrom_sizes=${chrom_sizes}" )
metadata_outputs+=( "prepared_bigwig=${prepared_dir}/data_unstranded.bw" )
metadata_params+=( "signal_type=${signal_type}" "assay=${assay}" )
require_input "${signal_path}" ""
if [[ "${signal_type}" != "bigwig" ]]; then
    # chrom.sizes bounds the pileup, so it is always needed for reads.
    require_input "${chrom_sizes}" scripts/bash/download_references.sh
    # The genome is needed ONLY to auto-detect the Tn5 shift, which reads
    # sequence around cut sites. With plus_shift/minus_shift in the config there
    # is nothing to detect and no FASTA is touched.
    if [[ -z "${plus_shift}" || -z "${minus_shift}" ]]; then
        require_input "${genome_fa}" scripts/bash/download_references.sh
    fi
fi
preflight_check

# No `ml` here, and no external tools at all for the pileup: it is numpy +
# pybigtools (Rust), ~27x faster than chrombpnet's awk|sort|genomecov|sort|
# bedGraphToBigWig on a 5M-fragment file, and verified interval-for-interval
# identical against chrombpnet's own command in tests/test_pileup.py.
# chrombpnet is still imported, for shift detection only.
activate_env "${CONDA_ENV}"

mkdir -p "${prepared_dir}"

if [[ -f "${prepared_dir}/data_unstranded.bw" && -f "${prepared_dir}/prepared_bigwig.json" ]]; then
    echo "[$(date)] Prepared bigwig already exists, skipping: ${prepared_dir}"
    exit 0
fi

# ── the read path: filter and convert in ONE pass ────────────────────────────
# Dropping rows on contigs absent from chrom.sizes IS the main-chromosome
# filter, and the pileup reads every row anyway, so there is no separate filter
# pass. --write-filtered additionally keeps those rows as a file, which is only
# needed for the fallback where chrombpnet reads the reads itself; set
# filter_main_chroms: false to skip writing it and rewrite nothing.
filtered_args=()
if [[ "${filter_main_chroms:-true}" == "true" ]]; then
    filtered_args+=( --write-filtered "${prepared_dir}/${dataset_name}_main_chrs.tsv.gz" )
    metadata_outputs+=( "filtered_reads=${prepared_dir}/${dataset_name}_main_chrs.tsv.gz" )
fi

if [[ "${signal_type}" == "bigwig" ]]; then
    # Already a bigwig: nothing to convert. Register it as the prepared bigwig so
    # the GPU steps install it directly.
    #
    # It has to be what chrombpnet's own reads_to_bigwig would have written:
    # per-base Tn5 insertion counts (genomecov -bg -5, 5' ends only) with the
    # shift normalised to +4/-4 for ATAC. A bigwig from a previous chrombpnet
    # run qualifies. Read COVERAGE from e.g. deeptools bamCoverage does not --
    # wrong quantity and unshifted -- and nothing here can tell the difference.
    echo "[$(date)] Signal is already a bigwig; registering it, no conversion needed."
    echo "           It must be Tn5 insertion counts shifted to +4/-4 (ATAC), as"
    echo "           chrombpnet's own auxiliary/data_unstranded.bw is. Coverage"
    echo "           bigwigs (deeptools etc.) are the wrong signal and will train."
    ln -sf "${signal_path}" "${prepared_dir}/data_unstranded.bw"
    python - "$@" <<PY
import hashlib, json, pathlib, sys
sig = pathlib.Path("${signal_path}")
h = hashlib.md5()
with open(sig, "rb") as fh:
    while chunk := fh.read(8 << 20):
        h.update(chunk)
pathlib.Path("${prepared_dir}/prepared_bigwig.json").write_text(json.dumps({
    "signal_path": str(sig.resolve()),
    "signal_md5": h.hexdigest(),
    "signal_type": "bigwig",
    "assay": "${assay}",
    "chrombpnet_version": "n/a (no conversion performed)",
}, indent=2) + "\n")
PY
    echo "[$(date)] Registered ${prepared_dir}/data_unstranded.bw"
    exit 0
fi

echo "[$(date)] Preparing signal on CPU (this is what the GPU jobs will skip)"

shift_args=()
if [[ -n "${plus_shift}" && -n "${minus_shift}" ]]; then
    # Known shift: no detection, so no genome sequence is read.
    shift_args+=( --plus-shift "${plus_shift}" --minus-shift "${minus_shift}" )
else
    echo "[$(date)] plus_shift/minus_shift not set; detecting (needs ${genome_fa})"
    shift_args+=( --genome "${genome_fa}" )
fi

python "${src_dir}/cli.py" prepare-bigwig \
    "${shift_args[@]}" \
    ${filtered_args[@]+"${filtered_args[@]}"} \
    --signal-path  "${signal_path}" \
    --signal-type  "${signal_type}" \
    --assay        "${assay}" \
    --chrom-sizes  "${chrom_sizes}" \
    --out-dir      "${prepared_dir}" \
    --metadata-dir "${metadata_dir}"

echo "[$(date)] Done. 03.0 and 04.0 will reuse ${prepared_dir}/data_unstranded.bw"
