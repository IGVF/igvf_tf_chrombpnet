#!/bin/bash
#SBATCH --job-name=prepare_signal
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 00.1.prepare_signal.sh
# Purpose: do on CPU, once, the work every GPU training task would otherwise
#          repeat inside its own GPU allocation.
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
#   cd workflows/SLURM && sbatch 00.1.prepare_signal.sh
#
# Prerequisites: the signal file in config.yaml must exist, and references must
#   be installed (scripts/bash/download_references.sh).

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

metadata_start "00.1.prepare_signal"
metadata_inputs+=( "signal=${signal_path}" "genome=${genome_fa}" "chrom_sizes=${chrom_sizes}" )
metadata_outputs+=( "prepared_bigwig=${prepared_dir}/data_unstranded.bw" )
metadata_params+=( "signal_type=${signal_type}" "assay=${assay}" )
require_input "${signal_path}" 00.0.copy_and_prepare_data.sh
require_input "${genome_fa}"   scripts/bash/download_references.sh
require_input "${chrom_sizes}" scripts/bash/download_references.sh
preflight_check

ml biology bedtools
load_render_modules
activate_env "${CONDA_ENV}"
metadata_tools+=( "$(tool_version chrombpnet chrombpnet --version)" )

mkdir -p "${prepared_dir}"

if [[ -f "${prepared_dir}/data_unstranded.bw" && -f "${prepared_dir}/prepared_bigwig.json" ]]; then
    echo "[$(date)] Prepared bigwig already exists, skipping: ${prepared_dir}"
    exit 0
fi

if [[ "${signal_type}" == "bigwig" ]]; then
    # Already a bigwig: nothing to convert. Register it as the prepared bigwig so
    # the GPU steps install it directly. It must already carry the Tn5 shift
    # chrombpnet expects (+4/-4 for ATAC) -- we cannot verify that, and an
    # unshifted bigwig will train without complaint and be subtly wrong.
    echo "[$(date)] Signal is already a bigwig; registering it, no conversion needed."
    echo "           NOTE: assumed to be Tn5-shifted the way chrombpnet expects."
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

python "${src_dir}/cli.py" prepare-bigwig \
    --signal-path  "${signal_path}" \
    --signal-type  "${signal_type}" \
    --assay        "${assay}" \
    --genome       "${genome_fa}" \
    --chrom-sizes  "${chrom_sizes}" \
    --out-dir      "${prepared_dir}" \
    --metadata-dir "${metadata_dir}"

echo "[$(date)] Done. 03.0 and 04.0 will reuse ${prepared_dir}/data_unstranded.bw"
