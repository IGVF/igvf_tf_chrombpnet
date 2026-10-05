#!/bin/bash
# checks.sh
# Purpose: the verification that would otherwise need a separate pilot job, run
#   alongside the datasets inside the same whole-node allocation (run_box.sh
#   starts it in the background). Each check writes its own log under
#   <BOX_DIR>/checks/ and one PASS/FAIL line to <BOX_DIR>/checks/summary.tsv.
#   None of them changes a pipeline output.
#
#   gpu        nvidia-smi driver + JAX sees the GPUs in the chrombpnet env
#   bigwig     figwig engine == numpy engine, every base of every main chromosome
#   macs       MACS3 PR #756 == the MACS3 3.0.5 release (byte-identical
#              narrowPeak, the PR's claim) on one pseudoreplicate of
#              CHECK_DATASET, chr21+chr22; how far 3.0.4 -- the version the
#              pseudobulking pipeline pins -- differs is reported alongside.
#              (3.0.4 and 3.0.5 split --call-summits sub-peaks and score them
#              slightly differently: an upstream change, not the PR's.)
#
# Input:  BOX_DIR, BOX_SCRATCH (from run_box.sh); CHECK_DATASET (file stem,
#         default the smallest library); MACS3_304 (a MACS3 3.0.4 macs3)
# Usage:  bash workflows/dcai/checks.sh   (normally via run_box.sh)

set -uo pipefail

: "${REPO_ROOT:?source workflows/dcai/env.sh first}"
out="${BOX_DIR:?}/checks"
mkdir -p "${out}"
summary="${out}/summary.tsv"
check_dataset="${CHECK_DATASET:-village2_d14_LH}"
frags="${DATASET_ROOT}/fragments/${check_dataset}.fragments.tsv.gz"
macs_304="${MACS3_304:-/dcai/users/mateug/envs/snapatac2/bin/macs3}"
# The 3.0.5 release from bioconda; pixi caches it after the first use.
macs_305=( pixi exec -c conda-forge -c bioconda --spec "macs3=3.0.5" -- macs3 )
genome_dir="${REFERENCE_ROOT}/hg38/Sequence"
sizes_main="${genome_dir}/chrom_sizes/IGVF.DACC.GRCh38.chrom.sizes.main.tsv"
pixi_peaks=( pixi run --frozen --manifest-path "${REPO_ROOT}/pixi.toml" -e peaks )

record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "${summary}"; echo "[$(date)] check $1: $2 ($3)"; }

check_gpu() {
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > "${out}/gpu.log" 2>&1
    if CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-13.0}" pixi run --frozen \
            --manifest-path "${CHROMBPNET_REPO}/pyproject.toml" -e "${CHROMBPNET_PIXI_ENV}" gpu-check \
            >> "${out}/gpu.log" 2>&1; then
        record gpu PASS "$(sed -n 2p "${out}/gpu.log")"
    else
        record gpu FAIL "see ${out}/gpu.log"
    fi
}

check_bigwig() {
    local d="${out}/bigwig"
    mkdir -p "${d}/numpy" "${d}/figwig"
    for engine in figwig numpy; do
        "${pixi_peaks[@]}" python "${REPO_ROOT}/src/cli.py" prepare-bigwig \
            --signal-path "${frags}" --signal-type fragments --assay ATAC \
            --plus-shift 4 --minus-shift -5 --chrom-sizes "${sizes_main}" \
            --engine "${engine}" --threads 8 --out-dir "${d}/${engine}" \
            --metadata-dir "${d}/metadata" > "${d}/${engine}.log" 2>&1 \
            || { record bigwig FAIL "prepare-bigwig --engine ${engine}, see ${d}/${engine}.log"; return; }
    done
    if "${pixi_peaks[@]}" python "${REPO_ROOT}/workflows/dcai/compare_bigwigs.py" \
            "${d}/figwig/data_unstranded.bw" "${d}/numpy/data_unstranded.bw" "${sizes_main}" \
            > "${d}/compare.tsv" 2>&1; then
        record bigwig PASS "$(grep ^TOTAL "${d}/compare.tsv")"
    else
        record bigwig FAIL "$(grep ^TOTAL "${d}/compare.tsv")"
    fi
}

check_macs() {
    local d="${out}/macs" s="${BOX_SCRATCH}/checks_macs"
    mkdir -p "${d}" "${s}"
    grep -P '^chr(21|22)\t' "${genome_dir}/chrom_sizes/IGVF.DACC.GRCh38.chrom.sizes.tsv" > "${s}/chr21_22.sizes"
    # The same insertion records the step gives MACS (BED route, +4/-5 present,
    # so the correction is 0/+1).
    (cd "${REPO_ROOT}" && "${pixi_peaks[@]}" python -c "
import sys; sys.path.insert(0, 'lib/python')
from utils import peakcall
s = sys.argv[2]
print(peakcall.split_insertions(sys.argv[1], s + '/chr21_22.sizes', s + '/rep1.bed', s + '/rep2.bed', 0, 1, macs_input='bed', threads=4))
" "${frags}" "${s}") >> "${d}/split.log" 2>&1 || { record macs FAIL "split, see ${d}/split.log"; return; }
    local flags=( -f BED -g hs -p 0.01 --shift -75 --extsize 150 --nomodel --keep-dup all --call-summits )
    /usr/bin/time -v "${pixi_peaks[@]}" macs3 callpeak -t "${s}/rep1.bed" -n pr --outdir "${d}/pr" "${flags[@]}" \
        > "${d}/pr.log" 2>&1 || { record macs FAIL "PR build, see ${d}/pr.log"; return; }
    /usr/bin/time -v "${macs_305[@]}" callpeak -t "${s}/rep1.bed" -n pr --outdir "${d}/v305" "${flags[@]}" \
        > "${d}/v305.log" 2>&1 || { record macs FAIL "3.0.5 release, see ${d}/v305.log"; return; }
    "${macs_304}" callpeak -t "${s}/rep1.bed" -n pr --outdir "${d}/v304" "${flags[@]}" \
        > "${d}/v304.log" 2>&1
    local times v304
    times="PR $(grep -m1 'Elapsed (wall' "${d}/pr.log" | awk '{print $NF}') vs 3.0.5 $(grep -m1 'Elapsed (wall' "${d}/v305.log" | awk '{print $NF}')"
    v304="3.0.4: $(wc -l < "${d}/v304/pr_peaks.narrowPeak" 2>/dev/null || echo '?') rows vs $(wc -l < "${d}/pr/pr_peaks.narrowPeak") (informational)"
    if cmp -s "${d}/pr/pr_peaks.narrowPeak" "${d}/v305/pr_peaks.narrowPeak"; then
        record macs PASS "PR == 3.0.5 release, narrowPeak byte-identical; ${times}; ${v304}"
    else
        record macs FAIL "PR != 3.0.5 release; ${times}; ${v304}"
    fi
    rm -rf "${s}"
}

check_gpu &
check_bigwig &
check_macs &
wait
echo "[$(date)] checks done:"
cat "${summary}"
