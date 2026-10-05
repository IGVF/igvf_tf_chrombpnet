#!/usr/bin/env python3
"""Compare two bigWigs base by base over the chromosomes in a chrom.sizes.

figwig writes single-base entries where pybigtools writes merged runs, so the
two engines' files differ in bytes; what must agree is every decoded value.
Reads each chromosome whole with pyBigWig (the reader chrombpnet trains with),
missing bases as 0. Prints one line per chromosome and exits 1 on any
difference.

    python workflows/dcai/compare_bigwigs.py a.bw b.bw chrom.sizes.main.tsv
"""

import sys

import numpy as np
import pyBigWig


def values(bw, chrom, length):
    if chrom not in bw.chroms():
        return np.zeros(length, dtype=np.float64)
    v = bw.values(chrom, 0, length, numpy=True)
    return np.nan_to_num(v, nan=0.0)


def main(argv=None) -> int:
    a_path, b_path, sizes_path = (argv or sys.argv[1:])[:3]
    sizes = [ln.split("\t")[:2] for ln in open(sizes_path).read().splitlines() if ln.strip()]
    a, b = pyBigWig.open(a_path), pyBigWig.open(b_path)
    bad = 0
    total_a = total_b = 0.0
    for chrom, length in sizes:
        va, vb = values(a, chrom, int(length)), values(b, chrom, int(length))
        diff = int(np.count_nonzero(va != vb))
        total_a, total_b = total_a + va.sum(), total_b + vb.sum()
        print(f"{chrom}\tsum_a={va.sum():.0f}\tsum_b={vb.sum():.0f}\tbases_differing={diff}")
        bad += diff
    print(f"TOTAL\tsum_a={total_a:.0f}\tsum_b={total_b:.0f}\tbases_differing={bad}")
    print("IDENTICAL" if bad == 0 else "DIFFERENT")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
