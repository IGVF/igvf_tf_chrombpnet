#!/usr/bin/env python3
"""
08.contribs_to_bigwig.py
Convert the fold-averaged contribution score H5 produced by
07.average_contrib_scores.py into a bigwig file.

The averaged H5 stores projected_shap/seq with shape (N, 4, seqlen).
Summing over the 4-base axis yields the per-position contribution score
for the actual nucleotide at that position (identical to what
chrombpnet's importance_hdf5_to_bigwig.py does on a per-fold H5).

Input:
  {averaged_dir}/{dataset}/{dataset}_average_shaps.counts.h5
  interpretation.interpreted_regions.bed  (from any single fold;
      rows must match the averaged H5 - all folds use the same peaks)

Output:
  {averaged_dir}/{dataset}/{dataset}_average_shaps.counts.bw

Usage:
  python 08.contribs_to_bigwig.py \\
      --h5          results/contrib_scores/d0/d0_average_shaps.counts.h5 \\
      --regions     results/full_models/d0_all_fold_0/interpretation/interpretation.interpreted_regions.bed \\
      --chrom-sizes results/preprocessing/hg38.chrom.sizes \\
      --output-bw   results/contrib_scores/d0/d0_average_shaps.counts.bw
"""

import argparse
import os

import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="Path to average_shaps.counts.h5")
    p.add_argument(
        "--regions",
        required=True,
        help="10-column BED file of interpreted regions "
        "(interpretation.interpreted_regions.bed from any fold)",
    )
    p.add_argument("--chrom-sizes", required=True, help="Chromosome sizes 2-column TSV")
    p.add_argument("--output-bw", required=True, help="Output bigwig path")
    p.add_argument(
        "--debug-chr",
        type=str,
        default=None,
        help="Restrict to one chromosome for debugging (e.g. chr1)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output_bw):
        print(f"Output already exists, skipping: {args.output_bw}")
        return

    print(f"Loading projected SHAP scores from: {args.h5}")
    with h5py.File(args.h5, "r") as fh:
        projected = np.array(fh["projected_shap"]["seq"])  # (N, 4, seqlen)

    seqlen = projected.shape[2]
    assert seqlen % 2 == 0, f"seqlen must be even, got {seqlen}"
    print(f"  Shape: {projected.shape}  (N={projected.shape[0]}, seqlen={seqlen})")

    # Sum over 4 bases → per-position contribution score  (N, seqlen)
    scores = projected.sum(axis=1).astype(np.float32)

    gs = bigwig_helper.read_chrom_sizes(args.chrom_sizes)
    regions = bigwig_helper.get_regions(args.regions, seqlen)

    assert scores.shape[0] == len(regions), (
        f"Row count mismatch: H5 has {scores.shape[0]} peaks but "
        f"regions BED has {len(regions)} rows."
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_bw)), exist_ok=True)
    bigwig_helper.write_bigwig(
        scores, regions, gs, args.output_bw, debug_chr=args.debug_chr
    )
    print(f"Wrote bigwig: {args.output_bw}")


if __name__ == "__main__":
    main()
