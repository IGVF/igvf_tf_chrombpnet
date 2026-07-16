#!/usr/bin/env python3
"""
07.average_contrib_scores.py
Average DeepLIFT contribution scores across all 5 cross-validation folds for
a given dataset. Produces a single "average_shaps.counts.h5" per dataset with reduced
fold-specific noise, for use in motif discovery (step 09) and pattern merging
(step 11).

Why average:
  Each fold's model was trained on a different 80% of peaks, so its
  contribution scores reflect slightly different random variation. Averaging
  cancels this noise while preserving the signal that is consistently
  important across all models.

Input per fold:
  {full_model_dir}/{dataset}_{peak_type}_fold_{fold}/interpretation/
      interpretation.counts_scores.h5

  Expected H5 keys (shape: n_peaks x 4 x seq_len):
      raw/seq             - one-hot encoded sequences (identical across folds)
      shap/seq            - hypothetical contribution scores
      projected_shap/seq  - projected contribution scores (used by MoDISco)

Output:
  {averaged_dir}/{dataset}/{dataset}_average_shaps.counts.h5  (same key structure)

Usage:
  python 07.average_contrib_scores.py \\
      --dataset d0 \\
      --folds 0 1 2 3 4 \\
      --full-model-dir results/full_models \\
      --peak-type all \\
      --out-dir results/contrib_scores/d0
"""

import argparse
import os
import sys

import h5py
import hdf5plugin
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset label, e.g. d0")
    p.add_argument(
        "--folds",
        required=True,
        nargs="+",
        type=str,
        help="Fold indices, e.g. 0 1 2 3 4",
    )
    p.add_argument(
        "--full-model-dir",
        required=True,
        help="Path to chrombpnet_full_model_selected",
    )
    p.add_argument("--peak-type", default="all")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for average_shaps.counts.h5",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f"{args.dataset}_average_shaps.counts.h5")
    if os.path.exists(out_path):
        print(f"Output already exists, skipping: {out_path}")
        return

    # Locate available fold H5 files
    fold_h5s = []
    for fold in args.folds:
        h5_path = os.path.join(
            args.full_model_dir,
            f"{args.dataset}_{args.peak_type}_fold_{fold}",
            "interpretation",
            "interpretation.counts_scores.h5",
        )
        if os.path.exists(h5_path):
            fold_h5s.append((fold, h5_path))
        else:
            print(
                f"  [WARN] not found, skipping fold {fold}: {h5_path}",
                file=sys.stderr,
            )

    if len(fold_h5s) == 0:
        sys.exit(f"ERROR: no contribution H5 files found for dataset={args.dataset}")

    print(f"  Averaging {len(fold_h5s)} folds: {[f for f, _ in fold_h5s]}")

    # Accumulate sums
    avg_projected = None
    avg_shap = None
    raw = None
    n = 0

    for fold, h5_path in fold_h5s:
        print(f"  Loading fold {fold}: {h5_path}")
        with h5py.File(h5_path, "r") as fh:
            avg_projected_shap = np.array(fh["projected_shap"]["seq"])
            shap = np.array(fh["shap"]["seq"])

            if raw is None:
                raw = np.array(fh["raw"]["seq"])
                avg_projected = avg_projected_shap.copy()
                avg_shap = shap.copy()
            else:
                assert raw.shape == avg_projected_shap.shape, (
                    f"Shape mismatch at fold {fold}: {avg_projected_shap.shape} vs {raw.shape}"
                )
                avg_projected += avg_projected_shap
                avg_shap += shap
        n += 1

    avg_projected /= n
    avg_shap /= n

    print(f"Saving averaged SHAP scores to {out_path}")
    with h5py.File(out_path, "w") as fh:
        fh.create_dataset("raw/seq", data=raw, compression="gzip")
        fh.create_dataset("shap/seq", data=avg_shap, compression="gzip")
        fh.create_dataset("projected_shap/seq", data=avg_projected, compression="gzip")

    print("Done!")


if __name__ == "__main__":
    main()
