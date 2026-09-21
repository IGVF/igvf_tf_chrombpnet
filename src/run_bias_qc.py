#!/usr/bin/env python3
"""
run_bias_qc.py
Full ChromBPNet Tn5 bias model QC (marginal footprinting + DeepLIFT
interpretation + TF-MoDISco) for one already-trained fold x bias-factor
combination - the expensive step that verifies the Tn5 signal has actually
been learned (checked by inspecting the resulting motifs for Tn5 vs GC-rich
TF-like patterns).

The `chrombpnet bias qc` CLI can't be pointed at the same output dir that
`chrombpnet bias train` already populated - it recreates auxiliary/
and evaluation/ with exist_ok=False and crashes. This script instead calls
chrombpnet's pipelines.bias_model_qc() directly, reusing the filtered
peaks/nonpeaks bed files and shifted bigwig already written by
`chrombpnet bias train` into <output-dir>/auxiliary/.

Run only for the bias model selected per fold by select_bias_model.py (03.1),
via 03.2.qc_selected_bias.sh - NOT for the full fold x bias-factor sweep.

Output (written under <output-dir>):
  evaluation/<file-prefix>_bias_metrics.json          (same metrics as 03.0's fast step, recomputed)
  evaluation/modisco_profile/, modisco_counts/         TF-MoDISco motif reports
  evaluation/<file-prefix>_bias_profile.pdf            rendered motif report

Usage:
  python run_bias_qc.py \\
      --bias-model results/bias_models/bias_model_08/igvf3_cardiomyocyte_all_fold_0/models/igvf3_cardiomyocyte_all_fold_0_bias.h5 \\
      --output-dir results/bias_models/bias_model_08/igvf3_cardiomyocyte_all_fold_0 \\
      --file-prefix igvf3_cardiomyocyte_all_fold_0 \\
      --genome genome/hg38.fa \\
      --chrom-sizes genome/hg38.chrom.sizes \\
      --fold-json genome/folds/fold_0.json
"""

import argparse
import sys
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

from utils import log  # noqa: E402

logger = log.get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bias-model", required=True, help="Path to trained bias model .h5")
    p.add_argument(
        "--output-dir", required=True, help="Same output dir used by chrombpnet bias train"
    )
    p.add_argument(
        "--file-prefix", required=True, help="Same file prefix used by chrombpnet bias train"
    )
    p.add_argument("--genome", required=True, help="Reference genome fasta")
    p.add_argument("--chrom-sizes", required=True, help="Chrom sizes file")
    p.add_argument("--fold-json", required=True, help="Fold chr split json (train/valid/test)")
    p.add_argument("--data-type", default="ATAC", choices=["ATAC", "DNASE"])
    p.add_argument("--batch-size", type=int, default=64)
    log.add_logging_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    log.setup_from_args(args)
    output_dir = Path(args.output_dir)
    fpx = f"{args.file_prefix}_"

    bias_model = Path(args.bias_model)
    peaks = output_dir / "auxiliary" / f"{fpx}filtered.bias_peaks.bed"
    nonpeaks = output_dir / "auxiliary" / f"{fpx}filtered.bias_nonpeaks.bed"
    bigwig = output_dir / "auxiliary" / f"{fpx}data_unstranded.bw"

    for f in (bias_model, peaks, nonpeaks, bigwig, args.genome, args.chrom_sizes, args.fold_json):
        if not Path(f).exists():
            raise FileNotFoundError(
                f"Missing {f} - run chrombpnet bias train for this fold/bias combo first."
            )

    interpret_subsample_dir = output_dir / "auxiliary" / "interpret_subsample"
    if interpret_subsample_dir.exists():
        logger.info(
            f"  {interpret_subsample_dir} already exists, assuming QC already ran. Skipping."
        )
        return

    import chrombpnet.pipelines as pipelines

    ns = argparse.Namespace(
        bigwig=str(bigwig),
        bias_model=str(bias_model),
        genome=args.genome,
        chrom_sizes=args.chrom_sizes,
        output_dir=str(output_dir),
        data_type=args.data_type,
        peaks=str(peaks),
        nonpeaks=str(nonpeaks),
        chr_fold_path=args.fold_json,
        file_prefix=args.file_prefix,
        batch_size=args.batch_size,
        html_prefix="./",
        cmd_bias="qc",
    )

    pipelines.bias_model_qc(ns)
    logger.info(f"  QC complete for {args.file_prefix} -> {output_dir}/evaluation/")


if __name__ == "__main__":
    main()
