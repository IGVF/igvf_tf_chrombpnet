#!/usr/bin/env python3
"""
predict_bias_metrics.py
Fast QC metrics for a single trained ChromBPNet Tn5 bias model (one fold x
bias-factor combination), without running interpretation or TF-MoDISco.

`chrombpnet bias qc` bundles prediction (fast) together with DeepLIFT
interpretation and TF-MoDISco (slow) in one call, and can't be re-run on a
directory that `chrombpnet bias train` already populated (it recreates
auxiliary/evaluation with exist_ok=False). This script instead calls
chrombpnet's predict.main() directly, reusing the filtered peaks/nonpeaks
bed files and the shifted Tn5-insertion bigwig that `chrombpnet bias train`
already wrote to <output-dir>/auxiliary/, to produce just the counts/profile
metrics JSON that select_bias_model.py (03.1) needs to pick a winner per fold.

Output:
  <output-dir>/evaluation/<file-prefix>_bias_metrics.json

Run as part of 03.0.train_bias_model.sh, immediately after
`chrombpnet bias train` for the same fold x bias-factor combination.

Usage:
  python predict_bias_metrics.py \\
      --bias-model results/bias_models/bias_model_08/igvf3_cardiomyocyte_all_fold_0/models/igvf3_cardiomyocyte_all_fold_0_bias.h5 \\
      --output-dir results/bias_models/bias_model_08/igvf3_cardiomyocyte_all_fold_0 \\
      --file-prefix igvf3_cardiomyocyte_all_fold_0 \\
      --genome genome/hg38.fa \\
      --fold-json genome/folds/fold_0.json
"""

import argparse
import os
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
    p.add_argument("--fold-json", required=True, help="Fold chr split json (train/valid/test)")
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
    metrics_json = output_dir / "evaluation" / f"{fpx}bias_metrics.json"

    for f in (bias_model, peaks, nonpeaks, bigwig, args.genome, args.fold_json):
        if not Path(f).exists():
            raise FileNotFoundError(
                f"Missing {f} - run chrombpnet bias train for this fold/bias combo first."
            )

    if metrics_json.exists():
        logger.info(f"  {metrics_json} already exists, skipping.")
        return

    # Deferred: chrombpnet's model code imports Keras 3 and JAX, which is slow,
    # so --help and the input checks above stay instant. load_model_wrapper
    # reads chrombpnet 1.x and 2.x model files alike; the shapes are read the
    # way chrombpnet's own bias_model_qc reads them.
    import chrombpnet.training.predict as predict
    from chrombpnet.helpers.hyperparameters.param_utils import load_model_wrapper

    bias_md = load_model_wrapper(str(bias_model))
    inputlen = int(bias_md.input_shape[1])
    outputlen = int(bias_md.output_shape[0][1])
    del bias_md  # predict.main loads the model again

    ns = argparse.Namespace(
        model_h5=str(bias_model),
        peaks=str(peaks),
        nonpeaks=str(nonpeaks),
        output_prefix=str(output_dir / "evaluation" / f"{fpx}bias"),
        batch_size=args.batch_size,
        genome=args.genome,
        bigwig=str(bigwig),
        chr_fold_path=args.fold_json,
        inputlen=inputlen,
        outputlen=outputlen,
    )

    os.makedirs(output_dir / "evaluation", exist_ok=True)
    predict.main(ns)
    logger.info(f"  Wrote {metrics_json}")


if __name__ == "__main__":
    main()
