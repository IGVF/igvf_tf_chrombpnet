#!/usr/bin/env python3
"""
run_bias_qc.py
GPU half of the selected bias model's QC (03.2): predictions with metrics on
the test chromosomes, then DeepLIFT contribution scores for BOTH heads on a
30K subsample of the bias peaks -- what `chrombpnet bias qc` runs before
TF-MoDISco. 03.3 (src/motif_qc.py, CPU) finds the motifs in these scores;
inspecting them for Tn5 vs GC-rich TF-like patterns is what verifies the Tn5
signal has actually been learned.

The `chrombpnet bias qc` CLI can't be pointed at the same output dir that
`chrombpnet bias train` already populated - it recreates auxiliary/
and evaluation/ with exist_ok=False and crashes. This script instead calls
the same chrombpnet functions pipelines.bias_model_qc() does (predict.main,
interpret.main), reusing the filtered peaks/nonpeaks bed files
`chrombpnet bias train` wrote into <output-dir>/auxiliary/. The observed
signal is 00.0's prepared bigwig (--bigwig): training reads it in place
(`-bw`), so there is no copy under auxiliary/ to reuse.

Only this half is here. The TF-MoDISco half (`--stage modisco`/`all`, which
ran `modisco report` and read its motifs.html) is gone: 03.3 runs
src/motif_qc.py instead, and modisco 2.5's `modisco report` no longer writes
motifs.html. DeepSHAP's references are seeded (chrombpnet's default, 1234),
and --device gpu (the default) makes interpretation fail rather than fall
back to the CPU; predict has no such switch, so 03.2 also calls require_gpu.

Run only for the bias model selected per fold by select_bias_model.py (03.1),
via 03.2.qc_selected_bias.sh - NOT for the full fold x bias-factor sweep.

Output (written under <output-dir>):
  evaluation/<file-prefix>_bias_predictions.h5, _bias_metrics.json and the
    per-split counts/JSD plots (same metrics as 03.0's fast step, recomputed)
  auxiliary/<file-prefix>_30K_subsample_peaks.bed
  auxiliary/interpret_subsample/<file-prefix>_bias.{counts,profile}_scores.h5,
    .interpreted_regions.bed and .interpret.args.json

Usage:
  python run_bias_qc.py \\
      --bias-model results/bias_models/bias_model_08/igvf3_cardiomyocyte_all_fold_0/models/igvf3_cardiomyocyte_all_fold_0_bias.h5 \\
      --output-dir results/bias_models/bias_model_08/igvf3_cardiomyocyte_all_fold_0 \\
      --file-prefix igvf3_cardiomyocyte_all_fold_0 \\
      --genome genome/hg38.fa \\
      --fold-json genome/folds/fold_0.json \\
      --bigwig results/preprocessing/signal/data_unstranded.bw
"""

import argparse
import copy
import os
import sys
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

from utils import log, metadata, regions  # noqa: E402

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
    p.add_argument(
        "--bigwig",
        required=True,
        help="Observed signal the bias model was trained on: 00.0's prepared "
        "preprocessing/signal/data_unstranded.bw",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--stage",
        default="gpu",
        choices=["gpu"],
        help="Predictions + DeepLIFT interpretation, the only stage; TF-MoDISco is 03.3.",
    )
    p.add_argument(
        "--device",
        default="gpu",
        choices=["auto", "gpu", "cpu"],
        help="Interpretation device (chrombpnet's --device): 'gpu' fails if JAX sees no GPU",
    )
    p.add_argument(
        "--force", action="store_true", help="Re-run a stage whose outputs already exist"
    )
    log.add_logging_args(p)
    return p.parse_args()


def _run_gpu_stage(args, ns, fpx, output_dir):
    """Predictions + DeepLIFT contribution scores. Needs the GPU."""
    import chrombpnet.evaluation.interpret.interpret as interpret
    import chrombpnet.training.predict as predict
    from chrombpnet.helpers.hyperparameters.param_utils import load_model_wrapper

    bias_md = load_model_wrapper(model_h5=str(args.bias_model))
    ns.inputlen = int(bias_md.input_shape[1])
    ns.outputlen = int(bias_md.output_shape[0][1])

    logger.info("  [gpu] predictions")
    a = copy.deepcopy(ns)
    a.output_prefix = str(output_dir / "evaluation" / f"{fpx}bias")
    a.model_h5 = str(args.bias_model)
    predict.main(a)

    # chrombpnet interprets a 30K subsample, seed 1234 (utils.regions keeps the rule).
    sub = regions.subsample_regions(
        ns.peaks, output_dir / "auxiliary" / f"{fpx}30K_subsample_peaks.bed"
    )

    # exist_ok=True where chrombpnet uses False, so a killed job can be resumed
    # instead of dying on FileExistsError before doing any work.
    os.makedirs(output_dir / "auxiliary" / "interpret_subsample", exist_ok=True)
    logger.info("  [gpu] DeepLIFT interpretation (counts + profile)")
    a = copy.deepcopy(ns)
    a.profile_or_counts = ["counts", "profile"]
    a.regions = str(sub)
    a.model_h5 = str(args.bias_model)
    a.output_prefix = str(output_dir / "auxiliary" / "interpret_subsample" / f"{fpx}bias")
    a.debug_chr = None
    interpret.main(a)


def main():
    args = parse_args()
    log.setup_from_args(args)
    output_dir = Path(args.output_dir)
    fpx = f"{args.file_prefix}_"

    bias_model = Path(args.bias_model)
    peaks = output_dir / "auxiliary" / f"{fpx}filtered.bias_peaks.bed"
    nonpeaks = output_dir / "auxiliary" / f"{fpx}filtered.bias_nonpeaks.bed"
    bigwig = Path(args.bigwig)

    if not bigwig.exists():
        raise FileNotFoundError(f"Missing {bigwig} - run 00.0.prepare_signal.sh first.")
    for f in (bias_model, peaks, nonpeaks, args.genome, args.fold_json):
        if not Path(f).exists():
            raise FileNotFoundError(
                f"Missing {f} - run chrombpnet bias train for this fold/bias combo first."
            )

    interpret_dir = output_dir / "auxiliary" / "interpret_subsample"
    profile_h5 = interpret_dir / f"{fpx}bias.profile_scores.h5"
    counts_h5 = interpret_dir / f"{fpx}bias.counts_scores.h5"

    # The fields predict.main (bigwig, peaks, nonpeaks, chr_fold_path, genome,
    # batch_size, and inputlen/outputlen set from the model) and interpret.main
    # read. interpret takes its DeepSHAP settings via getattr with chrombpnet's
    # defaults (seed 1234, precision auto); device is the one set here.
    ns = argparse.Namespace(
        bigwig=str(bigwig),
        genome=args.genome,
        peaks=str(peaks),
        nonpeaks=str(nonpeaks),
        chr_fold_path=args.fold_json,
        batch_size=args.batch_size,
        device=args.device,
    )

    if profile_h5.exists() and counts_h5.exists() and not args.force:
        logger.info("  contribution scores already exist, skipping interpretation")
    else:
        _run_gpu_stage(args, ns, fpx, output_dir)

    logger.info(
        "  stage '%s' complete for %s -> %s/evaluation/",
        args.stage,
        args.file_prefix,
        output_dir,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        metadata.report_peak_rss()  # DeepLIFT's peak (03.2)
