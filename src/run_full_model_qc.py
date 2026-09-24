#!/usr/bin/env python3
"""
run_full_model_qc.py
DeepLIFT contribution scores for a trained ChromBPNet full model on a 30K peak
subsample -- the interpretation `chrombpnet pipeline` runs after training,
which 04.0 stops short of. This is 04.4, the GPU half of the full model's
per-fold QC; 04.5 (src/motif_qc.py, CPU) finds the motifs in these scores.

Why a separate script: `chrombpnet pipeline` runs interpretation and
TF-MoDISco inside the training job, so a GPU allocation sat idle through
TF-MoDISco, which is CPU-only and the long pole -- the same split 03.2/03.3
make for the bias model. 04.0 stops `pipeline` after the marginal footprints,
before interpretation; this script does the interpretation as
chrombpnet_train_pipeline would -- the 30K subsample (seed 1234) of
auxiliary/filtered.peaks.bed, DeepSHAP references seeded with chrombpnet's
default (1234) -- with one deliberate difference: it scores BOTH heads, where
the pipeline's counts run is commented out upstream. The counts head is where
a model absorbs composition (GC content, on d0's bias model) without any of it
showing in the profile patterns, so the motif QC (04.5) looks at both. It
skips if both score files already exist.

--device gpu (the default, chrombpnet's --device) makes interpretation fail
when JAX sees no GPU instead of running a day-long job on the CPU.

The motif half used to live here too (`--stage modisco`). 04.5 runs
src/motif_qc.py instead, in the same chrombpnet env; see that file for why.

Input  (written by 04.0 under <model-dir>):
  models/<fpx>chrombpnet_nobias.h5, auxiliary/<fpx>filtered.peaks.bed,
  evaluation/<fpx>chrombpnet_nobias_max_bias_response.txt (the footprints)
Output (under <model-dir>):
  auxiliary/<fpx>30K_subsample_peaks.bed
  auxiliary/interpret_subsample/<fpx>chrombpnet_nobias.{profile,counts}_scores.h5
  auxiliary/interpret_subsample/<fpx>chrombpnet_nobias.interpreted_regions.bed
  auxiliary/interpret_subsample/<fpx>chrombpnet_nobias.interpret.args.json

Usage:
  python run_full_model_qc.py --stage gpu \\
      --model-dir results/full_models/d0_all_fold_0 --genome genome/hg38.fa
"""

import argparse
import os
import sys
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

from utils import log, metadata, regions  # noqa: E402

logger = log.get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Full-model interpretation QC (DeepLIFT / TF-MoDISco).")
    p.add_argument("--stage", required=True, choices=["gpu"])
    p.add_argument("--model-dir", required=True, help="04.0's output dir for one dataset x fold")
    p.add_argument("--genome", required=True, help="Reference genome fasta")
    p.add_argument("--file-prefix", default="", help="chrombpnet file prefix; 04.0 uses none")
    p.add_argument("--data-type", default="ATAC", choices=["ATAC", "DNASE"])
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


def paths(model_dir: Path, fpx: str) -> dict:
    interp = model_dir / "auxiliary" / "interpret_subsample"
    return {
        "model": model_dir / "models" / f"{fpx}chrombpnet_nobias.h5",
        "peaks": model_dir / "auxiliary" / f"{fpx}filtered.peaks.bed",
        "footprints": model_dir / "evaluation" / f"{fpx}chrombpnet_nobias_max_bias_response.txt",
        "subsample": model_dir / "auxiliary" / f"{fpx}30K_subsample_peaks.bed",
        "interp_dir": interp,
        "profile_scores": interp / f"{fpx}chrombpnet_nobias.profile_scores.h5",
        "counts_scores": interp / f"{fpx}chrombpnet_nobias.counts_scores.h5",
    }


def run_gpu(args, p: dict, fpx: str) -> None:
    """DeepLIFT on both heads (chrombpnet_train_pipeline does profile only)."""
    if p["profile_scores"].exists() and p["counts_scores"].exists() and not args.force:
        logger.info("  profile and counts scores exist, skipping interpretation")
        return
    import chrombpnet.evaluation.interpret.interpret as interpret

    regions.subsample_regions(p["peaks"], p["subsample"])
    # exist_ok=True where chrombpnet uses False, so a killed job resumes
    # instead of dying on FileExistsError before doing any work.
    os.makedirs(p["interp_dir"], exist_ok=True)
    a = argparse.Namespace(
        genome=args.genome,
        regions=str(p["subsample"]),
        model_h5=str(p["model"]),
        output_prefix=str(p["interp_dir"] / f"{fpx}chrombpnet_nobias"),
        profile_or_counts=["profile", "counts"],  # the pipeline: profile only
        debug_chr=None,
        # DeepSHAP settings not given here (seed, precision) take chrombpnet's
        # defaults inside interpret.main, as in the pipeline.
        device=args.device,
    )
    logger.info("  [gpu] DeepLIFT interpretation (profile + counts) on %s", p["subsample"].name)
    interpret.main(a)


def main():
    args = parse_args()
    log.setup_from_args(args)
    model_dir = Path(args.model_dir)
    fpx = f"{args.file_prefix}_" if args.file_prefix else ""
    p = paths(model_dir, fpx)
    for key in ("model", "peaks", "footprints"):
        if not p[key].exists():
            raise FileNotFoundError(
                f"Missing {p[key]} - run 04.0.train_full_model.sh for this fold first."
            )
    run_gpu(args, p, fpx)
    logger.info("  stage '%s' complete -> %s", args.stage, model_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        metadata.report_peak_rss()  # DeepLIFT's peak (04.4)
