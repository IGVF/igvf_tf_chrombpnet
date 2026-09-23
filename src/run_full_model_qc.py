#!/usr/bin/env python3
"""
run_full_model_qc.py
Per-fold interpretation QC for a trained ChromBPNet full model: DeepLIFT
contribution scores on a 30K peak subsample, TF-MoDISco on them, the motif
report, and chrombpnet's pipeline HTML report -- the part of
`chrombpnet pipeline` that 04.0 stops short of.

Why a separate script: `chrombpnet pipeline` runs this inside the training
job, so a GPU allocation sat idle through TF-MoDISco, which is CPU-only and
the long pole -- the same split 03.2/03.3 make for the bias model. 04.0 now
stops `pipeline` just before interpretation (chrombpnet_train.py
--stop-before-interpretation); this script does the rest, in two stages:

  --stage gpu      DeepLIFT on the profile head (04.4)
  --stage modisco  TF-MoDISco + report + PDF + pipeline HTML report (04.5)

It reproduces `chrombpnet.pipelines.chrombpnet_train_pipeline` from its
interpretation step onward, with chrombpnet's own functions and arguments:
the 30K subsample (seed 1234) of auxiliary/filtered.peaks.bed, profile scores
only (the pipeline's counts run is commented out upstream), `modisco motifs
-n 50000 -w 500`, `modisco report` against chrombpnet's motif DB, and
make_html in "pipeline" mode. Each stage skips work whose outputs exist, so a
failed TF-MoDISco re-runs without redoing DeepLIFT.

Input  (written by 04.0 under <model-dir>):
  models/<fpx>chrombpnet_nobias.h5, auxiliary/<fpx>filtered.peaks.bed,
  evaluation/<fpx>chrombpnet_nobias_max_bias_response.txt (the footprints)
Output (under <model-dir>):
  auxiliary/<fpx>30K_subsample_peaks.bed
  auxiliary/interpret_subsample/<fpx>chrombpnet_nobias.profile_scores.h5   (gpu)
  auxiliary/interpret_subsample/<fpx>modisco_results_profile_scores.h5     (modisco)
  evaluation/modisco_profile/motifs.html, evaluation/<fpx>chrombpnet_nobias_profile.pdf
  the pipeline-mode overall report (make_html)

Usage:
  python run_full_model_qc.py --stage gpu \\
      --model-dir results/full_models/d0_all_fold_0 --genome genome/hg38.fa
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
    p = argparse.ArgumentParser(description="Full-model interpretation QC (DeepLIFT / TF-MoDISco).")
    p.add_argument("--stage", required=True, choices=["gpu", "modisco"])
    p.add_argument("--model-dir", required=True, help="04.0's output dir for one dataset x fold")
    p.add_argument("--genome", required=True, help="Reference genome fasta")
    p.add_argument("--file-prefix", default="", help="chrombpnet file prefix; 04.0 uses none")
    p.add_argument("--data-type", default="ATAC", choices=["ATAC", "DNASE"])
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
        "modisco": interp / f"{fpx}modisco_results_profile_scores.h5",
        "report_dir": model_dir / "evaluation" / "modisco_profile",
        "pdf": model_dir / "evaluation" / f"{fpx}chrombpnet_nobias_profile.pdf",
    }


def run_gpu(args, p: dict, fpx: str) -> None:
    """DeepLIFT on the profile head, as chrombpnet_train_pipeline does."""
    if p["profile_scores"].exists() and not args.force:
        logger.info("  %s exists, skipping interpretation", p["profile_scores"].name)
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
        profile_or_counts=["profile"],  # pipelines.py: counts is commented out upstream
        debug_chr=None,
    )
    logger.info("  [gpu] DeepLIFT interpretation (profile) on %s", p["subsample"].name)
    interpret.main(a)


def run_modisco(args, p: dict, model_dir: Path) -> None:
    """TF-MoDISco + reports, as chrombpnet_train_pipeline does. CPU only."""
    import chrombpnet.evaluation.modisco.convert_html_to_pdf as convert_html_to_pdf
    import chrombpnet.helpers.generate_reports.make_html as make_html
    from chrombpnet.data import DefaultDataFile, get_default_data_path

    if not p["profile_scores"].exists():
        raise FileNotFoundError(f"{p['profile_scores']} missing - run --stage gpu (04.4) first.")
    meme = get_default_data_path(DefaultDataFile.motifs_meme)

    if p["modisco"].exists() and not args.force:
        logger.info("  [modisco] %s exists, skipping motif discovery", p["modisco"].name)
    else:
        logger.info("  [modisco] motifs (profile)")
        rc = os.system(f"modisco motifs -i {p['profile_scores']} -n 50000 -o {p['modisco']} -w 500")
        if rc != 0 or not p["modisco"].exists():
            raise RuntimeError(f"modisco motifs failed (exit {rc})")

    if (p["report_dir"] / "motifs.html").exists() and not args.force:
        logger.info("  [modisco] report exists, skipping")
    else:
        logger.info("  [modisco] report (profile)")
        rc = os.system(f"modisco report -i {p['modisco']} -o {p['report_dir']}/ -m {meme}")
        if rc != 0:
            raise RuntimeError(f"modisco report failed (exit {rc})")

    convert_html_to_pdf.main(str(p["report_dir"] / "motifs.html"), str(p["pdf"]))

    # The pipeline-mode report: it needs the footprints (04.0) and the motif
    # report (above), which is why it can only be written here.
    report = argparse.Namespace(
        input_dir=str(model_dir),
        command="pipeline",
        data_type=args.data_type,
        file_prefix=args.file_prefix or None,
        html_prefix="./",
    )
    make_html.main(copy.deepcopy(report))


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
    if args.stage == "gpu":
        run_gpu(args, p, fpx)
    else:
        run_modisco(args, p, model_dir)
    logger.info("  stage '%s' complete -> %s", args.stage, model_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        metadata.report_peak_rss()  # DeepLIFT (04.4) and TF-MoDISco (04.5) peaks
