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
peaks/nonpeaks bed files `chrombpnet bias train` wrote into
<output-dir>/auxiliary/. The observed signal is 00.0's prepared bigwig
(--bigwig): training reads it in place (`-bw`), so there is no copy under
auxiliary/ to reuse.

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
    p.add_argument("--chrom-sizes", required=True, help="Chrom sizes file")
    p.add_argument("--fold-json", required=True, help="Fold chr split json (train/valid/test)")
    p.add_argument(
        "--bigwig",
        required=True,
        help="Observed signal the bias model was trained on: 00.0's prepared "
        "preprocessing/signal/data_unstranded.bw",
    )
    p.add_argument("--data-type", default="ATAC", choices=["ATAC", "DNASE"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--stage",
        default="all",
        choices=["all", "gpu", "modisco"],
        help="Which half to run. 'gpu': predictions + DeepLIFT interpretation. "
        "'modisco': TF-MoDISco motif discovery and reports, which use no GPU. "
        "'all' (default) runs both in one process, as before.",
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


def _run_modisco_stage(args, ns, fpx, output_dir, profile_h5, counts_h5):
    """TF-MoDISco + reports. CPU only -- this is why the stages are separable."""
    import chrombpnet.evaluation.modisco.convert_html_to_pdf as convert_html_to_pdf
    import chrombpnet.helpers.generate_reports.make_html_bias as make_html_bias
    from chrombpnet.data import DefaultDataFile, get_default_data_path

    interpret_dir = output_dir / "auxiliary" / "interpret_subsample"
    meme = get_default_data_path(DefaultDataFile.motifs_meme)

    for kind, scores in (("profile", profile_h5), ("counts", counts_h5)):
        if not scores.exists():
            raise FileNotFoundError(f"{scores} missing - run --stage gpu for this fold first.")
        results = interpret_dir / f"{fpx}modisco_results_{kind}_scores.h5"
        report_dir = output_dir / "evaluation" / f"modisco_{kind}"

        if results.exists() and not args.force:
            logger.info("  [modisco] %s exists, skipping motif discovery", results.name)
        else:
            logger.info("  [modisco] motifs (%s)", kind)
            rc = os.system(f"modisco motifs -i {scores} -n 50000 -o {results} -w 500")
            if rc != 0 or not results.exists():
                raise RuntimeError(f"modisco motifs failed for {kind} (exit {rc})")

        if (report_dir / "motifs.html").exists() and not args.force:
            logger.info("  [modisco] %s report exists, skipping", kind)
        else:
            logger.info("  [modisco] report (%s)", kind)
            rc = os.system(f"modisco report -i {results} -o {report_dir}/ -m {meme}")
            if rc != 0:
                raise RuntimeError(f"modisco report failed for {kind} (exit {rc})")

        convert_html_to_pdf.main(
            str(report_dir / "motifs.html"),
            str(output_dir / "evaluation" / f"{fpx}bias_{kind}.pdf"),
        )

    a = copy.deepcopy(ns)
    a.input_dir = str(output_dir)
    a.command = a.cmd_bias
    make_html_bias.main(a)


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
    for f in (bias_model, peaks, nonpeaks, args.genome, args.chrom_sizes, args.fold_json):
        if not Path(f).exists():
            raise FileNotFoundError(
                f"Missing {f} - run chrombpnet bias train for this fold/bias combo first."
            )

    interpret_dir = output_dir / "auxiliary" / "interpret_subsample"
    profile_h5 = interpret_dir / f"{fpx}bias.profile_scores.h5"
    counts_h5 = interpret_dir / f"{fpx}bias.counts_scores.h5"

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

    if args.stage == "all":
        if interpret_dir.exists() and not args.force:
            logger.info("  %s already exists, assuming QC already ran. Skipping.", interpret_dir)
            return
        import chrombpnet.pipelines as pipelines

        pipelines.bias_model_qc(ns)
    elif args.stage == "gpu":
        if profile_h5.exists() and counts_h5.exists() and not args.force:
            logger.info("  contribution scores already exist, skipping interpretation")
        else:
            _run_gpu_stage(args, ns, fpx, output_dir)
    else:
        _run_modisco_stage(args, ns, fpx, output_dir, profile_h5, counts_h5)

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
        metadata.report_peak_rss()  # DeepLIFT (03.2) and TF-MoDISco (03.3) peaks
