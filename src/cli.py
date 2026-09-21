#!/usr/bin/env python3
"""``igvf-tf`` — one command, one subcommand per pipeline tool.

    python src/cli.py --help
    python src/cli.py preprocess-peaks --help

The workflow steps in ``workflows/SLURM/`` call these subcommands. The CLI never
reads the pipeline config: a step sources ``lib/bash/config.sh`` first and passes
the resolved paths as options, so the same command works by hand, from sbatch,
and (later) from Nextflow without a second config reader.

Conversion is partial and deliberately so. Natively Click: the tools written
during the bedtools -> pyranges port. Still standalone argparse scripts, listed
by ``legacy`` below: select_bias_model.py, qc_full_model.py, qc_datasets.py,
average_contrib_scores.py, contribs_to_bigwig.py, predict_and_avg.py,
predict_bias_metrics.py, run_bias_qc.py, motif_compendium.py. They work exactly
as before; converting select_bias_model.py alone is 1152 lines and buys nothing
until someone needs it.
"""

import sys
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

import click  # noqa: E402

from utils import config as cfg  # noqa: E402
from utils import (  # noqa: E402
    intervals,
    log,
    metadata,
    pileup,
    references,
    shift,
)

logger = log.get_logger(__name__)


def _optional_version(dist: str):
    """Version of an installed distribution, or None. Never raises."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v

    try:
        return _v(dist)
    except PackageNotFoundError:
        return None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

LEGACY_SCRIPTS = [
    "average_contrib_scores.py",
    "contribs_to_bigwig.py",
    "motif_compendium.py",
    "predict_and_avg.py",
    "predict_bias_metrics.py",
    "qc_datasets.py",
    "qc_full_model.py",
    "run_bias_qc.py",
    "select_bias_model.py",
]

DEFAULT_INPUT_WINDOW = 2114

verbose_opt = click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
quiet_opt = click.option("-q", "--quiet", is_flag=True, help="Warnings and errors only.")


def _setup_logging(verbose, quiet):
    log.setup(level="WARNING" if quiet else None, verbose=verbose)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name=None, version="0.1.0", prog_name="igvf-tf")
def cli():
    """ChromBPNet pipeline tools for the IGVF TF collaboration."""


# ── preprocess-peaks ──────────────────────────────────────────────────────────


@cli.command("preprocess-peaks")
@click.option(
    "--peaks", required=True, type=click.Path(exists=True, dir_okay=False), help="Input peak BED."
)
@click.option(
    "--blacklist",
    required=True,
    help="ENCODE accession (e.g. ENCFF356LFX), URL, or local BED(.gz).",
)
@click.option(
    "--chrom-sizes",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="2-column chrom.sizes TSV; the slop is clamped to these lengths.",
)
@click.option(
    "--input-window",
    type=int,
    default=DEFAULT_INPUT_WINDOW,
    show_default=True,
    help="ChromBPNet input window (bp). The blacklist is extended by HALF this "
    "on each side, so a peak is dropped when the window the model reads "
    "overlaps a blacklist region.",
)
@click.option(
    "--out-dir", required=True, type=click.Path(file_okay=False), help="Directory for both outputs."
)
@click.option("--prefix", required=True, help="Output basename stem, e.g. igvf3_cardiomyocyte_all.")
@click.option(
    "--metadata-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Run-metadata directory  [default: <out-dir>/../metadata]",
)
@verbose_opt
@quiet_opt
def preprocess_peaks(
    peaks, blacklist, chrom_sizes, input_window, out_dir, prefix, metadata_dir, verbose, quiet
):
    """Blacklist-filter peaks and write chrombpnet's narrowPeak.

    Replaces `bedtools intersect -v` + awk. The blacklist is slopped by half the
    input window first; equivalence to bedtools is pinned by tests/test_intervals.py.
    """
    _setup_logging(verbose, quiet)
    if input_window <= 0:
        raise click.BadParameter("must be positive", param_hint="--input-window")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = metadata_dir or (out_dir.parent / "metadata")

    with metadata.record("preprocess_peaks", dataset=prefix, out_dir=meta_dir) as md:
        md.add_param("input_window", input_window)
        md.add_param("slop_bp", input_window // 2)
        md.add_param("blacklist_source", references.describe_source(blacklist))
        md.add_input("peaks", peaks)
        md.add_input("chrom_sizes", chrom_sizes)
        if Path(blacklist).exists():
            md.add_input("blacklist", blacklist)

        peaks_pr = intervals.read_bed(peaks)
        logger.info(f"{len(peaks_pr)} peaks in from {peaks}")
        logger.info(f"blacklist: {references.describe_source(blacklist)}")
        try:
            bl = intervals.read_bed3(blacklist)
        except OSError as exc:
            raise click.ClickException(str(exc)) from exc
        logger.info(f"{len(bl)} blacklist regions")

        slop_bp = input_window // 2
        chromsizes = intervals.read_chromsizes(chrom_sizes)

        # bedtools slop warns and skips intervals on contigs the genome file does
        # not list; clip_ranges would raise instead, so drop them here with a count.
        known = bl["Chromosome"].isin(chromsizes.keys())
        if not known.all():
            missing = sorted(set(bl.loc[~known, "Chromosome"]))
            logger.warning(
                f"skipping {(~known).sum()} blacklist interval(s) on {len(missing)} "
                f"contig(s) absent from chrom.sizes: {', '.join(missing[:5])}"
                + (" ..." if len(missing) > 5 else "")
            )
            bl = bl[known]

        bl = intervals.slop(bl, slop_bp, chromsizes)
        logger.info(f"blacklist extended +/-{slop_bp}bp (half the {input_window}bp window)")

        # Keep peaks on the same contigs the signal covers. With the
        # main-chromosome chrom.sizes this drops scaffolds and chrM, which the
        # bigwig has no signal for.
        peaks_pr, off_contig = intervals.restrict_to_chromosomes(peaks_pr, chromsizes)
        if off_contig:
            logger.warning("%d peak(s) dropped: on contigs absent from %s", off_contig, chrom_sizes)
        md.add_param("peaks_off_contig", off_contig)

        kept = intervals.remove_blacklisted(peaks_pr, bl)
        md.add_param("peaks_in", len(peaks_pr))
        md.add_param("peaks_kept", len(kept))
        md.add_param("peaks_dropped", len(peaks_pr) - len(kept))
        logger.info(
            f"{len(peaks_pr) - len(kept)} peak(s) hit the slopped blacklist, {len(kept)} kept"
        )

        # chrombpnet would drop these silently at contribs time; do it here so
        # the peak set is stable from this step onward.
        kept, overhanging = intervals.drop_windows_off_chromosome(kept, chromsizes, input_window)
        if overhanging:
            logger.warning(
                "%d peak(s) dropped: their %dbp model window runs off a chromosome end",
                overhanging,
                input_window,
            )
        md.add_param("peaks_window_overhang", overhanging)

        if len(kept) == 0:
            raise click.ClickException(
                "every peak was filtered out. Check that the peak file and the "
                "blacklist use the same chromosome naming (chr1 vs 1)."
            )

        bed_out = out_dir / f"{prefix}_peaks_no_blacklist.bed"
        kept[["Chromosome", "Start", "End"]].to_csv(bed_out, sep="\t", header=False, index=False)
        np_out = out_dir / f"{prefix}_peaks_no_blacklist.narrowPeak"
        intervals.to_narrowpeak(kept).to_csv(np_out, sep="\t", header=False, index=False)

        md.add_output("bed", bed_out)
        md.add_output("narrowpeak", np_out)
        logger.info(f"-> {bed_out}")
        logger.info(f"-> {np_out}")


# ── filter-fragments ──────────────────────────────────────────────────────────

MAIN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]


@cli.command("filter-fragments")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Fragments TSV(.gz).",
)
@click.option("--output", "output_path", required=True, help="Filtered fragments TSV.gz (bgzip).")
@click.option(
    "--chroms",
    multiple=True,
    default=tuple(MAIN_CHROMS),
    help="Chromosomes to keep  [default: chr1-22, chrX, chrY, chrM]",
)
@click.option("--index", is_flag=True, help="Also write a tabix .tbi (input must be sorted).")
@click.option(
    "--metadata-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Run-metadata directory  [default: <output dir>/metadata]",
)
@verbose_opt
@quiet_opt
def filter_fragments(input_path, output_path, chroms, index, metadata_dir, verbose, quiet):
    """Filter an ATAC fragments TSV to the main chromosomes, bgzipped."""
    _setup_logging(verbose, quiet)
    import gzip

    from utils import compression

    meta_dir = metadata_dir or (Path(output_path).parent / "metadata")
    with metadata.record("filter_fragments", out_dir=meta_dir) as md:
        md.add_input("fragments", input_path)
        md.add_param("chroms", ",".join(chroms))
        md.add_param("index", index)

        keep = {c.encode() for c in chroms}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if str(input_path).endswith(".gz") else open

        kept = total = 0
        with opener(input_path, "rb") as fin, compression.open_write(output_path) as fout:
            while chunk := fin.readlines(1 << 22):
                out = []
                for line in chunk:
                    total += 1
                    if line[:1] != b"#" and line.split(b"\t", 1)[0] in keep:
                        out.append(line)
                kept += len(out)
                fout.write(b"".join(out))  # BGZFile has no writelines()

        md.add_param("fragments_in", total)
        md.add_param("fragments_kept", kept)
        md.add_output("fragments", output_path)

        if kept == 0:
            raise click.ClickException(
                f"no fragments kept from {total} line(s). Check the chromosome "
                f"naming in {input_path} (chr1 vs 1)."
            )
        logger.info(f"kept {kept} of {total} fragment(s) -> {output_path}")

        if index:
            tbi = compression.tabix_index(output_path, preset="bed")
            md.add_output("tabix_index", tbi)
            logger.info(f"-> {tbi}")


# ── prepare-bigwig (B) ────────────────────────────────────────────────────────


@cli.command("prepare-bigwig")
@click.option("--signal-path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--signal-type", required=True, type=click.Choice(["fragments", "bam", "tagalign"]))
@click.option("--assay", default="ATAC", type=click.Choice(["ATAC", "DNASE"]))
@click.option(
    "--genome",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Reference FASTA. Needed ONLY to detect the Tn5 shift, which reads "
    "sequence around cut sites. Not needed at all if you give "
    "--plus-shift/--minus-shift: the pileup itself never uses it.",
)
@click.option("--chrom-sizes", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Prepared-bigwig directory, passed to --prepared-bigwig later.",
)
@click.option(
    "--write-filtered",
    default=None,
    type=click.Path(),
    help="Also write the main-chromosome rows here, in the SAME pass (bgzipped "
    "if .gz). Only needed for the fallback where chrombpnet reads the reads "
    "itself; omit it and nothing is rewritten.",
)
@click.option(
    "--plus-shift",
    type=int,
    default=None,
    help="Tn5 shift already present in the reads, plus strand. Give this with "
    "--minus-shift to skip detection entirely -- then no FASTA is needed, "
    "because the pileup does not read sequence.",
)
@click.option(
    "--minus-shift",
    type=int,
    default=None,
    help="As --plus-shift, minus strand.",
)
@click.option(
    "--num-samples",
    type=int,
    default=10000,
    show_default=True,
    help="Reads sampled for Tn5 shift detection (only used when detecting).",
)
@click.option("--metadata-dir", default=None, type=click.Path(file_okay=False))
@verbose_opt
@quiet_opt
def prepare_bigwig(
    signal_path,
    signal_type,
    assay,
    genome,
    chrom_sizes,
    out_dir,
    write_filtered,
    plus_shift,
    minus_shift,
    num_samples,
    metadata_dir,
    verbose,
    quiet,
):
    """Do chrombpnet's reads->bigwig conversion once, on CPU.

    Every `chrombpnet train` / `bias train` starts with this conversion plus an
    enzyme-shift detection pass, unconditionally and on the GPU node. A 5-fold x
    4-factor sweep repeats it 20 times. Run this once on CPU and pass --out-dir to
    src/chrombpnet_train.py as --prepared-bigwig; the GPU jobs then skip it.

    Writes data_unstranded.bw plus a sidecar recording the signal file, its md5,
    the assay and the chrombpnet version — the wrapper refuses to reuse a bigwig
    whose sidecar does not match what it is about to train on.
    """
    _setup_logging(verbose, quiet)
    import json as _json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_dir = metadata_dir or (out / "metadata")

    with metadata.record("prepare_bigwig", out_dir=meta_dir) as md:
        md.add_input("signal", signal_path)
        if genome:
            md.add_input("genome", genome)
        md.add_param("signal_type", signal_type)
        md.add_param("assay", assay)
        md.add_param("num_samples", num_samples)

        # ── 1. shift detection: ours, no chrombpnet ─────────────────────
        # utils.shift (vendored scPrinter) reports the shift ALREADY PRESENT in
        # the reads; shift_deltas then adjusts to chrombpnet's +4/-4 target.
        if plus_shift is None or minus_shift is None:
            if genome is None:
                raise click.UsageError(
                    "--genome is required to detect the Tn5 shift (it reads sequence "
                    "around cut sites). Either pass it, or pass --plus-shift and "
                    "--minus-shift and no FASTA is needed."
                )
            logger.info("detecting the Tn5 shift already present in the reads")
            plus_shift, minus_shift = shift.detect_shift_raw(
                signal_path, genome, signal_type=signal_type
            )
        logger.info("shift in the reads: %+d/%+d", plus_shift, minus_shift)

        # Adjust from the detected shift to chrombpnet's target: +4/-4 (ATAC),
        # 0/+1 (DNASE). NOT scPrinter's +4/-5 -- see utils.shift.
        plus_delta, minus_delta = shift.shift_deltas(plus_shift, minus_shift, assay)
        logger.info("applying delta %+d/%+d", plus_delta, minus_delta)
        md.add_param("plus_shift", plus_shift)
        md.add_param("minus_shift", minus_shift)
        md.add_param("plus_delta", plus_delta)
        md.add_param("minus_delta", minus_delta)

        # ── 2. pileup: ours, not chrombpnet's two external sorts ─────────────
        chromsizes = intervals.read_chromsizes(chrom_sizes)
        logger.info("filtering to the main chromosomes and counting cut sites (one pass)")
        if signal_type == "bam":
            # One cut per mapped read at its own strand's 5' end, as
            # `bedtools bamtobed` would emit; --write-filtered does not apply.
            if write_filtered:
                raise click.UsageError("--write-filtered applies to fragments/tagalign, not BAM")
            cuts, skipped, kept = pileup.collect_cuts_bam(
                signal_path, chromsizes, plus_delta, minus_delta
            )
        else:
            cuts, skipped, kept = pileup.collect_cuts(
                signal_path, chromsizes, plus_delta, minus_delta, write_filtered=write_filtered
            )
        md.add_param("reads_kept", kept)
        if write_filtered:
            md.add_output("filtered_reads", write_filtered)
            logger.info("filtered reads -> %s", write_filtered)
        if skipped:
            logger.warning(
                "skipped %d read(s) on %d contig(s) absent from chrom.sizes: %s",
                sum(skipped.values()),
                len(skipped),
                ", ".join(sorted(skipped)[:5]),
            )
        logger.info("writing %s", out / "data_unstranded.bw")
        pileup.write_bigwig(out / "data_unstranded.bw", chromsizes, cuts)

        bw = out / "data_unstranded.bw"
        if not bw.is_file():
            raise click.ClickException(f"conversion produced no {bw}")

        sidecar = {
            "signal_path": str(Path(signal_path).resolve()),
            "signal_md5": metadata.md5sum(signal_path),
            "signal_type": signal_type,
            "assay": assay,
            # Recorded when chrombpnet happens to be installed, but preprocessing
            # does not import it, so its absence is not an error.
            "chrombpnet_version": _optional_version("chrombpnet"),
            # Which code produced the pileup, so a record says so.
            "pileup": "numpy",
            "shift_detection": "utils.shift (scPrinter)",
            "plus_shift": int(plus_shift),
            "minus_shift": int(minus_shift),
        }
        (out / "prepared_bigwig.json").write_text(_json.dumps(sidecar, indent=2) + "\n")
        md.add_output("bigwig", bw)
        md.add_output("sidecar", out / "prepared_bigwig.json")
        logger.info("-> %s  (pass --prepared-bigwig %s to the training steps)", bw, out)


# ── download-references ───────────────────────────────────────────────────────


@cli.command("download-references")
@click.option(
    "--reference-root",
    default=None,
    type=click.Path(file_okay=False),
    help="Where to install  [default: $REFERENCE_ROOT]",
)
@click.option("--metadata-dir", default=None, type=click.Path(file_okay=False))
@verbose_opt
@quiet_opt
def download_references(dataset, reference_root, metadata_dir, verbose, quiet):
    """Fetch the shared genome, chrom.sizes, blacklist and motif DB.

    Run once per cluster. Idempotent: files already present are left alone, and
    the genome is checked against the md5 IGVF publishes.

    Replaces `cli.py download-references`. Needs no curl and no samtools:
    downloads go through urllib and the FASTA index through pysam. The paths it
    writes come from utils.references, the same module the pipeline reads them
    from, so the two cannot disagree.
    """
    _setup_logging(verbose, quiet)
    if reference_root is None and dataset:
        reference_root = cfg.load(REPO_ROOT / "config" / dataset / "config.yaml").get(
            "reference_root"
        )
    ref = references.layout(reference_root)
    meta_dir = metadata_dir or (Path(ref["REFERENCE_ROOT"]) / "metadata")

    with metadata.record("download_references", out_dir=meta_dir) as md:
        md.add_param("reference_root", ref["REFERENCE_ROOT"])
        md.add_param("genome_accession", ref["genome_accession"])
        md.add_param("blacklist_accession", ref["blacklist_accession"])
        try:
            references.fetch_all(reference_root, log=logger.info)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        for role in (
            "genome_fa",
            "chrom_sizes",
            "chrom_sizes_main",
            "blacklist",
            "ref_db_meme",
        ):
            md.add_output(role, ref[role])
        md.add_output("chrom_sizes_main_sidecar", ref["chrom_sizes_main"] + ".json")
        md.add_param("main_chromosomes", ",".join(references.main_chromosomes()))


# ── qc-signal ─────────────────────────────────────────────────────────────────


@cli.command("qc-signal")
@click.option(
    "--bigwig",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Prepared signal bigwig (00.0 output).",
)
@click.option(
    "--peaks",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filtered narrowPeak (01.0 output).",
)
@click.option(
    "--tss",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="TSS BED for the enrichment metric. Skipped if not given.",
)
@click.option(
    "--input-window",
    type=int,
    default=2114,
    show_default=True,
    help="Profile spans +/- half this around peak summits.",
)
@click.option(
    "--min-insertions",
    type=int,
    default=20_000_000,
    show_default=True,
    help="Depth below which ChromBPNet is unlikely to train well.",
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False))
@click.option("--prefix", required=True)
@click.option("--metadata-dir", default=None, type=click.Path(file_okay=False))
@verbose_opt
@quiet_opt
def qc_signal(
    bigwig, peaks, tss, input_window, min_insertions, out_dir, prefix, metadata_dir, verbose, quiet
):
    """QC the signal and peaks about to be handed to ChromBPNet.

    Reads the two artifacts training will actually use -- the prepared bigwig
    and the filtered narrowPeak -- so every number describes what the model
    sees, after all filtering.

    Writes <prefix>_signal_qc.json (all metrics), <prefix>_signal_qc.tsv (the
    flat ones, for DuckDB alongside the run metadata) and two profile plots.
    """
    _setup_logging(verbose, quiet)
    import json as _json

    import pandas as pd

    from utils import plotting, qc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_dir = metadata_dir or (out.parent / "metadata")

    with metadata.record("qc_signal", dataset=prefix, out_dir=meta_dir) as md:
        md.add_input("bigwig", bigwig)
        md.add_input("peaks", peaks)
        if tss:
            md.add_input("tss", tss)

        peak_rows = pd.read_csv(peaks, sep="\t", header=None, dtype={0: str}).values.tolist()
        logger.info("%d peaks", len(peak_rows))

        metrics = {"dataset": prefix}
        metrics |= qc.signal_summary(bigwig)
        metrics |= qc.peak_width_summary(peak_rows)
        metrics |= qc.peak_signal_distribution(bigwig, peak_rows)

        # Fraction of all insertions that land in peaks -- the FRiP of the
        # signal chrombpnet sees, rather than of the original fragments.
        if metrics.get("total_insertions"):
            metrics["frac_insertions_in_peaks"] = (
                metrics["insertions_in_peaks"] / metrics["total_insertions"]
            )
        metrics["enough_depth_for_chrombpnet"] = bool(
            metrics.get("total_insertions", 0) >= min_insertions
        )

        offsets, profile, used = qc.aggregate_profile(bigwig, peak_rows, flank=input_window // 2)
        metrics["n_peaks_profiled"] = used
        metrics["peak_profile_centre_over_flank"] = (
            float(profile[len(profile) // 2] / profile[:100].mean())
            if used and profile[:100].mean() > 0
            else None
        )

        tss_metrics = {}
        if tss:
            tss_rows = pd.read_csv(tss, sep="\t", header=None, dtype={0: str}).values.tolist()
            tss_metrics = qc.tss_enrichment(bigwig, tss_rows)
            metrics["tss_enrichment"] = tss_metrics.get("tss_enrichment")
            metrics["n_tss_used"] = tss_metrics.get("n_tss_used")
        else:
            logger.warning("no --tss given; skipping TSS enrichment")

        # ── report ────────────────────────────────────────────────────────
        for key in (
            "total_insertions",
            "n_peaks",
            "frac_insertions_in_peaks",
            "frac_peaks_zero_signal",
            "peak_profile_centre_over_flank",
            "tss_enrichment",
        ):
            if metrics.get(key) is not None:
                logger.info("  %-32s %s", key, metrics[key])
        if not metrics["enough_depth_for_chrombpnet"]:
            logger.warning(
                "only %.1fM insertions; ChromBPNet usually needs >= %.0fM",
                metrics.get("total_insertions", 0) / 1e6,
                min_insertions / 1e6,
            )
        if metrics.get("frac_peaks_zero_signal", 0) > 0.01:
            logger.warning(
                "%.1f%% of peaks have NO signal under them -- peaks and signal may "
                "not come from the same sample",
                100 * metrics["frac_peaks_zero_signal"],
            )

        json_out = out / f"{prefix}_signal_qc.json"
        json_out.write_text(_json.dumps({**metrics, "tss": tss_metrics}, indent=2) + "\n")
        tsv_out = out / f"{prefix}_signal_qc.tsv"
        flat = {k: v for k, v in metrics.items() if not isinstance(v, list | dict)}
        pd.DataFrame([flat]).to_csv(tsv_out, sep="\t", index=False)
        md.add_output("qc_json", json_out)
        md.add_output("qc_tsv", tsv_out)
        for k, v in flat.items():
            md.add_param(k, v)

        # ── plots ─────────────────────────────────────────────────────────
        import matplotlib.pyplot as plt

        plotting.apply_style(font_size=10)
        from utils.palettes import OKABE_ITO

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(offsets, profile, color=OKABE_ITO["blue"], lw=1.2)
        ax.set_xlabel("distance from peak summit (bp)")
        ax.set_ylabel("mean insertions per base")
        ax.set_title(f"{prefix}: signal at peaks (n={used})")
        plotting.save_fig(fig, out / f"{prefix}_profile_peaks")
        plt.close(fig)
        md.add_output("profile_peaks", out / f"{prefix}_profile_peaks.pdf")

        if tss_metrics.get("profile"):
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(
                tss_metrics["profile_offsets"],
                tss_metrics["profile"],
                color=OKABE_ITO["vermillion"],
                lw=1.2,
            )
            ax.axhline(1.0, color=OKABE_ITO["black"], lw=0.6, ls="--")
            ax.set_xlabel("distance from TSS (bp)")
            ax.set_ylabel("enrichment over flanks")
            ax.set_title(f"{prefix}: TSS enrichment = {tss_metrics['tss_enrichment']:.1f}")
            plotting.save_fig(fig, out / f"{prefix}_profile_tss")
            plt.close(fig)
            md.add_output("profile_tss", out / f"{prefix}_profile_tss.pdf")


# ── config ────────────────────────────────────────────────────────────────────


@cli.group("config")
def config_group():
    """Inspect a dataset's config.yaml."""


def _config_path(dataset, path):
    if path:
        return Path(path)
    if not dataset:
        raise click.UsageError("give --dataset NAME or --path config.yaml")
    return REPO_ROOT / "config" / dataset / "config.yaml"


@config_group.command("show")
@click.option("--dataset", default=None, help="Dataset name under config/.")
@click.option("--path", default=None, type=click.Path(), help="Explicit config.yaml.")
def config_show(dataset, path):
    """Print the parsed config as JSON."""
    import json as _json

    click.echo(_json.dumps(cfg.load(_config_path(dataset, path)), indent=2))


@config_group.command("export")
@click.option("--dataset", default=None, help="Dataset name under config/.")
@click.option("--path", default=None, type=click.Path(), help="Explicit config.yaml.")
def config_export(dataset, path):
    """Print the config as shell assignments (what lib/bash/config.sh evals)."""
    click.echo(cfg.to_shell(cfg.load(_config_path(dataset, path))))


@config_group.command("validate")
@click.option("--dataset", default=None, help="Dataset name under config/.")
@click.option("--path", default=None, type=click.Path(), help="Explicit config.yaml.")
def config_validate(dataset, path):
    """Check a config parses, renders to shell, and has the required keys."""
    target = _config_path(dataset, path)
    try:
        parsed = cfg.load(target)
        cfg.to_shell(parsed)
    except cfg.ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    required = [
        "datasets",
        "bias_dataset",
        "peak_type",
        "folds",
        "regions",
        "signal_path",
        "signal_type",
        "output_dir",
    ]
    missing = [k for k in required if not parsed.get(k)]
    if missing:
        raise click.ClickException(f"{target}: missing required key(s): {', '.join(missing)}")

    if len(parsed.get("bias_factors") or []) != len(parsed.get("bias_suffixes_sweep") or []):
        raise click.ClickException(
            f"{target}: bias_factors and bias_suffixes_sweep must be the same length "
            "(they are positional)"
        )
    if parsed.get("signal_type") not in {"fragments", "bam", "tagalign"}:
        raise click.ClickException(
            f"{target}: signal_type must be fragments, bam or tagalign "
            f"(got {parsed.get('signal_type')!r}). chrombpnet cannot train from a bigwig."
        )
    if parsed.get("assay", "ATAC") not in {"ATAC", "DNASE"}:
        raise click.ClickException(f"{target}: assay must be ATAC or DNASE")

    sweep_folds = parsed.get("bias_sweep_folds") or parsed["folds"]
    n_tasks = len(sweep_folds) * len(parsed.get("bias_factors") or [1])
    click.echo(f"{target}: OK")
    click.echo(f"  signal     : {parsed['signal_type']} -> {parsed['signal_path']}")
    click.echo(f"  output     : {parsed['output_dir']}")
    click.echo(f"  folds      : {len(parsed['folds'])}")
    click.echo(
        f"  03.0 sweep : {len(sweep_folds)} fold(s) x "
        f"{len(parsed.get('bias_factors') or [])} factor(s)  ->  --array=0-{n_tasks - 1}"
    )
    if parsed.get("signal_type") == "bam":
        click.echo(
            "  note       : run 00.0.prepare_signal.sh to normalise the BAM to tagAlign once on CPU"
        )


# ── legacy ────────────────────────────────────────────────────────────────────


@cli.command("legacy")
def legacy():
    """List the tools that are still standalone argparse scripts."""
    click.echo("Not yet Click subcommands — run these directly:\n")
    for name in LEGACY_SCRIPTS:
        click.echo(f"  python {SRC.relative_to(REPO_ROOT)}/{name} --help")


if __name__ == "__main__":
    cli()
