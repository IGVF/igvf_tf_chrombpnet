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
from utils import intervals, log, metadata, references  # noqa: E402

logger = log.get_logger(__name__)

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

        kept = intervals.remove_blacklisted(peaks_pr, bl)
        md.add_param("peaks_in", len(peaks_pr))
        md.add_param("peaks_kept", len(kept))
        md.add_param("peaks_dropped", len(peaks_pr) - len(kept))
        logger.info(
            f"{len(peaks_pr) - len(kept)} peak(s) hit the slopped blacklist, {len(kept)} kept"
        )

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
@click.option("--genome", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--chrom-sizes", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Prepared-bigwig directory, passed to --prepared-bigwig later.",
)
@click.option(
    "--num-samples",
    type=int,
    default=10000,
    show_default=True,
    help="Reads sampled for Tn5 shift detection. This is chrombpnet's own default; "
    "changing it makes the bigwig differ from what chrombpnet would have produced.",
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
    import os
    from argparse import Namespace
    from importlib.metadata import version as dist_version

    import chrombpnet.helpers.preprocessing.reads_to_bigwig as reads_to_bigwig

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_dir = metadata_dir or (out / "metadata")

    with metadata.record("prepare_bigwig", out_dir=meta_dir) as md:
        md.add_input("signal", signal_path)
        md.add_input("genome", genome)
        md.add_param("signal_type", signal_type)
        md.add_param("assay", assay)
        md.add_param("num_samples", num_samples)

        args = Namespace(
            input_bam_file=signal_path if signal_type == "bam" else None,
            input_fragment_file=signal_path if signal_type == "fragments" else None,
            input_tagalign_file=signal_path if signal_type == "tagalign" else None,
            data_type=assay,
            genome=genome,
            chrom_sizes=chrom_sizes,
            output_prefix=str(out / "data"),
            plus_shift=None,
            minus_shift=None,
            # 10000 is chrombpnet 1.0.1's own default (parsers.py --num-samples
            # and reads_to_bigwig.py agree). It must match: the shift is detected
            # from this many sampled reads, and a prepared bigwig is only a
            # faithful substitute if it was made the way chrombpnet would have.
            num_samples=num_samples,
            ATAC_ref_path=None,
            DNASE_ref_path=None,
            bsort=False,
            # chrombpnet defaults to None (system /tmp). On a compute node the
            # genome-scale `sort` can overrun /tmp, so prefer SLURM's node-local
            # scratch when it exists. This only changes where sort spills, not
            # the output.
            tmpdir=os.environ.get("TMPDIR") or None,
            no_st=False,
        )
        logger.info(
            "converting %s (%s) -> %s", signal_path, signal_type, out / "data_unstranded.bw"
        )
        reads_to_bigwig.main(args)

        bw = out / "data_unstranded.bw"
        if not bw.is_file():
            raise click.ClickException(f"conversion produced no {bw}")

        sidecar = {
            "signal_path": str(Path(signal_path).resolve()),
            "signal_md5": metadata.md5sum(signal_path),
            "signal_type": signal_type,
            "assay": assay,
            "chrombpnet_version": dist_version("chrombpnet"),
        }
        (out / "prepared_bigwig.json").write_text(_json.dumps(sidecar, indent=2) + "\n")
        md.add_output("bigwig", bw)
        md.add_output("sidecar", out / "prepared_bigwig.json")
        logger.info("-> %s  (pass --prepared-bigwig %s to the training steps)", bw, out)


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
