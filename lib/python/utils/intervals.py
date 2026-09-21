"""Genomic interval operations, replacing the pipeline's bedtools calls.

**The distribution is `pyranges1` and so is the module: `import pyranges1 as pr`.**
`import pyranges` gets you the unrelated 0.x line (PyPI `pyranges` stops at 0.1.4).
PyRanges 1 is backed by `ruranges`, a Rust extension, and a `PyRanges` is a
`pandas.DataFrame` subclass — there is no `.df` attribute, you index it directly.

**This is the only module in `utils` allowed to import `pyranges1`.** It needs
Python >= 3.12, which none of the three cluster conda envs have (chrombpnet and
motif_compendium are 3.10, finemo is 3.11) — hence `envs/preprocess.yml`. Keeping
the import here means `folds`, `palettes`, `plotting` and `regions` stay importable
from the chrombpnet env, where the QC scripts run.

Each function documents the bedtools command it replaces. The equivalences are
pinned by `tests/test_intervals.py`; three of them are not obvious:

- `bedtools intersect -v` requires >= 1bp of overlap, so book-ended intervals are
  kept. `overlap(invert=True)` agrees at its default `slack=0`.
- `bedtools slop` clamps to `[0, chromlen]` and never drops an interval.
  `clip_ranges(chromsizes)` agrees (`remove=False` is the default).
- narrowPeak's summit is an offset from `Start`, floor-divided.
"""

from __future__ import annotations

import pandas as pd
import pyranges1 as pr

from utils import references

# The 10-column narrowPeak layout chrombpnet expects, in PyRanges' column names.
# Kept separate from regions.NARROWPEAK_SCHEMA, which names the same columns the
# way pandas reads the file (chr/start/end); PyRanges requires Chromosome/Start/End.
NARROWPEAK_OUT_COLUMNS = [
    "Chromosome",
    "Start",
    "End",
    "name",
    "score",
    "strand",
    "signal",
    "pvalue",
    "qvalue",
    "summit",
]


def read_bed(path) -> pr.PyRanges:
    """Read a local BED (optionally gzipped) into a PyRanges. Extra columns are kept."""
    return pr.read_bed(str(path))


def read_bed3(source) -> pr.PyRanges:
    """Read the first three BED columns from a path, URL, or ENCODE accession.

    Goes through pandas rather than ``pr.read_bed`` so that a URL works, and
    takes only chrom/start/end because that is all an interval filter needs --
    the ENCODE blacklist's 4th column is a reason string ("Low Mappability").
    """
    src = references.resolve_bed_source(source)
    try:
        df = pd.read_csv(
            src,
            sep="\t",
            header=None,
            comment="#",
            usecols=[0, 1, 2],
            names=["Chromosome", "Start", "End"],
        )
    except OSError as exc:  # network failure, or a path that is not there
        raise OSError(
            f"could not read {references.describe_source(source)}: {exc}\n"
            "If this is an accession and the job runs on a compute node without "
            "outbound network access, pass a local path instead."
        ) from exc
    return pr.PyRanges(df)


def remove_blacklisted(peaks: pr.PyRanges, blacklist: pr.PyRanges) -> pr.PyRanges:
    """Drop every peak overlapping ``blacklist``.

    Replaces ``bedtools intersect -v -a peaks -b blacklist``. Book-ended
    intervals do not overlap and are kept, matching bedtools.
    """
    return peaks.overlap(blacklist, invert=True)


def slop(ranges: pr.PyRanges, bp: int, chromsizes: dict[str, int]) -> pr.PyRanges:
    """Extend every interval by ``bp`` on both sides, clamped to the chromosome.

    Replaces ``bedtools slop -b <bp> -g <chrom.sizes>``. Intervals running off
    either end are clamped to ``[0, chromlen]``, never dropped.
    """
    return ranges.extend_ranges(ext=bp).clip_ranges(chromsizes=chromsizes)


def summit_offsets(peaks: pr.PyRanges):
    """narrowPeak summit: the midpoint, as a floored offset from Start.

    One definition, used both to write the summit column and to decide whether
    a peak's model window fits on the chromosome -- those two must agree or the
    window check tests a different position from the one chrombpnet reads.
    """
    return (peaks["End"] - peaks["Start"]) // 2


def restrict_to_chromosomes(peaks: pr.PyRanges, chromsizes: dict[str, int]):
    """Drop peaks on contigs absent from ``chromsizes``.

    Handed the same main-chromosome chrom.sizes the signal pileup uses, this
    keeps peaks and signal on the same contigs. Without it a peak can sit on a
    scaffold the bigwig does not cover, and chrombpnet trains on a region with
    no data under it.
    """
    keep = peaks["Chromosome"].astype(str).isin(set(chromsizes)).to_numpy()
    return peaks[keep], int((~keep).sum())


def drop_windows_off_chromosome(peaks: pr.PyRanges, chromsizes: dict[str, int], input_window: int):
    """Drop peaks whose model input window runs past either chromosome end.

    chrombpnet extracts ``input_window`` bases centred on ``start + summit``.
    When that window overhangs, ``chrombpnet contribs_bw`` silently drops the
    peak later -- which is why interpretation.interpreted_regions.bed has fewer
    rows than the peak file it was given, and why steps 07/10 have to take
    their regions from that file instead. Dropping them here instead makes the
    peak set stable from step 01 onward, and counts them out loud.
    """
    half = input_window // 2
    centre = (peaks["Start"] + summit_offsets(peaks)).to_numpy()
    # PyRanges keeps Chromosome as a pandas Categorical, and mapping one yields
    # a Categorical that cannot be compared with <=. Go through plain arrays.
    lengths = peaks["Chromosome"].astype(str).map(chromsizes).to_numpy(dtype="int64")
    fits = (centre - half >= 0) & (centre + half <= lengths)
    return peaks[fits], int((~fits).sum())


def to_narrowpeak(peaks: pr.PyRanges) -> pd.DataFrame:
    """Build the 10-column narrowPeak chrombpnet consumes, summit at the midpoint.

    Replaces the awk in the old ``00.1.preprocess_peaks.sh``:

        summit = int(($3-$2)/2); print $1,$2,$3,"peak_"NR,0,".",0,-1,-1,summit

    ``NR`` numbered the rows *after* blacklist filtering, so names are assigned
    over the filtered set, 1-based. The summit is an offset from ``Start``, and
    the division floors.
    """
    out = pd.DataFrame(
        {
            "Chromosome": peaks["Chromosome"].to_numpy(),
            "Start": peaks["Start"].to_numpy(),
            "End": peaks["End"].to_numpy(),
        }
    )
    out["name"] = [f"peak_{i}" for i in range(1, len(out) + 1)]
    out["score"] = 0
    out["strand"] = "."
    out["signal"] = 0
    out["pvalue"] = -1
    out["qvalue"] = -1
    out["summit"] = (out["End"] - out["Start"]) // 2  # see summit_offsets()
    return out[NARROWPEAK_OUT_COLUMNS]


def read_chromsizes(path) -> dict[str, int]:
    """Read a 2-column chrom.sizes TSV into the dict ``slop`` wants."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=["chrom", "size"])
    return dict(zip(df["chrom"], df["size"], strict=True))


def count_in_peaks(fragments: pr.PyRanges, peaks: pr.PyRanges) -> int:
    """Number of fragments overlapping at least one peak (the numerator of FRIP).

    Replaces ``bedtools intersect -a fragments -b peaks -u | wc -l``. ``-u``
    reports each fragment at most once however many peaks it hits, which is what
    ``overlap`` (not ``join_overlaps``) does.
    """
    return len(fragments.overlap(peaks))
