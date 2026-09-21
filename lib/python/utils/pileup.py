"""Tn5 insertion pileup to bigWig, without chrombpnet's two external sorts.

chrombpnet builds its training bigwig like this (reads_to_bigwig.generate_bigwig,
verbatim from 1.0.1)::

    awk (apply shift) | sort -k1,1 | bedtools genomecov -bg -5 -i stdin -g cs
                      | LC_COLLATE=C sort -k1,1 -k2,2n            -> bedGraph
    bedGraphToBigWig bedGraph cs out.bw

Two full external sorts of a text stream with two lines per fragment. Counting
cut sites into an array needs no sort at all: ``np.bincount`` is O(n), and the
bedGraph runs fall out of a single ``np.diff``. The bigWig is then written by
``pybigtools`` (Rust) rather than by piping through a UCSC binary.

**The output is required to be identical, not merely similar.**
``tests/test_pileup.py`` runs chrombpnet's own command string -- extracted from
the installed wheel, not paraphrased -- over the same input and asserts every
interval matches. These conventions were derived from that command's actual
output, not from reading it:

===========================  ==================================================
``+`` strand cut             ``start + plus_delta``
``-`` strand cut             ``end + minus_delta - 1``   (genomecov -5 on a
                             minus feature reports the last covered base)
cut position < 0             **error** -- bedtools rejects the record outright,
                             so silently clamping would diverge
cut position >= chrom length **dropped silently**, as bedtools does
adjacent equal values        **merged** into one interval, as ``-bg`` does
zero coverage                omitted, as ``-bg`` does
chromosome with no reads     contributes no intervals
===========================  ==================================================

This module does not detect the Tn5 shift. That stays chrombpnet's
``auto_shift_detect.compute_shift``, so only the pileup changes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CutSiteError", "cut_positions", "pileup_runs", "write_bigwig"]

# Fragment files can be huge; read them in chunks this size (rows).
CHUNK_ROWS = 5_000_000


class CutSiteError(ValueError):
    """A cut site fell off the start of a chromosome, as bedtools would reject."""


def cut_positions(
    starts: np.ndarray, ends: np.ndarray, plus_delta: int, minus_delta: int
) -> np.ndarray:
    """Tn5 cut sites for fragments, matching ``genomecov -5`` after the shift.

    Each fragment contributes two cuts, because
    ``auto_shift_detect.fragment_to_tagalign_stream`` emits it twice, once per
    strand. The ``-1`` on the minus side is the half-open BED convention: the 5'
    end of a minus-strand feature is its last covered base.
    """
    plus = starts.astype(np.int64) + plus_delta
    minus = ends.astype(np.int64) + minus_delta - 1
    return np.concatenate([plus, minus])


def pileup_runs(cuts: np.ndarray, chrom_length: int):
    """Count cuts per base and return the non-zero runs as (starts, ends, values).

    Adjacent bases with equal counts are merged and zero runs omitted, which is
    what ``bedtools genomecov -bg`` emits.
    """
    if cuts.size and cuts.min() < 0:
        bad = int(cuts.min())
        raise CutSiteError(
            f"cut site at {bad} is before the start of the chromosome. bedtools "
            "rejects such a record outright, so this is an error rather than a "
            "clamp -- the shift is probably wrong for this data."
        )
    # Off the far end is dropped silently, which is what bedtools does.
    cuts = cuts[cuts < chrom_length]
    if cuts.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=np.float32)

    counts = np.bincount(cuts, minlength=chrom_length)

    # Run boundaries: every index where the value changes, plus the two ends.
    change = np.flatnonzero(np.diff(counts)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [counts.size]))
    values = counts[starts]

    keep = values > 0
    return starts[keep], ends[keep], values[keep].astype(np.float32)


def _iter_fragment_chunks(path, chunk_rows: int = CHUNK_ROWS):
    """Yield (chrom, start, end) frames from a fragments TSV, gz or plain."""
    import pandas as pd

    yield from pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64},
        chunksize=chunk_rows,
    )


def collect_cuts(fragments_path, chrom_sizes: dict[str, int], plus_delta, minus_delta):
    """Stream the fragments once, returning {chrom: array of cut positions}.

    Memory is proportional to the number of cut sites (2 per fragment, int64),
    not to the genome, and the file is never materialised in full.
    """
    per_chrom: dict[str, list[np.ndarray]] = {}
    skipped: dict[str, int] = {}
    for chunk in _iter_fragment_chunks(fragments_path):
        for chrom, grp in chunk.groupby("chrom", sort=False):
            if chrom not in chrom_sizes:
                skipped[chrom] = skipped.get(chrom, 0) + len(grp)
                continue
            per_chrom.setdefault(chrom, []).append(
                cut_positions(
                    grp["start"].to_numpy(), grp["end"].to_numpy(), plus_delta, minus_delta
                )
            )
    return (
        {c: np.concatenate(parts) for c, parts in per_chrom.items()},
        skipped,
    )


def write_bigwig(out_path, chrom_sizes: dict[str, int], cuts_by_chrom: dict[str, np.ndarray]):
    """Write the pileup as a bigWig via pybigtools.

    Chromosomes are emitted in ``chrom_sizes`` order and positions ascending,
    which is what the writer requires; getting it wrong raises rather than
    producing a subtly broken file.
    """
    import pybigtools

    def intervals():
        for chrom, length in chrom_sizes.items():
            cuts = cuts_by_chrom.get(chrom)
            if cuts is None or cuts.size == 0:
                continue
            starts, ends, values = pileup_runs(cuts, length)
            for s, e, v in zip(starts.tolist(), ends.tolist(), values.tolist(), strict=True):
                yield chrom, s, e, v

    writer = pybigtools.open(str(out_path), "w")
    try:
        writer.write(dict(chrom_sizes), intervals())
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001  # already closed by write() in some versions
            pass
    return out_path
