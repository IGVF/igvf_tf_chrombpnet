"""Pin utils.intervals to the bedtools behaviour it replaces.

The pipeline ran on bedtools for its whole history, so the question these tests
answer is not "is pyranges correct" but "does pyranges do what bedtools did".
Every expected value below is what the replaced bedtools command produces.

Run with `pixi run test`. Needs the `qc` environment (pyranges1, Python >= 3.12).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

pr = pytest.importorskip("pyranges1", reason="needs the pixi 'qc' environment")

from utils import intervals, references  # noqa: E402


def ranges(*rows) -> pr.PyRanges:
    """Build a PyRanges from (chrom, start, end) tuples."""
    return pr.PyRanges(pd.DataFrame(list(rows), columns=["Chromosome", "Start", "End"]))


# ── remove_blacklisted  ==  bedtools intersect -v ─────────────────────────────


def test_overlapping_peak_is_dropped():
    peaks = ranges(("chr1", 100, 200), ("chr1", 500, 600), ("chr1", 900, 1000))
    kept = intervals.remove_blacklisted(peaks, ranges(("chr1", 550, 560)))
    assert list(kept["Start"]) == [100, 900]


def test_bookended_peak_is_kept():
    """bedtools -v needs >=1bp of overlap, so touching intervals survive."""
    peaks = ranges(("chr1", 100, 200), ("chr1", 300, 400))
    kept = intervals.remove_blacklisted(peaks, ranges(("chr1", 200, 300)))
    assert len(kept) == 2


def test_single_base_overlap_is_enough_to_drop():
    peaks = ranges(("chr1", 100, 200))
    kept = intervals.remove_blacklisted(peaks, ranges(("chr1", 199, 250)))
    assert len(kept) == 0


def test_blacklist_on_another_chromosome_is_ignored():
    peaks = ranges(("chr1", 100, 200))
    kept = intervals.remove_blacklisted(peaks, ranges(("chr2", 100, 200)))
    assert len(kept) == 1


# ── slop  ==  bedtools slop -b ────────────────────────────────────────────────


def test_slop_extends_both_sides():
    out = intervals.slop(ranges(("chr1", 5000, 5100)), 1057, {"chr1": 100_000})
    assert (out["Start"].iloc[0], out["End"].iloc[0]) == (3943, 6157)


def test_slop_clamps_at_zero_without_dropping():
    """A negative start becomes 0; bedtools never drops the interval."""
    out = intervals.slop(ranges(("chr1", 550, 560)), 1057, {"chr1": 100_000})
    assert len(out) == 1
    assert out["Start"].iloc[0] == 0


def test_slop_clamps_at_chromosome_end():
    out = intervals.slop(ranges(("chr1", 990, 1000)), 1057, {"chr1": 1200})
    assert (out["Start"].iloc[0], out["End"].iloc[0]) == (0, 1200)


# ── to_narrowpeak  ==  the awk it replaces ────────────────────────────────────


def test_narrowpeak_has_ten_columns_in_order():
    np_df = intervals.to_narrowpeak(ranges(("chr1", 100, 300)))
    assert list(np_df.columns) == intervals.NARROWPEAK_OUT_COLUMNS
    assert len(np_df.columns) == 10


def test_narrowpeak_summit_is_floored_offset_from_start():
    """awk did int(($3-$2)/2): an offset, not a coordinate, and it floors."""
    np_df = intervals.to_narrowpeak(ranges(("chr1", 100, 300), ("chr1", 500, 601)))
    assert list(np_df["summit"]) == [100, 50]


def test_narrowpeak_names_are_numbered_over_the_filtered_set():
    """awk's NR ran after filtering, so names are 1..N of what survived."""
    peaks = ranges(("chr1", 100, 200), ("chr1", 500, 600), ("chr1", 900, 1000))
    kept = intervals.remove_blacklisted(peaks, ranges(("chr1", 550, 560)))
    np_df = intervals.to_narrowpeak(kept)
    assert list(np_df["name"]) == ["peak_1", "peak_2"]


def test_narrowpeak_constant_columns_match_awk():
    np_df = intervals.to_narrowpeak(ranges(("chr1", 100, 300)))
    row = np_df.iloc[0]
    assert (row["score"], row["strand"], row["signal"], row["pvalue"], row["qvalue"]) == (
        0,
        ".",
        0,
        -1,
        -1,
    )


# ── count_in_peaks  ==  bedtools intersect -u | wc -l ─────────────────────────


def test_fragment_hitting_two_peaks_counts_once():
    """-u reports each -a record at most once."""
    frags = ranges(("chr1", 150, 650))
    peaks = ranges(("chr1", 100, 200), ("chr1", 600, 700))
    assert intervals.count_in_peaks(frags, peaks) == 1


def test_count_in_peaks_excludes_non_overlapping():
    frags = ranges(("chr1", 100, 200), ("chr1", 5000, 5100))
    assert intervals.count_in_peaks(frags, ranges(("chr1", 150, 160))) == 1


# ── read_chromsizes ───────────────────────────────────────────────────────────


def test_read_chromsizes_ignores_extra_columns(tmp_path):
    p = tmp_path / "chrom.sizes"
    p.write_text("chr1\t248956422\textra\nchr2\t242193529\textra\n")
    assert intervals.read_chromsizes(p) == {"chr1": 248956422, "chr2": 242193529}


# ── references: accession / URL / path resolution ─────────────────────────────


@pytest.mark.parametrize("acc", ["ENCFF356LFX", "ENCFF001TDO"])
def test_encode_accessions_are_recognised(acc):
    assert references.is_encode_accession(acc)


@pytest.mark.parametrize(
    "spec",
    [
        "/oak/.../blacklist.bed.gz",
        "blacklist.bed.gz",
        "https://example.org/x.bed.gz",
        "ENCFF356",  # too short
        "ENCSR356LFX",  # experiment, not a file
        "encff356lfx",  # lowercase is not an accession
    ],
)
def test_non_accessions_are_passed_through(spec):
    assert not references.is_encode_accession(spec)
    assert references.resolve_bed_source(spec) == spec


def test_accession_resolves_to_encode_download_url():
    assert references.resolve_bed_source("ENCFF356LFX") == (
        "https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz"
    )


def test_encode_file_url_rejects_a_path():
    with pytest.raises(ValueError):
        references.encode_file_url("/tmp/blacklist.bed.gz")


# ── the --input-window contract ───────────────────────────────────────────────


def test_window_slop_is_half_the_window():
    """--input-window 2114 must reproduce the old `bedtools slop -b 1057`."""
    bl = ranges(("chr1", 10_000, 10_100))
    by_window = intervals.slop(bl, 2114 // 2, {"chr1": 100_000})
    by_hand = intervals.slop(bl, 1057, {"chr1": 100_000})
    assert by_window["Start"].iloc[0] == by_hand["Start"].iloc[0] == 8943
    assert by_window["End"].iloc[0] == by_hand["End"].iloc[0] == 11_157


def test_peak_outside_interval_but_inside_window_is_dropped():
    """The point of slopping: the peak does not touch the blacklist, its window does."""
    peaks = ranges(("chr1", 10_000, 10_200))
    bl = ranges(("chr1", 10_500, 10_600))  # 300bp away, no direct overlap
    assert len(intervals.remove_blacklisted(peaks, bl)) == 1
    slopped = intervals.slop(bl, 2114 // 2, {"chr1": 100_000})
    assert len(intervals.remove_blacklisted(peaks, slopped)) == 0


def test_peak_beyond_the_window_survives():
    peaks = ranges(("chr1", 10_000, 10_200))
    bl = ranges(("chr1", 50_000, 50_100))
    slopped = intervals.slop(bl, 2114 // 2, {"chr1": 100_000})
    assert len(intervals.remove_blacklisted(peaks, slopped)) == 1
