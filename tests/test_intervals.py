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


# ── peaks live on the same contigs as the signal ────────────────────────────


def test_peaks_on_unlisted_contigs_are_dropped():
    """Handed the main-chromosome chrom.sizes, this is the peak-side filter."""
    peaks = ranges(("chr1", 100, 200), ("chrM", 10, 50), ("chrUn_GL1", 10, 50))
    kept, dropped = intervals.restrict_to_chromosomes(peaks, {"chr1": 1000})
    assert list(kept["Chromosome"]) == ["chr1"]
    assert dropped == 2


def test_restrict_keeps_everything_when_all_contigs_listed():
    peaks = ranges(("chr1", 100, 200), ("chr2", 10, 50))
    kept, dropped = intervals.restrict_to_chromosomes(peaks, {"chr1": 1000, "chr2": 1000})
    assert len(kept) == 2 and dropped == 0


# ── the model window has to fit on the chromosome ───────────────────────────


def test_peak_whose_window_overhangs_the_start_is_dropped():
    """chrombpnet would drop it silently at contribs time; drop it here instead."""
    peaks = ranges(("chr1", 0, 100))  # summit 50, window [50-1057, 50+1057)
    kept, dropped = intervals.drop_windows_off_chromosome(peaks, {"chr1": 100_000}, 2114)
    assert len(kept) == 0 and dropped == 1


def test_peak_whose_window_overhangs_the_end_is_dropped():
    peaks = ranges(("chr1", 9_900, 10_000))  # centre 9950, +1057 > 10000
    kept, dropped = intervals.drop_windows_off_chromosome(peaks, {"chr1": 10_000}, 2114)
    assert len(kept) == 0 and dropped == 1


def test_peak_with_room_on_both_sides_survives():
    peaks = ranges(("chr1", 50_000, 50_200))
    kept, dropped = intervals.drop_windows_off_chromosome(peaks, {"chr1": 100_000}, 2114)
    assert len(kept) == 1 and dropped == 0


def test_window_check_uses_the_same_summit_as_the_narrowpeak_column():
    """If these disagree, the check tests a different base from the one read."""
    peaks = ranges(("chr1", 100, 301))
    assert list(intervals.summit_offsets(peaks)) == list(intervals.to_narrowpeak(peaks)["summit"])


def test_window_boundary_is_inclusive_of_an_exactly_fitting_peak():
    # centre at exactly 1057 -> window starts at 0, which fits
    peaks = ranges(("chr1", 1_007, 1_107))  # summit 50 -> centre 1057
    kept, dropped = intervals.drop_windows_off_chromosome(peaks, {"chr1": 100_000}, 2114)
    assert len(kept) == 1 and dropped == 0


def test_peak_filters_work_on_a_real_bed(tmp_path):
    """pr.read_bed makes Chromosome a Categorical; a plain DataFrame does not.

    The filters compared a mapped Categorical with <=, which raises. Unit tests
    built from DataFrames missed it entirely, so this one goes through the file
    reader the CLI actually uses.
    """
    bed = tmp_path / "p.bed"
    bed.write_text("chr1\t1000\t1200\nchr1\t9000\t9100\nchrM\t10\t50\n")
    peaks = intervals.read_bed(bed)
    cs = {"chr1": 248_956_422}

    kept, off = intervals.restrict_to_chromosomes(peaks, cs)
    assert off == 1 and len(kept) == 2

    fitted, over = intervals.drop_windows_off_chromosome(kept, cs, 2114)
    assert over == 0 and len(fitted) == 2


# ── narrowPeak re-centred on its own summit ───────────────────────────────────


def _macs2(tmp_path, *rows):
    """Write narrowPeak rows (chrom, start, end, qvalue, summit_offset)."""
    path = tmp_path / "peaks.narrowPeak"
    path.write_text(
        "".join(
            f"{c}\t{s}\t{e}\tp{i}\t25\t.\t2.4\t3.0\t{q}\t{off}\n"
            for i, (c, s, e, q, off) in enumerate(rows)
        )
    )
    return path


def test_summit_window_is_summit_minus_half_to_plus_half(tmp_path):
    # summit at 1000 + 250 = 1250 -> [750, 1750): bases 750..1749 = summit-500..summit+499
    peaks = intervals.read_narrowpeak(_macs2(tmp_path, ("chr1", 1000, 2000, 5.0, 250)))
    win, dropped = intervals.summit_windows(peaks, 1000)
    assert (win["Start"].iloc[0], win["End"].iloc[0]) == (750, 1750)
    assert dropped == {"qvalue": 0, "before_chrom_start": 0}


def test_summit_is_the_midpoint_after_recentring(tmp_path):
    peaks = intervals.read_narrowpeak(_macs2(tmp_path, ("chr1", 1000, 2000, 5.0, 731)))
    win, _ = intervals.summit_windows(peaks, 1000)
    np_df = intervals.to_narrowpeak(win)
    assert np_df["Start"].iloc[0] + np_df["summit"].iloc[0] == 1000 + 731
    assert list(intervals.summit_offsets(win)) == [500]


def test_every_summit_of_a_region_gets_its_own_window(tmp_path):
    peaks = intervals.read_narrowpeak(
        _macs2(tmp_path, ("chr1", 1000, 3000, 5.0, 200), ("chr1", 1000, 3000, 5.0, 1800))
    )
    win, _ = intervals.summit_windows(peaks, 1000)
    assert sorted(win["Start"]) == [700, 2300]


def test_qvalue_filter_reads_minus_log10(tmp_path):
    # q <= 0.01 means -log10 q >= 2; 2.0 is kept (inclusive), 1.99 is not.
    peaks = intervals.read_narrowpeak(
        _macs2(tmp_path, ("chr1", 5000, 6000, 2.0, 500), ("chr1", 8000, 9000, 1.99, 500))
    )
    win, dropped = intervals.summit_windows(peaks, 1000, max_qvalue=0.01)
    assert list(win["Start"]) == [5000]
    assert dropped["qvalue"] == 1


def test_window_before_chromosome_start_is_dropped(tmp_path):
    peaks = intervals.read_narrowpeak(_macs2(tmp_path, ("chr1", 0, 400, 5.0, 100)))
    win, dropped = intervals.summit_windows(peaks, 1000)
    assert len(win) == 0 and dropped["before_chrom_start"] == 1


def test_caller_columns_survive_into_the_narrowpeak(tmp_path):
    peaks = intervals.read_narrowpeak(_macs2(tmp_path, ("chr1", 1000, 2000, 7.5, 500)))
    win, _ = intervals.summit_windows(peaks, 1000)
    row = intervals.to_narrowpeak(win).iloc[0]
    assert (row["signal"], row["pvalue"], row["qvalue"]) == (2.4, 3.0, 7.5)


def test_missing_qvalues_cannot_be_filtered(tmp_path):
    peaks = intervals.read_narrowpeak(_macs2(tmp_path, ("chr1", 1000, 2000, -1, 500)))
    with pytest.raises(ValueError, match="no q-values"):
        intervals.summit_windows(peaks, 1000, max_qvalue=0.01)


def test_read_narrowpeak_rejects_a_bed3(tmp_path):
    path = tmp_path / "r.bed"
    path.write_text("chr1\t100\t200\n")
    with pytest.raises(ValueError, match="10-column"):
        intervals.read_narrowpeak(path)
