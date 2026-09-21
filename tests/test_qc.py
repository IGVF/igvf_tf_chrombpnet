"""QC metrics must detect what they claim to detect.

Built on synthetic signal where the right answer is known by construction: a
metric that cannot tell enriched data from flat data is worse than no metric,
because it will be believed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))

pytest.importorskip("pyBigWig", reason="needs the pixi 'qc' environment")

from utils import pileup, qc  # noqa: E402

CS = {"chr1": 60_000}


def make_bigwig(path, cuts):
    pileup.write_bigwig(path, CS, {"chr1": np.asarray(cuts, dtype=np.int64)})
    return path


def peaks_at(centres, half=200):
    """narrowPeak rows centred on each position (summit = midpoint)."""
    return [
        ["chr1", c - half, c + half, f"p{i}", 0, ".", 0, -1, -1, half]
        for i, c in enumerate(centres)
    ]


def bed_at(positions):
    return [["chr1", p, p + 1, ".", 0, "+"] for p in positions]


# ── aggregate profile ────────────────────────────────────────────────────────


def test_profile_peaks_at_the_centre_for_enriched_signal(tmp_path):
    centres = list(range(5_000, 50_000, 5_000))
    cuts = [c + d for c in centres for d in range(-20, 20)]  # tight pile at each
    bw = make_bigwig(tmp_path / "s.bw", cuts)
    offsets, profile, used = qc.aggregate_profile(bw, peaks_at(centres), flank=500)
    assert used == len(centres)
    assert profile.argmax() == pytest.approx(len(profile) // 2, abs=25)
    assert profile.max() > 10 * profile[:50].mean()


def test_profile_is_flat_for_uniform_signal(tmp_path):
    rng = np.random.default_rng(0)
    centres = list(range(5_000, 50_000, 5_000))
    cuts = rng.integers(0, 60_000, size=40_000)
    bw = make_bigwig(tmp_path / "s.bw", cuts)
    _o, profile, _u = qc.aggregate_profile(bw, peaks_at(centres), flank=500)
    # no central spike: the max is not far above the edges
    assert profile.max() < 3 * profile[:50].mean()


def test_profile_subsamples_but_still_reports_how_many(tmp_path):
    centres = list(range(5_000, 50_000, 100))
    bw = make_bigwig(tmp_path / "s.bw", [c for c in centres])
    _o, _p, used = qc.aggregate_profile(bw, peaks_at(centres), flank=100, max_regions=50)
    assert used == 50


def test_regions_whose_window_leaves_the_chromosome_are_skipped(tmp_path):
    bw = make_bigwig(tmp_path / "s.bw", [100, 200])
    _o, _p, used = qc.aggregate_profile(bw, peaks_at([50]), flank=500)
    assert used == 0


# ── TSS enrichment ───────────────────────────────────────────────────────────


def test_tss_enrichment_is_high_when_signal_sits_at_tss(tmp_path):
    """Enriched at TSS, on top of a uniform background -- as real data is.

    Without background the flanks are empty and the metric correctly refuses
    to divide by zero, which is a different case (tested below).
    """
    rng = np.random.default_rng(7)
    tss = list(range(5_000, 50_000, 5_000))
    background = rng.integers(0, 60_000, size=20_000).tolist()
    enriched = [t + d for t in tss for d in range(-50, 50) for _ in range(5)]
    bw = make_bigwig(tmp_path / "s.bw", sorted(background + enriched))
    m = qc.tss_enrichment(bw, bed_at(tss), flank=2000, flank_width=100)
    assert m["tss_enrichment"] > 10
    assert m["n_tss_used"] == len(tss)


def test_tss_enrichment_is_about_one_for_uniform_signal(tmp_path):
    rng = np.random.default_rng(1)
    tss = list(range(5_000, 50_000, 5_000))
    bw = make_bigwig(tmp_path / "s.bw", rng.integers(0, 60_000, size=200_000))
    m = qc.tss_enrichment(bw, bed_at(tss), flank=2000, flank_width=100)
    assert 0.5 < m["tss_enrichment"] < 2.0, m["tss_enrichment"]


def test_tss_enrichment_reports_none_rather_than_dividing_by_zero(tmp_path):
    """All signal at the centre, nothing in the flanks -> background is 0."""
    tss = [25_000]
    bw = make_bigwig(tmp_path / "s.bw", [25_000] * 10)
    m = qc.tss_enrichment(bw, bed_at(tss), flank=2000, flank_width=100)
    assert m["tss_enrichment"] is None and "note" in m


# ── per-peak signal distribution ─────────────────────────────────────────────


def test_zero_signal_peaks_are_counted(tmp_path):
    """The metric that catches peaks and signal coming from different samples."""
    bw = make_bigwig(tmp_path / "s.bw", [10_000] * 5)
    peaks = peaks_at([10_000, 20_000, 30_000, 40_000])
    m = qc.peak_signal_distribution(bw, peaks)
    assert m["n_peaks"] == 4
    assert m["frac_peaks_zero_signal"] == pytest.approx(0.75)
    assert m["insertions_in_peaks"] == 5.0


def test_signal_quantiles_are_ordered(tmp_path):
    rng = np.random.default_rng(2)
    centres = list(range(5_000, 50_000, 1_000))
    cuts = np.concatenate([np.full(int(rng.integers(1, 50)), c) for c in centres])
    bw = make_bigwig(tmp_path / "s.bw", cuts)
    m = qc.peak_signal_distribution(bw, peaks_at(centres))
    assert (
        m["signal_per_peak_min"]
        <= m["signal_per_peak_median"]
        <= m["signal_per_peak_q90"]
        <= m["signal_per_peak_max"]
    )


# ── summaries ────────────────────────────────────────────────────────────────


def test_signal_summary_totals_match_the_cuts(tmp_path):
    bw = make_bigwig(tmp_path / "s.bw", [100, 100, 200, 300])
    m = qc.signal_summary(bw)
    assert m["total_insertions"] == 4.0  # two at 100, one each at 200/300
    assert m["covered_bases"] == 3
    assert m["max_per_base"] == 2.0


def test_peak_width_summary():
    m = qc.peak_width_summary([["chr1", 0, 100], ["chr1", 0, 300]])
    assert m["peak_width_min"] == 100 and m["peak_width_max"] == 300
    assert m["peak_bases_total"] == 400


# ── comparative: peaks vs GC-matched background ──────────────────────────────


def negatives_at(centres, width=2114):
    """chrombpnet's negatives layout: 10 columns, summit in the last."""
    half = width // 2
    return [
        ["chr1", c - half, c + half, ".", ".", ".", ".", ".", ".", half] for c in centres
    ]


@pytest.fixture
def separable(tmp_path):
    """Signal piled on the peak centres and nothing on the background ones."""
    peak_centres = list(range(6_000, 26_000, 2_000))
    neg_centres = list(range(31_000, 51_000, 2_000))
    cuts = np.repeat(peak_centres, 500)
    bw = make_bigwig(tmp_path / "sep.bw", cuts)
    return bw, peaks_at(peak_centres), negatives_at(neg_centres)


def test_auroc_is_one_when_signal_is_only_at_peaks(separable):
    bw, peaks, negs = separable
    m, _pos, _neg = qc.peak_vs_nonpeak_signal(bw, peaks, negs)
    assert m["auroc_peaks_vs_nonpeaks"] == 1.0
    assert m["frac_nonpeaks_zero_signal"] == 1.0


def test_auroc_is_zero_when_the_classes_are_swapped(separable):
    """The mirror case -- proof the metric is directional, not just non-0.5."""
    bw, peaks, negs = separable
    m, _pos, _neg = qc.peak_vs_nonpeak_signal(bw, negs, peaks)
    assert m["auroc_peaks_vs_nonpeaks"] == 0.0


def test_auroc_is_half_when_the_two_sets_are_identical(separable):
    """All values tie against themselves, and ties must count as half."""
    bw, peaks, _negs = separable
    m, _pos, _neg = qc.peak_vs_nonpeak_signal(bw, peaks, peaks)
    assert m["auroc_peaks_vs_nonpeaks"] == pytest.approx(0.5)


def test_auroc_is_near_half_when_background_is_as_open_as_peaks(tmp_path):
    """A dataset not worth training on: the metric has to say so."""
    peak_centres = list(range(6_000, 26_000, 2_000))
    neg_centres = list(range(31_000, 51_000, 2_000))
    cuts = np.repeat(peak_centres + neg_centres, 500)
    bw = make_bigwig(tmp_path / "flat.bw", cuts)
    m, _pos, _neg = qc.peak_vs_nonpeak_signal(
        bw, peaks_at(peak_centres), negatives_at(neg_centres)
    )
    assert m["auroc_peaks_vs_nonpeaks"] == pytest.approx(0.5, abs=0.05)
    assert m["signal_enrichment_peak_over_nonpeak"] == pytest.approx(1.0, abs=0.1)


def test_comparison_is_unaffected_by_called_peak_width(tmp_path):
    """The reason window_totals exists.

    Two peaks with identical signal DENSITY but very different called widths
    must score identically. Summing over each peak's own extent -- as
    peak_signal_distribution does -- would rank the wide one higher on width
    alone, and the negatives are all one fixed width, so that comparison would
    be decided by peak-caller settings rather than by signal.

    The signal is spread uniformly rather than piled on the centre: a single
    central spike would total the same in any window containing it, so it
    could not tell a fixed window from a per-region one.
    """
    centres = [10_000, 30_000]
    spread = np.arange(-950, 950)
    cuts = np.concatenate([c + spread for c in centres])
    bw = make_bigwig(tmp_path / "w.bw", cuts)

    narrow = peaks_at([centres[0]], half=100)
    wide = peaks_at([centres[1]], half=900)

    totals = qc.window_totals(bw, narrow + wide, window=1000)
    assert totals[0] == totals[1] == 1000

    # ... and the contrast: over their own extents they differ nearly 9-fold,
    # which is exactly the artifact the fixed window removes.
    own_widths = qc.peak_signal_distribution(bw, narrow + wide)
    assert own_widths["signal_per_peak_max"] / own_widths["signal_per_peak_min"] > 8


def test_regions_are_centred_on_the_summit_not_the_midpoint(tmp_path):
    """narrowPeak column 10 is an offset from the START, and it is rarely central.

    Centring on the midpoint would shift every window by the summit's offset,
    quietly measuring the wrong place for asymmetric peaks.
    """
    bw = make_bigwig(tmp_path / "s.bw", [12_000] * 500)
    # start 10000, end 14000: midpoint 12000, but the summit says 10600.
    off_centre = [["chr1", 10_000, 14_000, "p", 0, ".", 0, -1, -1, 600]]
    # A 1000bp window on the summit (10600) misses the signal at 12000 entirely;
    # one on the midpoint would catch all of it.
    assert qc.window_totals(bw, off_centre, window=1000)[0] == 0.0


def test_enrichment_reports_the_ratio_of_means(tmp_path):
    peak_centres = [10_000, 14_000]
    neg_centres = [30_000, 34_000]
    cuts = np.concatenate(
        [np.repeat(peak_centres, 400), np.repeat(neg_centres, 100)]
    )
    bw = make_bigwig(tmp_path / "e.bw", cuts)
    m, _pos, _neg = qc.peak_vs_nonpeak_signal(
        bw, peaks_at(peak_centres), negatives_at(neg_centres)
    )
    assert m["signal_enrichment_peak_over_nonpeak"] == pytest.approx(4.0)


def test_auroc_matches_scipy_on_tied_data():
    """Pin the hand-written rank code against the reference implementation."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    # Deliberately lumpy and full of ties, like real window totals.
    pos = rng.integers(0, 5, size=200).astype(float)
    neg = rng.integers(0, 5, size=300).astype(float)
    u = scipy_stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    assert qc.auroc(pos, neg) == pytest.approx(u / (pos.size * neg.size))


def test_auroc_is_none_when_a_class_is_empty():
    assert qc.auroc(np.array([1.0, 2.0]), np.array([])) is None


def test_windows_running_off_the_chromosome_are_dropped_not_zeroed(tmp_path):
    """A dropped window is honest; a zeroed one would fake a background region."""
    bw = make_bigwig(tmp_path / "edge.bw", [10_000] * 100)
    off_the_end = negatives_at([CS["chr1"] - 10])
    assert qc.window_totals(bw, off_the_end, window=1000).size == 0
