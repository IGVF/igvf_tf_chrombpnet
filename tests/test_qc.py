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
