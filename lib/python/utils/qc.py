"""QC on the signal and peaks that are about to be handed to ChromBPNet.

Runs after 01 (non-peaks) and before any GPU time, on the three artifacts that
actually go into training: the prepared bigwig of Tn5 insertion counts, the
filtered narrowPeak, and the GC-matched negatives. QC'ing those rather than the
upstream fragments means the numbers describe what the model will see, after
every filter.

Two halves, and the second is why this runs after 01 rather than before it:

- **individual** -- is the signal accessibility data, and does it agree with
  the peaks? (TSS enrichment, profiles, depth, signal per peak)
- **comparative** -- can signal tell a peak from its own GC-matched
  background? That is precisely the question training poses, so asking it here
  says whether the dataset is worth training on at all.

Deliberately not SnapATAC2. That is a single-cell toolkit: ``metrics.tsse``
wants an AnnData built by ``import_fragments`` and reports per-cell scores.
This pipeline trains on pseudobulk, so a per-cell distribution is not
actionable for "is this dataset worth training on", and building a cell x bin
matrix to reach the library aggregate is a lot of work for one number. Use
SnapATAC2 for per-cell QC upstream of pseudobulking; that is a different and
also worthwhile question.

Every metric here is computed from a bigwig, so it applies equally to a signal
that was supplied as a bigwig in the first place.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "aggregate_profile",
    "auroc",
    "peak_signal_distribution",
    "peak_vs_nonpeak_signal",
    "signal_summary",
    "tss_enrichment",
    "window_totals",
]

#: Profiles are averaged over at most this many regions. The aggregate shape
#: converges long before every region is used, and this keeps QC to seconds.
DEFAULT_MAX_REGIONS = 20_000


def _open(bigwig):
    import pyBigWig

    return pyBigWig.open(str(bigwig))


def _values(bw, chrom, start, end) -> np.ndarray | None:
    """Per-base values, NaN treated as zero coverage. None if out of bounds."""
    length = bw.chroms().get(chrom)
    if length is None or start < 0 or end > length:
        return None
    v = np.asarray(bw.values(chrom, start, end), dtype=np.float64)
    return np.nan_to_num(v, nan=0.0)


def _centres(regions, kind: str) -> list[tuple[str, int]]:
    """(chrom, centre) per region. narrowPeak centres on start+summit."""
    out = []
    for row in regions:
        if kind == "narrowpeak":
            out.append((row[0], int(row[1]) + int(row[9])))
        else:  # BED: use the midpoint
            out.append((row[0], (int(row[1]) + int(row[2])) // 2))
    return out


def aggregate_profile(
    bigwig,
    regions,
    flank: int = 1000,
    kind: str = "narrowpeak",
    max_regions: int = DEFAULT_MAX_REGIONS,
    seed: int = 0,
):
    """Mean insertion profile in +/-``flank`` around region centres.

    Returns ``(offsets, mean_profile, n_used)``. This is the cumulative /
    metaprofile plot: a sharp central enrichment is what open chromatin looks
    like, and a flat one means the signal and the regions disagree.
    """
    centres = _centres(regions, kind)
    rng = np.random.default_rng(seed)
    if len(centres) > max_regions:
        idx = rng.choice(len(centres), size=max_regions, replace=False)
        centres = [centres[i] for i in sorted(idx)]

    bw = _open(bigwig)
    try:
        total = np.zeros(2 * flank, dtype=np.float64)
        used = 0
        for chrom, centre in centres:
            v = _values(bw, chrom, centre - flank, centre + flank)
            if v is None:
                continue  # region's window runs off the chromosome
            total += v
            used += 1
    finally:
        bw.close()

    offsets = np.arange(-flank, flank)
    if used == 0:
        return offsets, np.zeros(2 * flank), 0
    return offsets, total / used, used


def tss_enrichment(bigwig, tss, flank: int = 2000, flank_width: int = 100, **kw):
    """TSS enrichment: central signal over the background in the far flanks.

    The ENCODE-style definition: build the mean profile around TSSs, take the
    background as the mean of the outermost ``flank_width`` bases at each end,
    and report the peak of the (background-normalised) profile.

    A value near 1 means no enrichment at all -- either the signal is not
    accessibility, or it does not match this genome's annotation.
    """
    offsets, profile, used = aggregate_profile(bigwig, tss, flank=flank, kind="bed", **kw)
    if used == 0:
        return {"tss_enrichment": None, "n_tss_used": 0}

    background = float(np.concatenate([profile[:flank_width], profile[-flank_width:]]).mean())
    if background <= 0:
        return {"tss_enrichment": None, "n_tss_used": used, "note": "zero flank signal"}

    normalised = profile / background
    return {
        "tss_enrichment": float(normalised.max()),
        "tss_enrichment_at_centre": float(normalised[flank]),
        "n_tss_used": used,
        "background_per_base": background,
        "profile_offsets": offsets.tolist(),
        "profile": normalised.tolist(),
    }


def peak_signal_distribution(bigwig, peaks, max_regions: int | None = None):
    """Total insertions per peak, and how unevenly they are spread.

    ``frac_peaks_zero_signal`` is the one to watch: peaks with no insertions
    under them are regions the model is asked to explain with no evidence, and
    a non-trivial fraction usually means the peaks and the signal came from
    different samples.
    """
    bw = _open(bigwig)
    try:
        totals = []
        for row in peaks if max_regions is None else peaks[:max_regions]:
            chrom, start, end = row[0], int(row[1]), int(row[2])
            v = _values(bw, chrom, start, end)
            totals.append(0.0 if v is None else float(v.sum()))
    finally:
        bw.close()

    t = np.asarray(totals, dtype=np.float64)
    if t.size == 0:
        return {"n_peaks": 0}
    q = np.percentile(t, [0, 25, 50, 75, 90, 99, 100])
    return {
        "n_peaks": int(t.size),
        "insertions_in_peaks": float(t.sum()),
        "signal_per_peak_mean": float(t.mean()),
        "signal_per_peak_min": float(q[0]),
        "signal_per_peak_q25": float(q[1]),
        "signal_per_peak_median": float(q[2]),
        "signal_per_peak_q75": float(q[3]),
        "signal_per_peak_q90": float(q[4]),
        "signal_per_peak_q99": float(q[5]),
        "signal_per_peak_max": float(q[6]),
        "frac_peaks_zero_signal": float((t == 0).mean()),
    }


def signal_summary(bigwig):
    """Genome-wide totals straight from the bigwig header."""
    bw = _open(bigwig)
    try:
        header = bw.header()
        chroms = bw.chroms()
    finally:
        bw.close()
    return {
        "total_insertions": float(header.get("sumData", 0.0)),
        "covered_bases": int(header.get("nBasesCovered", 0)),
        "max_per_base": float(header.get("maxVal", 0.0)),
        "mean_per_covered_base": float(header.get("sumData", 0.0) / header["nBasesCovered"])
        if header.get("nBasesCovered")
        else 0.0,
        "n_chromosomes": len(chroms),
        "genome_bases": int(sum(chroms.values())),
    }


# -- comparative: peaks against their own GC-matched background ---------------
# The negatives from 01 are the other half of what ChromBPNet trains on. They
# are matched to the peaks on GC and drawn from the same fold chromosomes, so
# the ONE thing that should separate them is accessibility. If it does not, the
# model has nothing to learn from this dataset and the GPU time is wasted --
# which is the whole reason this QC runs before training rather than after.


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, like ``scipy.stats.rankdata``.

    Written out rather than imported: ties are the common case here (peaks and
    background alike are full of zero-signal windows, which all tie at 0) and
    scipy is not installed in every environment this runs in.
    """
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    n = x.size
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(sx[1:], sx[:-1], out=is_new[1:])
    starts = np.flatnonzero(is_new)
    ends = np.append(starts[1:], n) - 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ((starts + ends) / 2.0 + 1.0)[np.cumsum(is_new) - 1]
    return ranks


def auroc(positive, negative) -> float | None:
    """Probability a random positive outscores a random negative; ties count half.

    The Mann-Whitney U form, so it stays exact under the heavy tying that
    zero-signal windows produce.
    """
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        return None
    ranks = _average_ranks(np.concatenate([positive, negative]))
    n_pos, n_neg = positive.size, negative.size
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def window_totals(
    bigwig,
    regions,
    window: int,
    kind: str = "narrowpeak",
    max_regions: int = DEFAULT_MAX_REGIONS,
    seed: int = 0,
) -> np.ndarray:
    """Total insertions in a FIXED ``window`` centred on each region's summit.

    Fixed width is the point. :func:`peak_signal_distribution` sums each peak
    over its own called width, which describes the peaks well but cannot
    compare them to anything: the negatives are all ``inputlen`` wide, so a
    300bp peak would lose to a background window on width alone. Here both
    sides get the same number of bases, so what differs is signal density.

    Both peaks and chrombpnet's negatives are 10-column BED with the summit
    offset in column 10, so ``kind="narrowpeak"`` centres both correctly.
    """
    centres = _centres(regions, kind)
    rng = np.random.default_rng(seed)
    if len(centres) > max_regions:
        idx = rng.choice(len(centres), size=max_regions, replace=False)
        centres = [centres[i] for i in sorted(idx)]

    half = window // 2
    bw = _open(bigwig)
    try:
        totals = []
        for chrom, centre in centres:
            v = _values(bw, chrom, centre - half, centre + half)
            if v is not None:  # drop, rather than zero, a window off the end
                totals.append(float(v.sum()))
    finally:
        bw.close()
    return np.asarray(totals, dtype=np.float64)


def peak_vs_nonpeak_signal(bigwig, peaks, nonpeaks, window: int = 1000, **kw):
    """How well signal alone separates peaks from their GC-matched background.

    ``window`` defaults to 1000, ChromBPNet's OUTPUT window -- the span the
    counts head is asked to predict -- so the separation measured here is the
    same quantity the trained model gets scored on.

    Returns ``(metrics, peak_totals, nonpeak_totals)``. The headline is
    ``auroc_peaks_vs_nonpeaks``: 1.0 is perfect separation, 0.5 means signal
    cannot tell a peak from background and there is nothing to train on.
    """
    pos = window_totals(bigwig, peaks, window, **kw)
    neg = window_totals(bigwig, nonpeaks, window, **kw)
    if pos.size == 0 or neg.size == 0:
        return (
            {"n_peaks_compared": int(pos.size), "n_nonpeaks_compared": int(neg.size)},
            pos,
            neg,
        )

    pos_median, neg_median = float(np.median(pos)), float(np.median(neg))
    metrics = {
        "compare_window": int(window),
        "n_peaks_compared": int(pos.size),
        "n_nonpeaks_compared": int(neg.size),
        "peak_signal_mean": float(pos.mean()),
        "peak_signal_median": pos_median,
        "nonpeak_signal_mean": float(neg.mean()),
        "nonpeak_signal_median": neg_median,
        "signal_enrichment_peak_over_nonpeak": float(pos.mean() / neg.mean())
        if neg.mean() > 0
        else None,
        "auroc_peaks_vs_nonpeaks": auroc(pos, neg),
        # Background windows as accessible as a typical peak. A few percent is
        # expected -- there is open chromatin outside called peaks -- but a
        # large fraction means the peak calls left real signal behind.
        "frac_nonpeaks_above_peak_median": float((neg >= pos_median).mean()),
        "frac_nonpeaks_zero_signal": float((neg == 0).mean()),
        "frac_peaks_below_nonpeak_median": float((pos <= neg_median).mean()),
    }
    return metrics, pos, neg


def peak_width_summary(peaks):
    widths = np.array([int(r[2]) - int(r[1]) for r in peaks], dtype=np.int64)
    if widths.size == 0:
        return {}
    q = np.percentile(widths, [0, 50, 100])
    return {
        "peak_width_min": int(q[0]),
        "peak_width_median": float(q[1]),
        "peak_width_max": int(q[2]),
        "peak_bases_total": int(widths.sum()),
    }
