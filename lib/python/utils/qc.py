"""QC on the signal and peaks that are about to be handed to ChromBPNet.

Runs between 01 (peaks) and 02 (non-peaks), on the two artifacts that actually
go into training: the prepared bigwig of Tn5 insertion counts and the filtered
narrowPeak. QC'ing those rather than the upstream fragments means the numbers
describe what the model will see, including every filter applied along the way.

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
    "peak_signal_distribution",
    "signal_summary",
    "tss_enrichment",
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
