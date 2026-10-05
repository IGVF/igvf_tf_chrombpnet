"""Pin utils.peakcall to the igvf_pseudobulking_pipeline commands it reproduces.

The oracles are the commands themselves where they can run here: GNU sort for
the top-N cut and bedtools for the reproducibility overlap. The insertion
stream is checked against chrombpnet's cut convention (utils.pileup), so peaks
and the bigwig agree on where an insertion is.

Run with `pixi run -e qc test`; the oracle tests skip without sort/bedtools/mawk.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

pytest.importorskip("pyranges1", reason="needs the pixi 'qc' environment")

from utils import intervals, peakcall, pileup, shift  # noqa: E402

needs = {
    tool: pytest.mark.skipif(not shutil.which(tool), reason=f"{tool} not on PATH")
    for tool in ("sort", "bedtools", "mawk", "bgzip")
}


def _narrowpeak(tmp_path, rows, name="x.narrowPeak") -> Path:
    path = tmp_path / name
    path.write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
    return path


def _peak(chrom, start, end, p, i=0):
    return (chrom, start, end, f"p{i}", 10, ".", 2.5, p, 1.0, (end - start) // 2)


# ── top-N == sort -k 8g,8g | tail -n N ────────────────────────────────────────


@needs["sort"]
@pytest.mark.parametrize("n", [1, 3, 5, 9, 20])
def test_top_n_matches_sort_tail_including_ties(tmp_path, n):
    rng = np.random.default_rng(0)
    pvals = rng.choice([2.5, 3.0, 3.0, 7.25, 10.0, 1e3], size=12)  # ties on purpose
    rows = [_peak(f"chr{1 + i % 3}", 100 * i, 100 * i + 50, p, i) for i, p in enumerate(pvals)]
    path = _narrowpeak(tmp_path, rows)
    oracle = subprocess.run(
        f"sort -k 8g,8g {path} | tail -n {n}",
        shell=True,
        capture_output=True,
        check=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    ).stdout.splitlines()
    ours = peakcall.top_n_by_pvalue(peakcall.read_narrowpeak(path), n)
    assert sorted(ours["_line"]) == sorted(oracle)


def test_top_n_keeps_the_strongest_not_the_weakest(tmp_path):
    """The pre-v2.0.1 `sort --reverse -k 8gr,8gr | tail` kept the weakest rows."""
    rows = [_peak("chr1", 100 * i, 100 * i + 50, p, i) for i, p in enumerate([10, 5, 3, 2, 1])]
    top = peakcall.top_n_by_pvalue(peakcall.read_narrowpeak(_narrowpeak(tmp_path, rows)), 2)
    assert sorted(top["pvalue"]) == [5, 10]


# ── reproducibility overlap == bedtools intersect -u -f F -F F -e ─────────────


@needs["bedtools"]
@pytest.mark.parametrize("fraction", [0.5, 0.2, 0.9])
def test_either_fraction_overlap_matches_bedtools(tmp_path, fraction):
    rng = np.random.default_rng(1)

    def random_bed(n):
        start = rng.integers(0, 20_000, n)
        return pd.DataFrame(
            {
                "Chromosome": rng.choice(["chr1", "chr2"], n),
                "Start": start,
                "End": start + rng.integers(1, 600, n),
            }
        ).sort_values(["Chromosome", "Start", "End"], ignore_index=True)

    a, b = random_bed(400), random_bed(300)
    a_path, b_path = tmp_path / "a.bed", tmp_path / "b.bed"
    a.to_csv(a_path, sep="\t", header=False, index=False)
    b.to_csv(b_path, sep="\t", header=False, index=False)
    oracle = subprocess.run(
        ["bedtools", "intersect", "-u", "-a", a_path, "-b", b_path,
         "-f", str(fraction), "-F", str(fraction), "-e"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    expected = sorted(tuple(line.split("\t")[:3]) for line in oracle.splitlines())
    mask = intervals.overlaps_either_fraction(a, b, fraction)
    got = sorted(tuple(map(str, r)) for r in a[mask].itertuples(index=False))
    assert got == expected


def test_any_overlap_is_bedtools_u_and_bookends_do_not_count():
    a = pd.DataFrame({"Chromosome": ["chr1"] * 3, "Start": [100, 500, 1000], "End": [200, 600, 1100]})
    b = pd.DataFrame({"Chromosome": ["chr1"], "Start": [199], "End": [1000]})
    assert list(intervals.overlaps_either_fraction(a, b, None)) == [True, True, False]


# ── finalize: blacklist, caps, order ──────────────────────────────────────────


def test_finalize_caps_score_and_end_and_orders_by_chrom_sizes():
    peaks = pd.DataFrame(
        [
            ("chr2", 50, 120, "b", 5000, ".", 1.0, 3.0, 1.0, 10),
            ("chr1", 900, 1200, "a", 10, ".", 1.0, 3.0, 1.0, 10),
            ("chr1", 10, 60, "c", 10, ".", 1.0, 3.0, 1.0, 10),
        ],
        columns=peakcall.NARROWPEAK_COLUMNS,
    )
    blacklist = pd.DataFrame({"Chromosome": ["chr1"], "Start": [40], "End": [45]})
    out = peakcall.finalize(peaks, blacklist, {"chr1": 1000, "chr2": 100})
    assert list(out["name"]) == ["a", "b"]  # c touches the blacklist
    assert list(out["End"]) == [1000, 100]  # capped at the chromosome length
    assert list(out["score"]) == [10, 1000]


# ── insertion stream: where chrombpnet's bigwig puts the cuts ────────────────


@needs["mawk"]
@needs["bgzip"]
@pytest.mark.parametrize("macs_input", ["bed", "frag"])
def test_insertions_sit_on_chrombpnet_cuts(tmp_path, macs_input):
    frags = tmp_path / "f.tsv"
    frags.write_text(
        "# header\nchr1\t100\t300\tA\t1\nchr1\t50\t80\tB\t2\nchrUn\t5\t9\tC\t1\nchr2\t0\t200\tD\t1\n"
    )
    subprocess.run(["bgzip", "-f", str(frags)], check=True)
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t10000\nchr2\t10000\n")
    dp, dm = shift.shift_deltas(4, -5)  # 10x fragments -> (0, +1)
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    kept, skipped = peakcall.split_insertions(
        str(frags) + ".gz", sizes, r1, r2, dp, dm, macs_input=macs_input
    )
    assert (kept, skipped) == (3, 1)
    rows = [ln.split("\t") for p in (r1, r2) if p.exists() for ln in p.read_text().splitlines()]
    centres = sorted(
        (r[0], int(r[1]) if macs_input == "bed" else (int(r[1]) + int(r[2])) // 2)
        for r in rows
        if not (macs_input == "frag" and int(r[1]) == 0)  # window clipped at base 0
    )
    starts = np.array([100, 50, 0])
    ends = np.array([300, 80, 200])
    cuts = pileup.cut_positions(starts, ends, dp, dm)
    expected = sorted(
        (c, int(x)) for c, x in zip(["chr1", "chr1", "chr2"] * 2, cuts, strict=True)
    )
    if macs_input == "frag":
        expected = [e for e in expected if e[1] >= peakcall.HALF_WINDOW]
    assert centres == expected
    if macs_input == "frag":
        widths = {int(r[2]) - int(r[1]) for r in rows if int(r[1]) > 0}
        assert widths == {2 * peakcall.HALF_WINDOW}


@needs["mawk"]
@needs["bgzip"]
def test_split_is_deterministic_and_keeps_fragments_whole(tmp_path):
    lines = "".join(f"chr1\t{10 * i}\t{10 * i + 100}\tB\t1\n" for i in range(2000))
    frags = tmp_path / "f.tsv.gz"
    with gzip.open(frags, "wt") as fh:  # plain gzip is fine for bgzip -dc
        fh.write(lines)
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t100000\n")
    outs = []
    for k in range(2):
        r1, r2 = tmp_path / f"a{k}", tmp_path / f"b{k}"
        peakcall.split_insertions(frags, sizes, r1, r2, 0, 1, macs_input="bed", seed=42)
        outs.append((r1.read_text(), r2.read_text()))
    assert outs[0] == outs[1]
    n1 = outs[0][0].count("\n")
    assert n1 % 2 == 0 and 0.4 < n1 / 4000 < 0.6  # two insertions per fragment, ~half


def test_macs_bed_route_is_the_pipelines_flags():
    cmd = peakcall.macs_command("x", ["r1.bed"], "out")
    flags = " ".join(cmd)
    for flag in ("-f BED", "--shift -75", "--extsize 150", "-g hs", "-p 0.01", "--nomodel",
                 "--keep-dup all", "--call-summits"):  # fmt: skip
        assert flag in flags
    assert "-B" not in cmd and "--SPMR" not in cmd


# ── the shift the detector reports ────────────────────────────────────────────


def test_detector_offsets_are_relative_to_the_10x_frame():
    """scPrinter's reference matrices sit at +4/-5: offset 0 means 10x fragments."""
    assert shift.shift_present(0, 0) == (4, -5)
    assert shift.shift_present(4, -5) == (0, 0)
    # and chrombpnet's correction for 10x fragments is (0, +1), giving +4/-4
    assert shift.shift_deltas(*shift.shift_present(0, 0)) == (0, 1)


def test_circular_detect_reads_a_rolled_reference_as_that_offset():
    for k in (-3, 0, 4):
        query = np.roll(shift.ref_forward_bias, k, axis=0)
        assert shift.circular_detect(shift.ref_forward_bias, query) == k
