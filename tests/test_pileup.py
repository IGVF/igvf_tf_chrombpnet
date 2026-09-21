"""Assert our numpy pileup is identical to chrombpnet's, interval for interval.

The oracle is chrombpnet's OWN command string, read out of the installed
package at test time and run verbatim -- not a reimplementation of it here.
If chrombpnet changes the command, this test starts failing instead of quietly
diverging.

Skipped unless bedtools, bedGraphToBigWig and an importable chrombpnet are all
present (the pixi `qc` environment has the first two; chrombpnet is cluster-only,
so a stub of its command string is used when it is absent -- see _command()).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))

pytest.importorskip("pybigtools", reason="needs the pixi 'qc' environment")

from utils import pileup  # noqa: E402

HAVE_TOOLS = bool(shutil.which("bedtools")) and bool(shutil.which("bedGraphToBigWig"))
needs_tools = pytest.mark.skipif(not HAVE_TOOLS, reason="bedtools/bedGraphToBigWig not on PATH")

# chrombpnet 1.0.1 reads_to_bigwig.generate_bigwig, the no-tmpdir/no-bsort branch.
# Kept here only as the fallback when chrombpnet is not importable; when it is,
# _command() reads the live source so the oracle cannot drift from the package.
_FALLBACK_CMD = (
    'awk -v OFS="\\t" \'{{if ($6=="+"){{print $1,$2{0:+},$3,$4,$5,$6}} '
    'else if ($6=="-") {{print $1,$2,$3{1:+},$4,$5,$6}}}}\' | sort -k1,1 '
    '| bedtools genomecov -bg -5 -i stdin -g {2} | LC_COLLATE="C" sort -k1,1 -k2,2n '
)


def _command() -> str:
    """chrombpnet's pileup command template, from the installed package if present."""
    try:
        import chrombpnet.helpers.preprocessing.reads_to_bigwig as r2b

        for line in Path(r2b.__file__).read_text().splitlines():
            line = line.strip()
            if line.startswith("cmd =") and "LC_COLLATE" in line and "tmpdir" not in line:
                return eval(line[len("cmd = ") :].rsplit(".format", 1)[0])  # noqa: S307
    except Exception:  # noqa: BLE001  # chrombpnet is cluster-only
        pass
    return _FALLBACK_CMD


def oracle_bigwig(tmp_path, fragments, chrom_sizes, dp, dm) -> Path:
    """Run chrombpnet's exact pipeline and return the bigWig it produces."""
    cs = tmp_path / "chrom.sizes"
    cs.write_text("".join(f"{c}\t{n}\n" for c, n in chrom_sizes.items()))
    # fragment_to_tagalign_stream: one + and one - record per fragment.
    tagalign = "".join(
        f"{c}\t{s}\t{e}\t1000\t0\t+\n{c}\t{s}\t{e}\t1000\t0\t-\n" for c, s, e in fragments
    )
    bg = tmp_path / "oracle.bedGraph"
    with open(bg, "w") as fh:
        r = subprocess.run(
            ["bash", "-c", _command().format(dp, dm, str(cs))],
            input=tagalign,
            text=True,
            stdout=fh,
            stderr=subprocess.PIPE,
        )
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    out = tmp_path / "oracle.bw"
    subprocess.run(["bedGraphToBigWig", str(bg), str(cs), str(out)], check=True)
    return out


def ours_bigwig(tmp_path, fragments, chrom_sizes, dp, dm) -> Path:
    frags = tmp_path / "frags.tsv"
    frags.write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in fragments))
    cuts, _skipped, _kept = pileup.collect_cuts(frags, chrom_sizes, dp, dm)
    out = tmp_path / "ours.bw"
    pileup.write_bigwig(out, chrom_sizes, cuts)
    return out


def intervals(path, chrom_sizes):
    """Interval structure, via pyBigWig (pybigtools 0.3 exposes only values())."""
    import pyBigWig

    bw = pyBigWig.open(str(path))
    got = {}
    for chrom in chrom_sizes:
        if chrom in bw.chroms():
            got[chrom] = [tuple(x) for x in (bw.intervals(chrom) or [])]
        else:
            got[chrom] = []
    bw.close()
    return got


def per_base(path, chrom_sizes):
    """Every base's value, so a difference cannot hide in the representation."""
    import pyBigWig

    bw = pyBigWig.open(str(path))
    got = {}
    for chrom, length in chrom_sizes.items():
        if chrom in bw.chroms():
            v = np.array(bw.values(chrom, 0, length), dtype=np.float64)
        else:
            v = np.full(length, np.nan)
        got[chrom] = np.nan_to_num(v, nan=0.0)  # no-data == zero coverage
    bw.close()
    return got


def assert_identical(tmp_path, fragments, chrom_sizes, dp=4, dm=-4):
    o = oracle_bigwig(tmp_path / "o", fragments, chrom_sizes, dp, dm)
    u = ours_bigwig(tmp_path / "u", fragments, chrom_sizes, dp, dm)

    a, b = intervals(o, chrom_sizes), intervals(u, chrom_sizes)
    assert a == b, f"\nintervals differ\nchrombpnet: {a}\nours      : {b}"

    pa, pb = per_base(o, chrom_sizes), per_base(u, chrom_sizes)
    for chrom in chrom_sizes:
        assert np.array_equal(pa[chrom], pb[chrom]), (
            f"per-base values differ on {chrom} at {np.flatnonzero(pa[chrom] != pb[chrom])[:10]}"
        )

    # A vacuous pass (both empty) would be worthless -- require real signal.
    assert any(v for v in a.values()), "fixture produced no intervals at all"
    return a


@pytest.fixture(autouse=True)
def _dirs(tmp_path):
    (tmp_path / "o").mkdir(exist_ok=True)
    (tmp_path / "u").mkdir(exist_ok=True)


CS = {"chr1": 1000, "chr2": 500}


# ── the conventions, each pinned against chrombpnet's own output ─────────────


@needs_tools
def test_single_fragment_five_prime_convention(tmp_path):
    """+ cut at start+delta; - cut at end+delta-1."""
    got = assert_identical(tmp_path, [("chr1", 100, 200)], CS)
    assert got["chr1"] == [(104, 105, 1.0), (195, 196, 1.0)]


@needs_tools
def test_overlapping_cuts_accumulate(tmp_path):
    got = assert_identical(tmp_path, [("chr1", 100, 200), ("chr1", 100, 300)], CS)
    assert (104, 105, 2.0) in got["chr1"]


@needs_tools
def test_adjacent_equal_values_are_merged(tmp_path):
    """bedGraph -bg merges runs; a per-base bigwig would differ."""
    got = assert_identical(tmp_path, [("chr1", 100, 200), ("chr1", 101, 201)], CS)
    assert (104, 106, 1.0) in got["chr1"]


@needs_tools
def test_chromosome_without_fragments_contributes_nothing(tmp_path):
    got = assert_identical(tmp_path, [("chr1", 100, 200)], CS)
    assert got["chr2"] == []


@needs_tools
def test_cut_past_chromosome_end_is_dropped(tmp_path):
    """bedtools drops it silently; clamping instead would add a phantom count."""
    assert_identical(tmp_path, [("chr1", 900, 1005)], CS, dp=4, dm=10)


@needs_tools
def test_many_fragments_across_chromosomes(tmp_path):
    rng = np.random.default_rng(0)
    frags = []
    for chrom, size in CS.items():
        starts = rng.integers(10, size - 60, size=200)
        frags += [(chrom, int(s), int(s + rng.integers(20, 50))) for s in starts]
    assert_identical(tmp_path, frags, CS)


@needs_tools
@pytest.mark.parametrize("dp,dm", [(4, -4), (0, 0), (5, -5), (-3, 2)])
def test_identical_across_shift_deltas(tmp_path, dp, dm):
    assert_identical(tmp_path, [("chr1", 100, 200), ("chr1", 150, 260)], CS, dp, dm)


# ── behaviour we implement deliberately, no oracle needed ────────────────────


def test_negative_cut_is_an_error_not_a_clamp():
    """bedtools rejects the record, so clamping to 0 would diverge silently."""
    with pytest.raises(pileup.CutSiteError):
        pileup.pileup_runs(np.array([-1, 5]), 1000)


def test_runs_are_nonzero_only():
    starts, ends, values = pileup.pileup_runs(np.array([10, 10, 12]), 100)
    assert list(zip(starts.tolist(), ends.tolist(), values.tolist())) == [
        (10, 11, 2.0),
        (12, 13, 1.0),
    ]


def test_no_cuts_gives_no_intervals():
    starts, ends, values = pileup.pileup_runs(np.array([], dtype=np.int64), 100)
    assert starts.size == ends.size == values.size == 0


# ── filtering and conversion happen in ONE pass ──────────────────────────────


def test_contigs_absent_from_chrom_sizes_are_dropped(tmp_path):
    """That drop IS the main-chromosome filter -- no separate pass needed."""
    f = tmp_path / "f.tsv"
    f.write_text("chr1\t100\t200\nchrUn_GL000220v1\t10\t50\nchr1\t300\t400\n")
    cuts, skipped, kept = pileup.collect_cuts(f, {"chr1": 1000}, 4, -4)
    assert set(cuts) == {"chr1"}
    assert skipped == {"chrUn_GL000220v1": 1}
    assert kept == 2


def test_write_filtered_emits_kept_rows_in_the_same_pass(tmp_path):
    f = tmp_path / "f.tsv"
    f.write_text("chr1\t100\t200\tAAA\t1\nchrUn\t10\t50\tBBB\t1\nchr2\t5\t9\tCCC\t2\n")
    out = tmp_path / "filtered.tsv.gz"
    cuts, skipped, kept = pileup.collect_cuts(
        f, {"chr1": 1000, "chr2": 500}, 4, -4, write_filtered=out
    )
    import gzip

    rows = gzip.open(out, "rt").read().strip().split("\n")
    assert [r.split("\t")[0] for r in rows] == ["chr1", "chr2"]
    # every column survives, so barcodes are not lost
    assert rows[0].split("\t")[3] == "AAA"
    assert kept == 2 and skipped == {"chrUn": 1}


def test_write_filtered_is_bgzipped_so_it_can_be_indexed(tmp_path):
    from utils import compression

    f = tmp_path / "f.tsv"
    f.write_text("chr1\t100\t200\n")
    out = tmp_path / "filtered.tsv.gz"
    pileup.collect_cuts(f, {"chr1": 1000}, 4, -4, write_filtered=out)
    assert compression.is_bgzf(out)


def test_filtered_output_is_optional(tmp_path):
    """Omitting it means a multi-GB file is never rewritten."""
    f = tmp_path / "f.tsv"
    f.write_text("chr1\t100\t200\n")
    pileup.collect_cuts(f, {"chr1": 1000}, 4, -4)
    assert list(tmp_path.glob("*.gz")) == []


# ── preprocessing must not depend on chrombpnet ──────────────────────────────


@pytest.mark.parametrize(
    "module", ["utils/pileup.py", "utils/shift.py", "utils/compression.py", "utils/intervals.py"]
)
def test_preprocessing_modules_do_not_import_chrombpnet(module):
    import ast

    tree = ast.parse((REPO / "lib" / "python" / module).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "chrombpnet" not in imported
    assert "torch" not in imported, "torch was dropped from shift detection"


def test_shift_deltas_target_chrombpnet_not_scprinter():
    """scPrinter's detect_shift targets +4/-5; chrombpnet wants +4/-4."""
    from utils import shift as sh

    assert sh.shift_deltas(0, 0, "ATAC") == (4, -4)
    assert sh.shift_deltas(0, 0, "DNASE") == (0, 1)
    # already at chrombpnet's convention -> no further adjustment
    assert sh.shift_deltas(4, -4, "ATAC") == (0, 0)


def test_shift_deltas_rejects_unknown_assay():
    from utils import shift as sh

    with pytest.raises(ValueError, match="ATAC or DNASE"):
        sh.shift_deltas(0, 0, "CHIP")
