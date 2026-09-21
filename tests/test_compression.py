"""Pin the bgzip output contract.

The point of bgzip here is that it buys tabix indexing *without* costing gzip
compatibility. Both halves of that need a test, because a regression to plain
gzip is silent -- everything still reads, only indexing breaks.
"""

from __future__ import annotations

import gzip
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))

pytest.importorskip("pysam", reason="needs the pixi 'qc' environment")

from utils import compression  # noqa: E402


def test_output_is_bgzf_not_plain_gzip(tmp_path):
    p = tmp_path / "x.bed.gz"
    compression.write_tsv(pd.DataFrame([["chr1", 1, 2]]), p)
    assert compression.is_bgzf(p)


def test_plain_gzip_fallback_is_not_bgzf(tmp_path):
    p = tmp_path / "x.bed.gz"
    with compression.open_write(p, bgzip=False) as fh:
        fh.write(b"chr1\t1\t2\n")
    assert not compression.is_bgzf(p)


def test_bgzf_is_readable_as_ordinary_gzip(tmp_path):
    """bgzip must stay transparent to zcat, pandas and chrombpnet."""
    p = tmp_path / "x.bed.gz"
    compression.write_tsv(pd.DataFrame([["chr1", 1, 2], ["chr2", 3, 4]]), p)
    assert gzip.open(p, "rt").read() == "chr1\t1\t2\nchr2\t3\t4\n"


def test_pandas_reads_it_back(tmp_path):
    p = tmp_path / "x.bed.gz"
    compression.write_tsv(pd.DataFrame([["chr1", 1, 2]]), p)
    df = pd.read_csv(p, sep="\t", header=None)
    assert df.iloc[0].tolist() == ["chr1", 1, 2]


def test_tabix_index_allows_region_fetch(tmp_path):
    import pysam

    p = tmp_path / "x.bed.gz"
    compression.write_tsv(pd.DataFrame([["chr1", 10, 20], ["chr1", 30, 40], ["chr2", 50, 60]]), p)
    assert compression.tabix_index(p, preset="bed").exists()
    assert list(pysam.TabixFile(str(p)).fetch("chr2", 0, 100)) == ["chr2\t50\t60"]


def test_uncompressed_path_is_left_uncompressed(tmp_path):
    p = tmp_path / "x.bed"
    compression.write_tsv(pd.DataFrame([["chr1", 1, 2]]), p)
    assert p.read_text() == "chr1\t1\t2\n"


# ── filter_fragments.py end to end ────────────────────────────────────────────


def _write_fragments(path, rows):
    with gzip.open(path, "wb") as fh:
        fh.writelines(rows)


def test_filter_fragments_keeps_main_chroms_and_bgzips(tmp_path):
    src, dst = tmp_path / "in.tsv.gz", tmp_path / "out.tsv.gz"
    _write_fragments(
        src,
        [
            b"#description\n",
            b"chr1\t1\t2\tAAA\t1\n",
            b"chrM\t5\t6\tCCC\t1\n",
            b"chrUn_GL000220v1\t9\t10\tGGG\t1\n",
            b"chrX\t7\t8\tNNN\t1\n",
        ],
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO / "src" / "cli.py"),
            "filter-fragments",
            "--input",
            str(src),
            "--output",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    kept = [ln.split("\t")[0] for ln in gzip.open(dst, "rt").read().strip().split("\n")]
    assert kept == ["chr1", "chrM", "chrX"]  # comment + scaffold dropped
    assert compression.is_bgzf(dst)  # indexable


def test_filter_fragments_writes_tabix_index_on_request(tmp_path):
    src, dst = tmp_path / "in.tsv.gz", tmp_path / "out.tsv.gz"
    _write_fragments(src, [b"chr1\t1\t2\tAAA\t1\n", b"chr2\t3\t4\tCCC\t1\n"])
    subprocess.run(
        [
            sys.executable,
            str(REPO / "src" / "cli.py"),
            "filter-fragments",
            "--input",
            str(src),
            "--output",
            str(dst),
            "--index",
        ],
        check=True,
        capture_output=True,
    )
    assert Path(f"{dst}.tbi").exists()


def test_filter_fragments_fails_loudly_when_nothing_matches(tmp_path):
    """A chr1-vs-1 naming mismatch must not silently produce an empty file."""
    src, dst = tmp_path / "in.tsv.gz", tmp_path / "out.tsv.gz"
    _write_fragments(src, [b"1\t1\t2\tAAA\t1\n", b"2\t3\t4\tCCC\t1\n"])
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "src" / "cli.py"),
            "filter-fragments",
            "--input",
            str(src),
            "--output",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "no fragments kept" in (r.stderr + r.stdout)


def test_cli_exposes_the_converted_tools():
    """One command, subcommands -- the entry point the steps call."""
    r = subprocess.run(
        [sys.executable, str(REPO / "src" / "cli.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for sub in ("preprocess-peaks", "filter-fragments", "legacy"):
        assert sub in r.stdout


def test_cli_preprocess_peaks_runs_end_to_end(tmp_path):
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t1000\t1200\nchr1\t5000\t5300\nchr1\t9000\t9100\n")
    bl = tmp_path / "bl.bed.gz"
    with gzip.open(bl, "wt") as fh:
        fh.write("chr1\t5100\t5150\n")
    cs = tmp_path / "cs.tsv"
    cs.write_text("chr1\t248956422\n")
    out = tmp_path / "preprocessing"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "src" / "cli.py"),
            "preprocess-peaks",
            "--peaks",
            str(peaks),
            "--blacklist",
            str(bl),
            "--chrom-sizes",
            str(cs),
            "--input-window",
            "2114",
            "--out-dir",
            str(out),
            "--prefix",
            "t",
            "--metadata-dir",
            str(tmp_path / "metadata"),
        ],
        check=True,
        capture_output=True,
    )
    rows = (out / "t_peaks_no_blacklist.narrowPeak").read_text().strip().split("\n")
    assert len(rows) == 2  # the blacklisted peak is gone
    assert rows[0].split("\t")[3] == "peak_1"  # renumbered over the filtered set
    assert (tmp_path / "metadata").exists()  # and it recorded the run
