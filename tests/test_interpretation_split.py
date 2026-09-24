"""04.0 trains, 04.4/04.5 interpret -- and the launcher 03.0/04.0 train through.

04.0 stops `chrombpnet pipeline` after the marginal footprints with
chrombpnet's own `--skip-interpretation`, and 04.4 runs the interpretation
from the outputs left behind. That is only a faithful split if 04.4 picks the
same 30K regions pipeline would; the subsample tests pin that.

03.0 and 04.0 launch chrombpnet through src/chrombpnet_train.py, which now
patches nothing: it checks that the -bw bigwig is the one 00.0 prepared from
the configured signal (its prepared_bigwig.json sidecar: resolved signal
path, md5, assay), runs chrombpnet in-process, and records peak RSS. The
launcher tests cover that check and the argv hand-off. None of them needs
chrombpnet installed: where the hand-off is exercised, a stand-in module
takes chrombpnet.CHROMBPNET's place.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))
sys.path.insert(0, str(REPO / "src"))

import chrombpnet_train as ct  # noqa: E402
from utils import metadata, regions  # noqa: E402

# ── the 30K interpretation subsample ─────────────────────────────────────────


def _chrombpnet_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim from chrombpnet/pipelines.py (bias QC and full pipeline alike)."""
    if df.shape[0] > 30000:
        return df.sample(30000, random_state=1234)
    return df


def _peaks(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"chr": ["chr1"] * n, "start": range(n), "end": range(1, n + 1), "summit": [0] * n}
    )


def test_subsample_matches_chrombpnet_row_for_row(tmp_path):
    src = tmp_path / "filtered.peaks.bed"
    _peaks(45_000).to_csv(src, sep="\t", header=False, index=False)
    out = regions.subsample_regions(src, tmp_path / "30K_subsample_peaks.bed")
    ours = pd.read_csv(out, sep="\t", header=None)
    theirs = _chrombpnet_rule(pd.read_csv(src, sep="\t", header=None))
    assert ours.shape == (30_000, 4)
    assert ours.values.tolist() == theirs.values.tolist()  # same rows, same order


def test_subsample_keeps_everything_at_or_below_30k(tmp_path):
    src = tmp_path / "p.bed"
    _peaks(30_000).to_csv(src, sep="\t", header=False, index=False)
    out = regions.subsample_regions(src, tmp_path / "s.bed")
    assert (
        pd.read_csv(out, sep="\t", header=None).values.tolist()
        == pd.read_csv(src, sep="\t", header=None).values.tolist()
    )


# ── the launcher: importing it ───────────────────────────────────────────────


def test_launcher_imports_without_chrombpnet():
    """The check must be able to fail before chrombpnet (Keras 3, JAX) loads.

    Run in a fresh interpreter so nothing another test put in sys.modules
    can hide an import.
    """
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import chrombpnet_train; "
        "assert 'chrombpnet' not in sys.modules, 'chrombpnet imported at module level'"
    )
    subprocess.run([sys.executable, "-c", code, str(REPO / "src")], check=True)


# ── the launcher: argv ───────────────────────────────────────────────────────


def test_split_argv_hands_everything_after_the_first_separator_to_chrombpnet():
    opts, theirs = ct.split_argv(
        [
            "--signal",
            "/s.tsv.gz",
            "--assay",
            "ATAC",
            "--",
            "bias",
            "train",
            "-bw",
            "/b.bw",
            "--",
            "x",
        ]
    )
    assert (opts.signal, opts.assay) == ("/s.tsv.gz", "ATAC")
    assert theirs == ["bias", "train", "-bw", "/b.bw", "--", "x"]


def test_split_argv_without_the_check():
    opts, theirs = ct.split_argv(["--", "pipeline", "-o", "/out"])
    assert (opts.signal, opts.assay) == (None, None)
    assert theirs == ["pipeline", "-o", "/out"]


@pytest.mark.parametrize(
    "argv",
    [
        ["bias", "train"],  # no separator: chrombpnet's argv would be parsed as ours
        ["--signal", "/s", "--assay", "ATAC"],  # no chrombpnet command at all
        ["--signal", "/s", "--assay", "ATAC", "--"],  # separator, then nothing
        ["--signal", "/s", "--", "bias", "train"],  # --signal without --assay
        ["--assay", "ATAC", "--", "bias", "train"],  # --assay without --signal
        ["--signal", "/s", "--assay", "CHIP", "--", "bias", "train"],  # not an assay
    ],
)
def test_split_argv_rejects(argv):
    with pytest.raises(SystemExit) as exc:
        ct.split_argv(argv)
    assert exc.value.code != 0


def test_split_argv_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        ct.split_argv(["--help"])
    assert exc.value.code == 0
    assert "--signal" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["-ibw", "-bw", "--bigwig"])
def test_option_value_finds_every_bigwig_spelling(flag):
    assert ct.option_value(["pipeline", flag, "/b.bw", "-o", "/out"], ct.BIGWIG_FLAGS) == "/b.bw"
    assert ct.option_value(["pipeline", f"{flag}=/b.bw"], ct.BIGWIG_FLAGS) == "/b.bw"


def test_option_value_last_one_wins_like_argparse():
    argv = ["-bw", "/first.bw", "--bigwig", "/second.bw"]
    assert ct.option_value(argv, ct.BIGWIG_FLAGS) == "/second.bw"
    assert ct.option_value(["-ifrag", "/f.tsv.gz"], ct.BIGWIG_FLAGS) is None
    assert ct.option_value(["-d", "DNASE"], ct.ASSAY_FLAGS) == "DNASE"
    assert ct.option_value(["--data-type=ATAC"], ct.ASSAY_FLAGS) == "ATAC"


# ── the launcher: the sidecar check ──────────────────────────────────────────


def _prepare(tmp_path: Path, assay: str = "ATAC", **override) -> tuple[Path, Path]:
    """Lay out a signal and a prepared bigwig the way 00.0 / cli.py prepare-bigwig do."""
    signal = tmp_path / "reads" / "sample.fragments.tsv.gz"
    signal.parent.mkdir()
    signal.write_bytes(b"chr1\t100\t200\tAAAC\t1\n" * 50)
    prepared = tmp_path / "preprocessing" / "signal"
    prepared.mkdir(parents=True)
    bigwig = prepared / "data_unstranded.bw"
    bigwig.write_bytes(b"not really a bigwig")
    record = {
        "signal_path": str(signal.resolve()),
        "signal_md5": metadata.md5sum(signal),
        "signal_type": "fragments",
        "assay": assay,
    }
    record.update(override)
    (prepared / ct.SIDECAR).write_text(json.dumps(record))
    return signal, bigwig


def test_sidecar_match(tmp_path):
    signal, bigwig = _prepare(tmp_path)
    assert ct.sidecar_mismatch(bigwig, str(signal), "ATAC") is None


def test_sidecar_wrong_signal_path(tmp_path):
    signal, bigwig = _prepare(tmp_path)
    other = tmp_path / "reads" / "other.fragments.tsv.gz"
    other.write_bytes(signal.read_bytes())  # same bytes, same md5: the path alone must fail it
    problem = ct.sidecar_mismatch(bigwig, str(other), "ATAC")
    assert problem and "prepared from" in problem


def test_sidecar_signal_changed_since_preparation(tmp_path):
    signal, bigwig = _prepare(tmp_path)
    signal.write_bytes(signal.read_bytes() + b"chr2\t5\t9\tAAAC\t1\n")
    problem = ct.sidecar_mismatch(bigwig, str(signal), "ATAC")
    assert problem and "changed" in problem and "md5" in problem


def test_sidecar_wrong_assay(tmp_path):
    signal, bigwig = _prepare(tmp_path, assay="DNASE")
    problem = ct.sidecar_mismatch(bigwig, str(signal), "ATAC")
    assert problem and "assay DNASE" in problem


def test_sidecar_without_md5_is_a_mismatch(tmp_path):
    """No fallback conversion any more, so an unverifiable bigwig must not pass."""
    signal, bigwig = _prepare(tmp_path, signal_md5=None)
    problem = ct.sidecar_mismatch(bigwig, str(signal), "ATAC")
    assert problem and "signal_md5" in problem


@pytest.mark.parametrize("what", ["sidecar", "bigwig", "signal"])
def test_sidecar_missing_files(tmp_path, what):
    signal, bigwig = _prepare(tmp_path)
    {"sidecar": bigwig.parent / ct.SIDECAR, "bigwig": bigwig, "signal": signal}[what].unlink()
    assert ct.sidecar_mismatch(bigwig, str(signal), "ATAC") is not None


def test_sidecar_unreadable(tmp_path):
    signal, bigwig = _prepare(tmp_path)
    (bigwig.parent / ct.SIDECAR).write_text("{not json")
    assert "cannot read" in ct.sidecar_mismatch(bigwig, str(signal), "ATAC")
    (bigwig.parent / ct.SIDECAR).write_text("[1, 2]")
    assert "JSON object" in ct.sidecar_mismatch(bigwig, str(signal), "ATAC")


def test_sidecar_is_found_beside_a_symlinked_bigwig(tmp_path):
    """A bigwig signal is registered by 00.0 as a symlink; the sidecar sits beside the
    link, not beside its target, so the -bw path must not be resolved first."""
    signal = tmp_path / "reads" / "sample.bw"
    signal.parent.mkdir()
    signal.write_bytes(b"a bigwig from elsewhere")
    prepared = tmp_path / "preprocessing" / "signal"
    prepared.mkdir(parents=True)
    link = prepared / "data_unstranded.bw"
    link.symlink_to(signal)
    (prepared / ct.SIDECAR).write_text(
        json.dumps(
            {
                "signal_path": str(signal.resolve()),
                "signal_md5": metadata.md5sum(signal),
                "signal_type": "bigwig",
                "assay": "ATAC",
            }
        )
    )
    assert ct.sidecar_mismatch(link, str(signal), "ATAC") is None


# ── the launcher: main() and the in-process hand-off ─────────────────────────


@pytest.fixture
def fake_chrombpnet(monkeypatch):
    """Stand in for chrombpnet.CHROMBPNET; records the argv main() saw."""
    calls: list[list[str]] = []
    pkg = types.ModuleType("chrombpnet")
    cli = types.ModuleType("chrombpnet.CHROMBPNET")
    cli.main = lambda: calls.append(list(sys.argv))
    pkg.CHROMBPNET = cli
    monkeypatch.setitem(sys.modules, "chrombpnet", pkg)
    monkeypatch.setitem(sys.modules, "chrombpnet.CHROMBPNET", cli)
    monkeypatch.setattr(sys, "argv", ["chrombpnet_train.py"])
    return cli, calls


def _train_argv(signal: Path, bigwig: Path, assay: str = "ATAC") -> list[str]:
    return [
        "--signal", str(signal), "--assay", assay, "--",
        "bias", "train", "-bw", str(bigwig), "-d", assay, "-o", "/out", "--device", "gpu",
    ]  # fmt: skip


def test_main_hands_chrombpnet_its_argv_and_records_peak_rss(
    tmp_path, monkeypatch, fake_chrombpnet
):
    _, calls = fake_chrombpnet
    rss = tmp_path / ".peak_rss_gb"
    monkeypatch.setenv("METADATA_RSS_FILE", str(rss))
    signal, bigwig = _prepare(tmp_path)
    argv = _train_argv(signal, bigwig)
    assert ct.main(argv) == 0
    assert calls == [["chrombpnet", *argv[argv.index("--") + 1 :]]]
    assert float(rss.read_text()) > 0


def test_main_stops_before_chrombpnet_on_a_mismatch(tmp_path, fake_chrombpnet):
    _, calls = fake_chrombpnet
    signal, bigwig = _prepare(tmp_path, assay="DNASE")
    assert ct.main(_train_argv(signal, bigwig)) == 1
    assert calls == []


def test_main_refuses_an_assay_chrombpnet_is_not_given(tmp_path, fake_chrombpnet):
    _, calls = fake_chrombpnet
    signal, bigwig = _prepare(tmp_path)
    argv = _train_argv(signal, bigwig)
    argv[argv.index("-d") + 1] = "DNASE"
    assert ct.main(argv) == 1
    assert calls == []


def test_main_refuses_the_check_without_a_bigwig(tmp_path, fake_chrombpnet):
    _, calls = fake_chrombpnet
    signal, _ = _prepare(tmp_path)
    assert (
        ct.main(
            [
                "--signal",
                str(signal),
                "--assay",
                "ATAC",
                "--",
                "bias",
                "train",
                "-ifrag",
                str(signal),
            ]
        )
        == 1
    )
    assert calls == []


def test_main_records_peak_rss_when_chrombpnet_fails(tmp_path, monkeypatch, fake_chrombpnet):
    cli, _ = fake_chrombpnet

    def boom():
        raise RuntimeError("training blew up")

    cli.main = boom
    rss = tmp_path / ".peak_rss_gb"
    monkeypatch.setenv("METADATA_RSS_FILE", str(rss))
    signal, bigwig = _prepare(tmp_path)
    with pytest.raises(RuntimeError, match="blew up"):
        ct.main(_train_argv(signal, bigwig))
    assert rss.is_file()
