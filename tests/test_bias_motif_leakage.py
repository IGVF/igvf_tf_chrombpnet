"""03.4's TF-site screen of TF-MoDISco patterns (src/bias_motif_leakage.py).

The screen reads each pattern's consensus from its contribution-weight matrix
and looks for real TF sites on both strands, instead of trusting tomtom-lite's
labels. These tests pin the consensus trimming, the site matching, the shares,
and the leaky-vs-clean comparison, on TF-MoDISco-shaped h5 files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "lib" / "python"))

h5py = pytest.importorskip("h5py", reason="needs the pixi 'qc' environment")
pd = pytest.importorskip("pandas", reason="needs the pixi 'qc' environment")

import bias_motif_leakage as bml  # noqa: E402


def _cwm(seq: str, weak_flank: str = "", weight: float = 1.0) -> np.ndarray:
    """A contribution-weight matrix: weight on the core's bases, 0.05 on the flanks."""
    full = weak_flank + seq + weak_flank
    cwm = np.zeros((len(full), 4))
    for i, base in enumerate(full):
        core = len(weak_flank) <= i < len(weak_flank) + len(seq)
        cwm[i, "ACGT".index(base)] = weight if core else 0.05
    return cwm


def _write_modisco(path: Path, patterns: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for i, (seq, n) in enumerate(patterns):
            g = h5.create_group(f"pos_patterns/pattern_{i}")
            g.create_dataset("contrib_scores", data=_cwm(seq, weak_flank="GGCC"))
            g.create_dataset("seqlets/n_seqlets", data=np.array([n]))


def test_consensus_keeps_only_the_important_span():
    assert bml.cwm_consensus(_cwm("TGAGTCA", weak_flank="CCGG")) == "TGAGTCA"
    assert bml.cwm_consensus(np.zeros((10, 4))) == ""


@pytest.mark.parametrize(
    "consensus,site",
    [
        ("GGTGAGTCACC", "AP-1"),
        ("TGACTCA", "AP-1"),
        ("TTGGCAGTGAGCCAA", "NFI"),
        ("ATTGCGCAATA", "CEBP"),
        ("TTGCGCAA"[::-1].translate(str.maketrans("ACGT", "TGCA")), "CEBP"),  # reverse strand
        ("CCACCAGGGGGCGC", "CTCF"),
        ("ACATTCCA", "TEAD"),
        ("GGAATG", "TEAD"),  # TEAD on the reverse strand
        ("GGGCGGGGCGG", ""),  # GC box: composition, not on the list
        ("AAAAAAAAAAAA", ""),  # poly-A
        ("GGCTCACGCCTGTAATCCC", ""),  # an Alu fragment
    ],
)
def test_tf_site(consensus, site):
    assert bml.tf_site(consensus) == site


def test_site_counts(tmp_path):
    h5 = tmp_path / "m.h5"
    _write_modisco(h5, [("TGAGTCA", 300), ("AAAAAAAAAA", 600), ("TTGCGCAA", 100)])
    total, sites = bml.site_counts(h5)
    assert total == 1000
    assert sites == {"AP-1": 300, "CEBP": 100}


def _tree(tmp_path: Path, bias: dict[str, list], full: dict[str, list]) -> list[str]:
    for fold, patterns in bias.items():
        prefix = f"bds_all_fold_{fold}"
        for head in ("counts", "profile"):
            _write_modisco(
                tmp_path / "bias_models" / "bias_model_07" / prefix / "evaluation" / f"motif_qc_{head}"
                / f"{prefix}_bias_modisco_results.h5",
                patterns if head == "counts" else [("AAAAAAAA", 500)],
            )  # fmt: skip
    for fold, patterns in full.items():
        _write_modisco(
            tmp_path / "full_models" / f"ds_all_fold_{fold}" / "evaluation" / "motif_qc_counts"
            / "chrombpnet_nobias_modisco_results.h5",
            patterns,
        )  # fmt: skip
    return [
        "--bias-models-dir", str(tmp_path / "bias_models"), "--full-model-dir", str(tmp_path / "full_models"),
        "--dataset", "ds", "--bias-dataset", "bds", "--folds", *bias,
        "--fold-bias", *[f"{f}:_07" for f in bias], "--out-dir", str(tmp_path / "out"),
    ]  # fmt: skip


def test_main_reports_leaky_folds_and_the_comparison(tmp_path):
    args = _tree(
        tmp_path,
        bias={"0": [("TGAGTCA", 500), ("AAAAAA", 500)], "1": [("AAAAAA", 900), ("GGGCGG", 100)]},
        full={"0": [("TGAGTCA", 450), ("CCACCAGGGGGCGC", 100), ("AAAA", 450)],
              "1": [("TGACTCA", 480), ("CCACCAGGGGGCGC", 120), ("AAAA", 400)]},
    )  # fmt: skip
    assert bml.main(args) == 0
    df = pd.read_csv(tmp_path / "out" / "bias_motif_leakage.tsv", sep="\t", dtype={"fold": str})
    bias = df[(df["model"] == "bias") & (df["head"] == "counts")].set_index("fold")["tf_site_share"]
    assert bias["0"] == pytest.approx(0.5) and bias["1"] == 0.0
    full = df[df["model"] == "full"].set_index("fold")
    assert full.loc["1", "tf_site_share"] == pytest.approx(0.6)
    summary = (tmp_path / "out" / "bias_motif_leakage_summary.txt").read_text()
    assert "counts head, bias models: 1 of 2 fold(s)" in summary
    assert "leaky (folds 0): AP-1 45.0-45.0%" in summary
    assert "clean (folds 1): AP-1 48.0-48.0%" in summary


def test_main_without_full_models_says_so(tmp_path):
    args = _tree(tmp_path, bias={"0": [("AAAAAA", 100)]}, full={})
    assert bml.main(args) == 0
    assert "no 04.5 results yet" in (tmp_path / "out" / "bias_motif_leakage_summary.txt").read_text()


def test_main_fails_without_any_03_3_result(tmp_path):
    args = [
        "--bias-models-dir", str(tmp_path / "bias_models"), "--full-model-dir", str(tmp_path / "full_models"),
        "--dataset", "ds", "--bias-dataset", "bds", "--folds", "0", "--fold-bias", "0:_07",
        "--out-dir", str(tmp_path / "out"),
    ]  # fmt: skip
    assert bml.main(args) == 1
