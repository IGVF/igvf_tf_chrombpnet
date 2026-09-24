"""The 04.0 / 04.4 / 04.5 split must interpret what `chrombpnet pipeline` would.

04.0 now stops `chrombpnet pipeline` before interpretation and 04.4 does it
from the outputs left behind. That is only a faithful split if 04.4 picks the
same 30K regions pipeline would, and if the wrapper really does stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))
sys.path.insert(0, str(REPO / "src"))

from utils import regions  # noqa: E402


def _chrombpnet_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim from chrombpnet/pipelines.py (bias QC and full pipeline alike)."""
    if df.shape[0] > 30000:
        return df.sample(30000, random_state=1234)
    return df


def _peaks(n: int) -> pd.DataFrame:
    return pd.DataFrame({"chr": ["chr1"] * n, "start": range(n), "end": range(1, n + 1), "summit": [0] * n})


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
    assert pd.read_csv(out, sep="\t", header=None).values.tolist() == pd.read_csv(src, sep="\t", header=None).values.tolist()


def test_wrapper_parses_stop_before_interpretation():
    import chrombpnet_train as ct

    prepared, require, stop, theirs = ct._split_argv(
        ["--prepared-bigwig", "/p", "--stop-before-interpretation", "--", "pipeline", "-o", "/out"]
    )
    assert (prepared, require, stop) == ("/p", False, True)
    assert theirs == ["pipeline", "-o", "/out"]
    assert ct._split_argv(["--", "bias", "train"])[2] is False


def test_stop_never_imports_the_real_interpret_module(monkeypatch):
    """The real module turns eager execution off at import; importing it before
    training broke find_chrombpnet_hyperparams. The stop must be a stand-in."""
    import chrombpnet_train as ct

    import types

    monkeypatch.delitem(sys.modules, ct.INTERPRET_MODULE, raising=False)
    # chrombpnet is not installed here: stand in its (empty) parent packages.
    for name in ("chrombpnet", "chrombpnet.evaluation", "chrombpnet.evaluation.interpret"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    stub = ct._install_interpretation_stop()
    parent = sys.modules["chrombpnet.evaluation.interpret"]
    assert parent.interpret is stub  # what `import ... as interpret` binds on py3.8
    try:
        assert sys.modules[ct.INTERPRET_MODULE] is stub
        assert "tensorflow" not in getattr(stub, "__dict__", {})
        try:
            stub.main(None)
        except ct._StopBeforeInterpretation:
            pass
        else:  # pragma: no cover
            raise AssertionError("stand-in main did not stop the pipeline")
    finally:
        sys.modules.pop(ct.INTERPRET_MODULE, None)
