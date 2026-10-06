"""A box rewrites only its own datasets' configs, and never loses the bias picks.

workflows/dcai/make_configs.py writes one config per fragments file. A box used
to rewrite all of them with its own bias_precision / bias_patience, which name
the directories a dataset's 03.0 models are in, and with an empty
fold_bias_suffix, the per-fold picks a person copies in after 03.1. These tests
pin --rewrite (only those stems) and the carry-over of the picks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_configs", REPO / "workflows" / "dcai" / "make_configs.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


def _setup(tmp_path: Path, stems: list[str]) -> list[str]:
    frags = tmp_path / "fragments"
    frags.mkdir()
    for stem in stems:
        (frags / f"{stem}.fragments.tsv.gz").write_bytes(b"")
    shifts = tmp_path / "shifts.json"
    shifts.write_text(json.dumps({s: {"plus_shift": 4, "minus_shift": -5} for s in stems}))
    return [
        "--fragments-dir", str(frags), "--dataset-root", str(tmp_path),
        "--shifts", str(shifts), "--configs-dir", str(tmp_path / "configs"),
    ]  # fmt: skip


def _config(tmp_path: Path, stem: str) -> Path:
    return tmp_path / "configs" / f"amsc_{stem}" / "config.yaml"


def test_rewrite_only_named_stems(tmp_path, capsys):
    args = _setup(tmp_path, ["a", "b"])
    assert mc.main(args + ["--bias-precision", "bf16"]) == 0
    before_b = _config(tmp_path, "b").read_text()

    assert mc.main(args + ["--bias-precision", "bf16", "--bias-patience", "3", "--rewrite", "a"]) == 0
    assert "bias_patience: 3" in _config(tmp_path, "a").read_text()
    assert _config(tmp_path, "b").read_text() == before_b
    # Every config is still listed, rewritten or not: the box splits its waves on it.
    listed = capsys.readouterr().out.split()
    assert str(_config(tmp_path, "a")) in listed and str(_config(tmp_path, "b")) in listed


def test_missing_config_is_written_even_outside_rewrite(tmp_path):
    args = _setup(tmp_path, ["a", "b"])
    assert mc.main(args + ["--rewrite", "a"]) == 0
    assert _config(tmp_path, "b").exists()


def test_rewrite_keeps_fold_bias_suffix(tmp_path):
    args = _setup(tmp_path, ["a"])
    assert mc.main(args) == 0
    cfg = _config(tmp_path, "a")
    picks = {"0": "_08_bf16", "1": "_07_bf16", "2": "_08_bf16", "3": "_07_bf16", "4": "_07_bf16"}
    text = cfg.read_text()
    for fold, suffix in picks.items():
        text = text.replace(f'  "{fold}": ""', f'  "{fold}": "{suffix}"')
    cfg.write_text(text)

    assert mc.main(args + ["--bias-patience", "3"]) == 0
    assert mc.existing_fold_bias_suffix(cfg) == picks
    assert "bias_patience: 3" in cfg.read_text()


def test_fresh_config_has_empty_picks(tmp_path):
    args = _setup(tmp_path, ["a"])
    assert mc.main(args) == 0
    assert mc.existing_fold_bias_suffix(_config(tmp_path, "a")) == {f: "" for f in mc.FOLDS}
