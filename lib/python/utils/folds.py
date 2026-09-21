"""Cross-validation fold splits, read from ``folds/fold_*.json``.

The fold JSONs are the single source of truth: they are what gets passed to
chrombpnet as ``-fl``, so anything that needs to know which chromosomes a fold
held out must read the same file rather than keeping its own copy.
"""

from __future__ import annotations

import json
from pathlib import Path


def folds_dir() -> Path:
    """The repo's ``folds/`` directory, located relative to this file."""
    return Path(__file__).resolve().parents[3] / "folds"


def load_fold(fold, directory=None) -> dict[str, list[str]]:
    """Return the ``{"train": [...], "valid": [...], "test": [...]}`` split for ``fold``."""
    directory = Path(directory) if directory is not None else folds_dir()
    path = directory / f"fold_{fold}.json"
    if not path.exists():
        raise FileNotFoundError(f"no fold definition at {path}")
    return json.loads(path.read_text())


def test_chroms(fold, directory=None) -> list[str]:
    """Held-out (test) chromosomes for ``fold``."""
    return load_fold(fold, directory)["test"]
