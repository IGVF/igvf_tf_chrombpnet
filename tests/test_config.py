"""Pin the config loader.

The risk this file exists for: the built-in parser is a YAML *subset*, and a
silently misparsed config would be worse than any crash. So it is tested
against PyYAML on every config in the repo, and required to raise — not guess —
on anything it does not handle.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))

from utils import config  # noqa: E402

REPO_CONFIGS = sorted((REPO / "config").glob("*/config.yaml"))


def test_repo_ships_configs():
    assert REPO_CONFIGS, "no config/*/config.yaml found"


# ── the fallback must agree with real YAML ────────────────────────────────────


@pytest.mark.parametrize("path", REPO_CONFIGS, ids=lambda p: p.parent.name)
def test_builtin_parser_matches_pyyaml(path):
    yaml = pytest.importorskip("yaml")
    assert config._parse_builtin(path.read_text()) == yaml.safe_load(path.read_text())


@pytest.mark.parametrize("path", REPO_CONFIGS, ids=lambda p: p.parent.name)
def test_every_repo_config_renders_to_shell(path):
    shell = config.to_shell(config.load(path))
    assert "datasets=(" in shell
    assert "declare -A fold_bias_suffix=(" in shell


# ── it must refuse what it cannot do ──────────────────────────────────────────


@pytest.mark.parametrize(
    "snippet",
    [
        "key: &anchor value",  # anchors
        "key: *alias",  # aliases
        "key: |\n  block text",  # literal block
        "key: >\n  folded text",  # folded block
        "key: !!python/object x",  # tags
    ],
)
def test_unsupported_yaml_raises_rather_than_guessing(snippet):
    with pytest.raises(config.ConfigError):
        config._parse_builtin(snippet)


def test_deep_nesting_raises():
    with pytest.raises(config.ConfigError, match="one level of nesting"):
        config._parse_builtin("a:\n  b:\n    c: 1\n")


def test_stray_indentation_raises():
    with pytest.raises(config.ConfigError, match="unexpected indentation"):
        config._parse_builtin("  orphan: 1\n")


# ── shape ─────────────────────────────────────────────────────────────────────


def test_scalars_lists_and_maps_round_trip():
    cfg = config._parse_builtin('name: x\nnums: ["1", "2"]\nblock:\n  - a\n  - b\nmap:\n  "0": v\n')
    assert cfg == {"name": "x", "nums": ["1", "2"], "block": ["a", "b"], "map": {"0": "v"}}


def test_interpolation_is_left_for_the_shell():
    """${dataset_dir} must survive to bash, so the value stays double-quoted."""
    shell = config.to_shell({"p": "${dataset_dir}/data"})
    assert shell == 'p="${dataset_dir}/data"'


def test_values_with_spaces_are_quoted():
    assert config.to_shell({"p": "/a b/c"}) == "p='/a b/c'"


def test_key_that_is_not_a_shell_identifier_is_rejected():
    with pytest.raises(config.ConfigError):
        config.to_shell({"not-a-var": 1})


def test_comments_and_blank_lines_ignored():
    assert config._parse_builtin("# c\n\nkey: v  # trailing\n") == {"key": "v"}


# ── the bootstrap path: system python, no conda, no third-party ───────────────


def test_loader_runs_on_bare_system_python():
    """lib/bash/config.sh calls this before any env is active."""
    py = "/usr/bin/python3"
    if not Path(py).exists():
        pytest.skip("no /usr/bin/python3")
    out = subprocess.run(
        [py, str(REPO / "lib" / "python" / "utils" / "config.py"), "export", str(REPO_CONFIGS[0])],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    assert "datasets=(" in out.stdout


def test_missing_config_reports_cleanly(tmp_path):
    with pytest.raises(config.ConfigError, match="no config file"):
        config.load(tmp_path / "nope.yaml")


# ── signal type is inferred from the extension ────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("reads.bam", "bam"),
        ("reads.tagAlign", "tagalign"),
        ("reads.tagAlign.gz", "tagalign"),
        ("d_atac_fragments.tsv.gz", "fragments"),
        ("x.fragments.tsv.gz", "fragments"),
        ("x.fragments.tsv", "fragments"),
        ("signal.bw", "bigwig"),
        ("signal.bigWig", "bigwig"),
        ("/ABS/Path/READS.BAM", "bam"),
    ],
)
def test_signal_type_inferred_from_suffix(name, expected):
    assert config.signal_type_for(name) == expected


def test_longest_suffix_wins():
    """'.fragments.tsv.gz' must not be read as a bare '.gz'."""
    assert config.signal_type_for("a.fragments.tsv.gz") == "fragments"


def test_unknown_extension_is_a_clear_error():
    with pytest.raises(config.ConfigError, match="cannot tell the signal type"):
        config.signal_type_for("reads.cram")


def test_bigwig_is_flagged_as_a_prepared_signal():
    """A bigwig cannot train directly; it is only usable as a prepared bigwig."""
    r = config.resolve({"dataset_name": "d", "signal_path": "x.bw", "folds": ["0"]})
    assert r["signal_type"] == "bigwig"
    assert r["signal_is_prepared_bigwig"] is True


def test_reads_are_not_flagged_as_prepared():
    r = config.resolve({"dataset_name": "d", "signal_path": "x.tsv.gz", "folds": ["0"]})
    assert r["signal_is_prepared_bigwig"] is False


# ── derived keys come from one place ──────────────────────────────────────────


def test_dataset_name_drives_datasets_and_bias_dataset():
    r = config.resolve({"dataset_name": "d1", "signal_path": "x.bam", "folds": ["0", "1"]})
    assert r["datasets"] == ["d1"] and r["bias_dataset"] == "d1"


def test_bias_sweep_folds_defaults_to_all_folds():
    r = config.resolve({"dataset_name": "d", "signal_path": "x.bam", "folds": ["0", "1"]})
    assert r["bias_sweep_folds"] == ["0", "1"]


def test_explicit_bias_sweep_folds_is_kept():
    r = config.resolve(
        {
            "dataset_name": "d",
            "signal_path": "x.bam",
            "folds": ["0", "1"],
            "bias_sweep_folds": ["0"],
        }
    )
    assert r["bias_sweep_folds"] == ["0"]


def test_assay_and_peak_type_default():
    r = config.resolve({"dataset_name": "d", "signal_path": "x.bam"})
    assert r["assay"] == "ATAC" and r["peak_type"] == "all"


def test_shipped_configs_name_themselves_after_their_folder():
    """DATASET=<folder> selects the config; dataset_name names the outputs.

    If they disagree you silently get outputs under a different name, which is
    what an unfinished template copy looks like.
    """
    for path in REPO_CONFIGS:
        assert config.load(path)["dataset_name"] == path.parent.name, path
