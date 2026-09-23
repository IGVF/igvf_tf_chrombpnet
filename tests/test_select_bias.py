"""The bias-selection sweep must say when its own range was the constraint.

`select_best` answers "which of the factors we tried is best". It cannot
answer "is the best one outside what we tried" -- and when the winner sits at
an end of the swept range, those two questions have different answers and the
metrics look identical either way. These tests pin the detection of that case,
and the ordering rule it depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "lib" / "python"))

pytest.importorskip("matplotlib", reason="needs the pixi 'qc' environment")

import select_bias_model as sb  # noqa: E402

SWEEP = ["05", "06", "07", "08"]


# ── ordering ─────────────────────────────────────────────────────────────────


def test_sweep_is_ordered_numerically_not_lexically():
    """The bug this guards: "10" sorts before "5" as a string."""
    assert sb.sweep_order(["5", "10", "7"]) == ["5", "7", "10"]


def test_sweep_order_handles_both_label_spellings():
    assert sb.sweep_order(["08", "05", "06"]) == ["05", "06", "08"]
    assert sb.sweep_order(["0.8", "0.5", "0.6"]) == ["0.5", "0.6", "0.8"]


def test_sweep_order_falls_back_to_the_given_order_for_non_numeric_labels():
    labels = ["strict", "loose", "default"]
    assert sb.sweep_order(labels) == labels


# ── edge detection ───────────────────────────────────────────────────────────


def test_lowest_factor_selected_is_flagged_low():
    assert sb.sweep_edge("05", SWEEP) == "low"


def test_highest_factor_selected_is_flagged_high():
    assert sb.sweep_edge("08", SWEEP) == "high"


@pytest.mark.parametrize("bias", ["06", "07"])
def test_interior_selections_are_not_flagged(bias):
    assert sb.sweep_edge(bias, SWEEP) == ""


def test_edge_is_decided_numerically_not_lexically():
    """With labels "5" and "10", a lexical rule would call "10" the low end."""
    assert sb.sweep_edge("10", ["5", "7", "10"]) == "high"
    assert sb.sweep_edge("5", ["5", "7", "10"]) == "low"


def test_a_single_value_sweep_is_flagged():
    """Nothing was compared, so "best" is meaningless; widening is the only move."""
    assert sb.sweep_edge("05", ["05"]) == "low"


def test_unswept_bias_is_not_flagged():
    assert sb.sweep_edge("09", SWEEP) == ""


# ── the flag reaches the selection table ─────────────────────────────────────


def metrics_frame(rows):
    """(fold, bias, norm_jsd) -> the columns select_best and classify_row read."""
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "bias": bias,
                "nonpeaks_pearsonr": 0.5,
                "nonpeaks_spearmanr": 0.5,
                "peaks_pearsonr": -0.1,
                "peaks_spearmanr": -0.1,
                "peaks_mse": 1.0,
                "peaks_median_jsd": 0.5,
                "peaks_median_norm_jsd": norm_jsd,
            }
            for fold, bias, norm_jsd in rows
        ]
    )


def test_selection_table_flags_a_fold_that_chose_the_lowest_factor():
    # fold 0 peaks at the bottom of the sweep, fold 1 in the middle.
    df = metrics_frame(
        [("0", "05", 0.90), ("0", "06", 0.50), ("0", "07", 0.40), ("0", "08", 0.30)]
        + [("1", "05", 0.30), ("1", "06", 0.90), ("1", "07", 0.40), ("1", "08", 0.30)]
    )
    sel = sb.build_selection_table(df)
    assert sel.loc["0", "selected_bias"] == "05"
    assert sel.loc["0", "sweep_edge"] == "low"
    assert sel.loc["1", "sweep_edge"] == ""


def test_selection_table_flags_a_fold_that_chose_the_highest_factor():
    df = metrics_frame(
        [("0", "05", 0.30), ("0", "06", 0.40), ("0", "07", 0.50), ("0", "08", 0.90)]
    )
    sel = sb.build_selection_table(df)
    assert sel.loc["0", "selected_bias"] == "08"
    assert sel.loc["0", "sweep_edge"] == "high"


# ── and reaches the written explanation ──────────────────────────────────────


def explanation_for(df):
    sel = sb.build_selection_table(df)
    return sb.generate_explanation(
        df,
        sel,
        dataset="d",
        biases=sorted(df["bias"].unique()),
        folds=sorted(df["fold"].unique()),
    )


def test_explanation_tells_the_reader_to_widen_the_sweep_downward():
    df = metrics_frame(
        [("0", "05", 0.90), ("0", "06", 0.50), ("0", "07", 0.40), ("0", "08", 0.30)]
    )
    text = explanation_for(df)
    assert "EDGE OF THE SWEPT RANGE" in text
    assert "LOWEST" in text
    assert "lower --bias_threshold_factor" in text


def test_explanation_tells_the_reader_to_widen_the_sweep_upward():
    df = metrics_frame(
        [("0", "05", 0.30), ("0", "06", 0.40), ("0", "07", 0.50), ("0", "08", 0.90)]
    )
    text = explanation_for(df)
    assert "HIGHEST" in text
    assert "higher --bias_threshold_factor" in text


def test_explanation_stays_quiet_when_the_winner_is_interior():
    """The warning has to be rare enough to mean something."""
    df = metrics_frame(
        [("0", "05", 0.30), ("0", "06", 0.90), ("0", "07", 0.50), ("0", "08", 0.30)]
    )
    assert "EDGE OF THE SWEPT RANGE" not in explanation_for(df)


def test_explanation_names_the_scored_edge_not_the_requested_one():
    """The d0 run: 02.0's scan requested 28 factors, 3 were trained. The winner
    (065) is the highest *scored* factor, and the text used to say the fold
    'chose the HIGHEST factor (bias_195)' -- a factor that never ran."""
    df = metrics_frame([("0", "05", 0.30), ("0", "055", 0.40), ("0", "065", 0.90)])
    requested = ["015", "03", "05", "055", "065", "07", "105", "195"]
    sel = sb.build_selection_table(df)
    text = sb.generate_explanation(df, sel, dataset="d", biases=requested, folds=["0"])
    assert "HIGHEST factor (bias_065)" in text
    assert "bias_195" not in text
    assert "3 of 8 requested have metrics" in text


def test_explanation_gives_the_yaml_key_not_config_sh():
    df = metrics_frame([("0", "05", 0.30), ("0", "06", 0.90), ("0", "07", 0.50)])
    text = explanation_for(df)
    assert "config.sh" not in text
    assert 'fold_bias_suffix:' in text and '"0": "_06"' in text
