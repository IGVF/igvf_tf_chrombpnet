#!/usr/bin/env python3
"""
04.0.select_bias_model.py
Select the best ChromBPNet Tn5 bias model per fold.

Plots (PDF + PNG):
  bias_model_pearsonr_boxplot  – major QC: nonpeaks Pearson r per bias across folds
  bias_model_metrics           – 4-panel bar chart (all key metrics)
  bias_model_selection_heatmap – traffic-light fold × bias heatmap

Tables (TSV):
  all_bias_metrics.tsv         – all raw metrics per bias, fold
  bias_comparison_table.tsv    – wide-format comparison: one row per fold, biases as columns
  selected_bias_per_fold.tsv   – final selection per fold

Text:
  bias_selection_explanation.txt – verbal rationale and action items per fold

Selection criteria (ChromBPNet developer guidelines):
  Counts metrics:
    nonpeaks Pearson r > 0       (required; higher is better)
    peaks Pearson r > -0.3       (warn: -0.3 to -0.5; fail: < -0.5)
    peaks MSE will be high       (expected; not a selection criterion)
  Profile metrics:
    peaks median norm JSD        (higher = better; primary criterion)
    peaks median JSD             (lower = better; secondary tie-break)
    Both JSD metrics are sensitive to read depth.

Usage:
  python 04.0.select_bias_model.py \
    --core-path /oak/stanford/groups/engreitz/Users/opushkar/igvf_tf_collab \
    --biases 05 06 07 08 \
    --folds 0 1 2 3 4 \
    --dataset igvf6_definitive_endoderm \
    --peak-type all
"""

# %%
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

mpl.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": True,
    }
)

# ── thresholds (ChromBPNet developer guidelines) ───────────────────────────────
NONPEAKS_PEARSONR_PASS = 0.0
PEAKS_PEARSONR_WARN = -0.3
PEAKS_PEARSONR_FAIL = -0.5

BIAS_COLORS = {
    "05": "#4393C3",
    "06": "#4C72B0",
    "07": "#DD8452",
    "08": "#55A868",
    "09": "#C44E52",
    "1": "#728FCE",
}
STATUS_COLORS = {"pass": "#2ca02c", "warn": "#ff7f0e", "fail": "#d62728"}


# ── data loading ───────────────────────────────────────────────────────────────


# %%
def load_metrics(
    core_path: Path,
    biases: list[str],
    dataset: str,
    peak_type: str,
    folds: list[str],
) -> pd.DataFrame:
    rows = []
    for bias in biases:
        bias_dir = (
            core_path / dataset / "results" / "bias_models" / f"bias_model_{bias}"
        )
        for fold in folds:
            tag = f"{dataset}_{peak_type}_fold_{fold}"
            path = bias_dir / tag / "evaluation" / f"{tag}_bias_metrics.json"
            if not path.exists():
                print(f"  [MISSING] {path}")
                continue
            with open(path) as f:
                m = json.load(f)
            cm = m["counts_metrics"]
            pm = m["profile_metrics"]
            rows.append(
                {
                    "bias": bias,
                    "fold": fold,
                    "nonpeaks_pearsonr": cm["nonpeaks"]["pearsonr"],
                    "nonpeaks_spearmanr": cm["nonpeaks"]["spearmanr"],
                    "peaks_pearsonr": cm["peaks"]["pearsonr"],
                    "peaks_spearmanr": cm["peaks"]["spearmanr"],
                    "peaks_mse": cm["peaks"]["mse"],
                    "peaks_median_jsd": pm["peaks"]["median_jsd"],
                    "peaks_median_norm_jsd": pm["peaks"]["median_norm_jsd"],
                }
            )
    return pd.DataFrame(rows)


# ── classification and selection ───────────────────────────────────────────────


# %%
def classify_row(row) -> str:
    """Return 'pass', 'warn', or 'fail' for a single (bias, fold) row."""
    np_ok = row["nonpeaks_pearsonr"] > NONPEAKS_PEARSONR_PASS
    pk_ok = row["peaks_pearsonr"] > PEAKS_PEARSONR_WARN
    pk_survivable = row["peaks_pearsonr"] > PEAKS_PEARSONR_FAIL

    if np_ok and pk_ok:
        return "pass"
    elif pk_survivable:
        return "warn"
    else:
        return "fail"


def select_best(group: pd.DataFrame) -> str:
    """Given all bias models for one fold, return the best bias string."""
    group = group.copy()
    group["status"] = group.apply(classify_row, axis=1)

    for tier in ("pass", "warn", "fail"):
        candidates = group[group["status"] == tier]
        if not candidates.empty:
            idx = (
                candidates["peaks_median_norm_jsd"]
                .sub(candidates["peaks_median_jsd"] / 100)
                .idxmax()
            )
            return candidates.loc[idx, "bias"]
    return group["bias"].iloc[0]


def build_selection_table(
    df: pd.DataFrame, overrides: dict[str, str] | None = None
) -> pd.DataFrame:
    """One row per fold: the selected bias model and its metrics.

    overrides: {fold_str: bias_str} from dataset_config.sh fold_bias_suffix.
    Folds present in overrides use that bias directly; others use automated selection.
    """
    records = []
    for fold, grp in df.groupby("fold"):
        fold_str = str(fold)
        if overrides and fold_str in overrides:
            best = overrides[fold_str]
            if best not in grp["bias"].values:
                print(
                    f"  [WARNING] Override bias '{best}' for fold {fold} not in "
                    f"evaluated biases {list(grp['bias'].unique())} — falling back to auto-select."
                )
                best = select_best(grp)
        else:
            best = select_best(grp)
        row = grp.set_index("bias").loc[best].to_dict()
        row["fold"] = fold
        row["selected_bias"] = best
        row["status"] = classify_row(grp.set_index("bias").loc[best])
        records.append(row)
    return pd.DataFrame(records).set_index("fold").sort_index()


# ── comparison table ───────────────────────────────────────────────────────────


# %%
def build_comparison_table(df: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """
    Wide-format comparison table: one row per fold, each bias model's key metrics
    as separate columns. Aids visual comparison across bias models for each fold.

    Columns per bias: np_r, pk_r, norm_jsd, jsd, status.
    Final columns: selected_bias, selected_status.
    """
    df = df.copy()
    df["status"] = df.apply(classify_row, axis=1)

    biases = sorted(df["bias"].unique())
    folds = sorted(df["fold"].unique())

    rows = []
    for fold in folds:
        row: dict = {"fold": fold}
        fold_df = df[df["fold"] == fold]
        for bias in biases:
            sub = fold_df[fold_df["bias"] == bias]
            prefix = f"bias_{bias}"
            if not sub.empty:
                r = sub.iloc[0]
                row[f"{prefix}_np_r"] = round(r["nonpeaks_pearsonr"], 3)
                row[f"{prefix}_pk_r"] = round(r["peaks_pearsonr"], 3)
                row[f"{prefix}_norm_jsd"] = round(r["peaks_median_norm_jsd"], 3)
                row[f"{prefix}_jsd"] = round(r["peaks_median_jsd"], 4)
                row[f"{prefix}_status"] = r["status"]
            else:
                for col in ("np_r", "pk_r", "norm_jsd", "jsd", "status"):
                    row[f"{prefix}_{col}"] = np.nan

        if fold in selection.index:
            sel = selection.loc[fold]
            row["selected_bias"] = sel["selected_bias"]
            row["selected_status"] = sel["status"]
        else:
            row["selected_bias"] = None
            row["selected_status"] = None
        rows.append(row)

    return pd.DataFrame(rows).set_index("fold")


# ── verbal explanation ─────────────────────────────────────────────────────────


# %%
def generate_explanation(
    df: pd.DataFrame,
    selection: pd.DataFrame,
    dataset: str,
    biases: list[str],
    folds: list[str],
) -> str:
    """
    Human-readable report: thresholds, per-fold results, and actionable guidance
    following ChromBPNet developer guidelines.
    """
    df = df.copy()
    df["status"] = df.apply(classify_row, axis=1)

    lines = [
        "ChromBPNet Bias Model Selection Report",
        "=" * 60,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Dataset   : {dataset}",
        f"Folds     : {', '.join(map(str, folds))}",
        f"Biases    : {', '.join(biases)}",
        "",
        "SELECTION CRITERIA (ChromBPNet developer guidelines)",
        "-" * 60,
        "Counts metrics:",
        "  Nonpeaks Pearson r > 0              required; higher is better",
        "  Peaks Pearson r > -0.3              warn if -0.3 to -0.5; fail if < -0.5",
        "  Peaks MSE will be high              expected; not a selection criterion",
        "Profile metrics:",
        "  Peaks median norm JSD               higher is better (primary criterion)",
        "  Peaks median JSD                    lower is better (secondary tie-break)",
        "  Note: JSD metrics are sensitive to read depth; deeper coverage = better values.",
        "",
        "SELECTION LOGIC",
        "-" * 60,
        "  1. Among PASS models (np_r > 0, pk_r > -0.3): select highest peaks norm JSD.",
        "     Tie-break: subtract peaks JSD / 100 to penalize higher JSD slightly.",
        "  2. If no PASS model exists, relax to pk_r > -0.5 (WARN): repeat criterion.",
        "  3. If still none, select the least-bad model (highest peaks Pearson r).",
        "",
        "RESULTS BY FOLD",
        "-" * 60,
    ]

    warn_folds: list = []
    fail_folds: list = []

    for fold in sorted(selection.index):
        sel = selection.loc[fold]
        bias = sel["selected_bias"]
        status = sel["status"].upper()
        flag = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[status]

        sub = df[(df["fold"] == fold) & (df["bias"] == bias)]
        if sub.empty:
            continue
        r = sub.iloc[0]

        np_r = r["nonpeaks_pearsonr"]
        pk_r = r["peaks_pearsonr"]
        norm_jsd = r["peaks_median_norm_jsd"]
        jsd = r["peaks_median_jsd"]

        np_flag = "✓" if np_r > NONPEAKS_PEARSONR_PASS else "✗"
        if pk_r > PEAKS_PEARSONR_WARN:
            pk_flag = "✓"
        elif pk_r > PEAKS_PEARSONR_FAIL:
            pk_flag = "⚠"
        else:
            pk_flag = "✗"

        lines += [
            f"  Fold {fold}: bias_{bias}  [{status}]  {flag}",
            f"    Nonpeaks Pearson r    = {np_r:+.3f}  (threshold > 0)  {np_flag}",
            f"    Peaks Pearson r       = {pk_r:+.3f}  (warn < -0.3; fail < -0.5)  {pk_flag}",
            f"    Peaks median norm JSD = {norm_jsd:.3f}  (higher is better)",
            f"    Peaks median JSD      = {jsd:.4f}  (lower is better)",
        ]

        if status == "PASS":
            lines.append(
                "    Action: Model passes all criteria. Proceed with ChromBPNet training."
            )
        elif status == "WARN":
            warn_folds.append(fold)
            lines += [
                f"    Action: CAUTION — peaks Pearson r ({pk_r:+.3f}) is in the warning zone [-0.5, -0.3).",
                "      The bias model may have learned a GC distribution different from your peaks.",
                f"      After running TFModisco on fold {fold}, inspect the top-10 motifs:",
                "        If > 3 of the top-10 motifs are GC-rich: increase --bias_threshold_factor",
                "          and retrain the bias model.",
                "        If ≤ 3 GC-rich motifs in the top-10: the model is acceptable; proceed.",
            ]
        elif status == "FAIL":
            fail_folds.append(fold)
            lines += [
                f"    Action: FAIL — peaks Pearson r ({pk_r:+.3f}) < -0.5.",
                "      ChromBPNet training will automatically abort with this bias model.",
                "      Increase --bias_threshold_factor and retrain the bias model before proceeding.",
            ]
        lines.append("")

    winner_counts = selection["selected_bias"].value_counts()
    top_bias = winner_counts.index[0]
    top_n = winner_counts.iloc[0]

    lines += [
        "OVERALL RECOMMENDATION",
        "-" * 60,
        f"  bias_{top_bias} wins {top_n}/{len(selection)} folds.",
        f'  → Set bias_suffix="_{top_bias}" in config.sh',
    ]

    if warn_folds:
        lines += [
            "",
            f"  ⚠ WARNING: {len(warn_folds)} fold(s) in the warn zone: {warn_folds}",
            "    After TFModisco: check if > 3 of the top-10 motifs are GC-rich.",
            "    If so, increase --bias_threshold_factor and retrain.",
        ]
    if fail_folds:
        lines += [
            "",
            f"  ✗ FAIL: {len(fail_folds)} fold(s) failed threshold: {fail_folds}",
            "    These folds require retraining with a higher --bias_threshold_factor.",
        ]

    return "\n".join(lines)


# ── plots ──────────────────────────────────────────────────────────────────────


# %%
def plot_boxplot(df: pd.DataFrame, selection: pd.DataFrame, out_stem: Path) -> None:
    """
    Major QC plot: distribution of nonpeaks Pearson r across folds per bias model.
    Nonpeaks Pearson r must be > 0 (ChromBPNet guideline).
    Selected (fold, bias) pairs are highlighted in navy; others in gray.
    """
    biases = sorted(df["bias"].unique())

    selected_pairs = {
        (str(row["fold"]), row["selected_bias"])
        for _, row in selection.reset_index().iterrows()
    }

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 5))

    bp_data = [df[df["bias"] == b]["nonpeaks_pearsonr"].values for b in biases]
    ax.boxplot(
        bp_data,
        positions=range(len(biases)),
        widths=0.35,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="lightgray", color="black", alpha=0.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    for xi, bias in enumerate(biases):
        grp = df[df["bias"] == bias].reset_index(drop=True)
        jitter = rng.uniform(-0.08, 0.08, len(grp))
        for i, row in grp.iterrows():
            is_sel = (str(row["fold"]), bias) in selected_pairs
            ax.scatter(
                xi + jitter[i],
                row["nonpeaks_pearsonr"],
                color="darkblue" if is_sel else "gray",
                s=22,
                zorder=4 if is_sel else 3,
                edgecolors="black",
                linewidth=0.5,
            )
            if is_sel:
                ax.text(
                    xi + jitter[i] + 0.12,
                    row["nonpeaks_pearsonr"],
                    f"fold {row['fold']}",
                    fontsize=7,
                    va="center",
                )

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="darkblue",
            markersize=7,
            label="Selected per fold",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=7,
            label="Other folds",
        ),
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=8, loc="lower right")

    ax.set_xticks(range(len(biases)))
    ax.set_xticklabels([f"bias_{b}" for b in biases], fontsize=9)
    ax.set_xlabel("Bias model", fontsize=10)
    ax.set_ylabel("Pearson r (nonpeaks)", fontsize=10)
    ax.set_title("Bias model fit on non-peak regions\n(should be > 0)", fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)

    out_data = df[["bias", "fold", "nonpeaks_pearsonr"]].copy()
    out_data["selected"] = out_data.apply(
        lambda r: (str(r["fold"]), r["bias"]) in selected_pairs, axis=1
    )
    out_data.to_csv(
        out_stem.parent / (out_stem.name + "_data.tsv"), sep="\t", index=False
    )
    print(f"Saved: {out_stem}.pdf/.png")


# %%
def plot_boxplot_by_fold(df: pd.DataFrame, out_stem: Path) -> None:
    """
    Companion to plot_boxplot: folds on X axis, bias models as colored points.
    Shows how all bias models compare within each fold.
    """
    folds = sorted(df["fold"].unique())
    biases = sorted(df["bias"].unique())

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 5))

    bp_data = [df[df["fold"] == f]["nonpeaks_pearsonr"].values for f in folds]
    ax.boxplot(
        bp_data,
        positions=range(len(folds)),
        widths=0.35,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="lightgray", color="black", alpha=0.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    for xi, fold in enumerate(folds):
        grp = df[df["fold"] == fold].reset_index(drop=True)
        jitter = rng.uniform(-0.08, 0.08, len(grp))
        for i, row in grp.iterrows():
            ax.scatter(
                xi + jitter[i],
                row["nonpeaks_pearsonr"],
                color=BIAS_COLORS.get(row["bias"], "gray"),
                s=22,
                zorder=3,
                edgecolors="none",
                linewidth=0,
            )

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=BIAS_COLORS.get(b, "gray"),
            markersize=7,
            label=f"bias_{b}",
        )
        for b in biases
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=8, loc="lower right")
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"fold {f}" for f in folds], fontsize=9)
    ax.set_xlabel("Fold", fontsize=10)
    ax.set_ylabel("Pearson r (nonpeaks)", fontsize=10)
    ax.set_title("Bias model fit on non-peak regions\n(should be > 0)", fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)

    df[["bias", "fold", "nonpeaks_pearsonr"]].to_csv(
        out_stem.parent / (out_stem.name + "_data.tsv"), sep="\t", index=False
    )
    print(f"Saved: {out_stem}.pdf/.png")


# %%
METRIC_CONFIG = [
    (
        "nonpeaks_pearsonr",
        "Nonpeaks Pearson r\n(should be > 0)",
        NONPEAKS_PEARSONR_PASS,
        "above",
        None,
    ),
    (
        "peaks_pearsonr",
        "Peaks Pearson r\n(should be > -0.3)",
        PEAKS_PEARSONR_WARN,
        "above",
        PEAKS_PEARSONR_FAIL,
    ),
    (
        "peaks_median_jsd",
        "Peaks median JSD\n(lower = better)",
        None,
        None,
        None,
    ),
    (
        "peaks_median_norm_jsd",
        "Peaks median norm JSD\n(higher = better)",
        None,
        None,
        None,
    ),
]


def plot_metrics(df: pd.DataFrame, selection: pd.DataFrame, out_stem: Path) -> None:
    """4-panel grouped bar chart: all key metrics across folds, coloured by bias model."""
    folds = sorted(df["fold"].unique())
    biases = sorted(df["bias"].unique())
    n_fold = len(folds)
    n_bias = len(biases)
    w = 0.8 / n_bias
    x = np.arange(n_fold)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric, ylabel, thresh1, direction, thresh2) in zip(axes, METRIC_CONFIG):
        for i, bias in enumerate(biases):
            vals = [
                df[(df["bias"] == bias) & (df["fold"] == f)][metric].values
                for f in folds
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(
                x + (i - n_bias / 2 + 0.5) * w,
                vals,
                width=w * 0.9,
                color=BIAS_COLORS.get(bias, "gray"),
                label=f"bias_{bias}",
                alpha=0.85,
                zorder=3,
            )

        if thresh1 is not None:
            ax.axhline(
                thresh1,
                color="black",
                lw=1.2,
                ls="--",
                label=f"threshold = {thresh1}",
                zorder=4,
            )
        if thresh2 is not None:
            ax.axhline(
                thresh2,
                color="red",
                lw=1.0,
                ls=":",
                label=f"fail < {thresh2}",
                zorder=4,
            )

        ylims = ax.get_ylim()
        if direction == "above" and thresh1 is not None:
            ax.axhspan(ylims[0], thresh1, color="tomato", alpha=0.08, zorder=1)
        if thresh2 is not None:
            ax.axhspan(ylims[0], thresh2, color="red", alpha=0.06, zorder=0)

        for fi, fold in enumerate(folds):
            if fold not in selection.index:
                continue
            best_bias = selection.loc[fold, "selected_bias"]
            val = df[(df["bias"] == best_bias) & (df["fold"] == fold)][metric].values
            if len(val):
                bi = biases.index(best_bias)
                xpos = fi + (bi - n_bias / 2 + 0.5) * w
                ax.annotate(
                    "★",
                    xy=(xpos, val[0]),
                    xytext=(xpos, val[0] + 0.003),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=BIAS_COLORS.get(best_bias, "black"),
                    zorder=5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"fold {f}" for f in folds])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "ChromBPNet bias model evaluation\n(★ = selected per fold)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)

    df[["bias", "fold"] + [m for m, *_ in METRIC_CONFIG]].to_csv(
        out_stem.parent / (out_stem.name + "_data.tsv"), sep="\t", index=False
    )
    print(f"Saved: {out_stem}.pdf/.png")


# %%
def plot_selection_heatmap(
    df: pd.DataFrame, selection: pd.DataFrame, out_stem: Path
) -> None:
    """
    Traffic-light heatmap: rows = folds, columns = bias models.
    Cell colour = pass (green) / warn (orange) / fail (red).
    Selected model per fold is outlined in navy with a checkmark.
    """
    folds = sorted(df["fold"].unique())
    biases = sorted(df["bias"].unique())

    status_mat = pd.DataFrame(index=folds, columns=biases, dtype=str)
    for _, row in df.iterrows():
        status_mat.loc[row["fold"], row["bias"]] = classify_row(row)

    color_map = {
        "pass": "#2ca02c",
        "warn": "#ff7f0e",
        "fail": "#d62728",
        "": "lightgray",
    }

    fig, ax = plt.subplots(figsize=(len(biases) * 2.8 + 1, len(folds) * 1.4 + 1.5))

    for ri, fold in enumerate(folds):
        for ci, bias in enumerate(biases):
            st = status_mat.loc[fold, bias]
            fc = mpl.colors.to_rgba(color_map.get(st, "lightgray"), alpha=0.35)
            rect = mpatches.FancyBboxPatch(
                (ci + 0.05, ri + 0.05),
                0.9,
                0.9,
                boxstyle="round,pad=0.05",
                linewidth=1,
                edgecolor="gray",
                facecolor=fc,
                zorder=2,
            )
            ax.add_patch(rect)

            sub = df[(df["bias"] == bias) & (df["fold"] == fold)]
            if not sub.empty:
                r = sub.iloc[0]
                ax.text(
                    ci + 0.5,
                    ri + 0.72,
                    f"np_r={r['nonpeaks_pearsonr']:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
                ax.text(
                    ci + 0.5,
                    ri + 0.52,
                    f"pk_r={r['peaks_pearsonr']:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
                ax.text(
                    ci + 0.5,
                    ri + 0.32,
                    f"nJSD={r['peaks_median_norm_jsd']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )

            if fold in selection.index and selection.loc[fold, "selected_bias"] == bias:
                outline = mpatches.FancyBboxPatch(
                    (ci + 0.02, ri + 0.02),
                    0.96,
                    0.96,
                    boxstyle="round,pad=0.05",
                    linewidth=3,
                    edgecolor="navy",
                    facecolor="none",
                    zorder=5,
                )
                ax.add_patch(outline)
                ax.text(
                    ci + 0.88,
                    ri + 0.88,
                    "✓",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="navy",
                    fontweight="bold",
                    zorder=6,
                )

    ax.set_xlim(0, len(biases))
    ax.set_ylim(0, len(folds))
    ax.set_xticks([c + 0.5 for c in range(len(biases))])
    ax.set_xticklabels(
        [f"bias_{b}\n(thresh {int(b) / 10:.1f})" for b in biases], fontsize=10
    )
    ax.set_yticks([r + 0.5 for r in range(len(folds))])
    ax.set_yticklabels([f"fold {f}" for f in folds], fontsize=10)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_patches = [
        mpatches.Patch(
            color=color_map["pass"],
            alpha=0.5,
            label="PASS  (np_r > 0, pk_r > -0.3)",
        ),
        mpatches.Patch(
            color=color_map["warn"],
            alpha=0.5,
            label="WARN  (pk_r in [-0.5, -0.3])",
        ),
        mpatches.Patch(color=color_map["fail"], alpha=0.5, label="FAIL  (pk_r < -0.5)"),
        mpatches.Patch(
            facecolor="none",
            linewidth=2,
            edgecolor="navy",
            label="Selected (✓)",
        ),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    fig.suptitle(
        "Bias model selection per fold\n"
        "np_r = nonpeaks Pearson r   pk_r = peaks Pearson r   nJSD = peaks median norm JSD",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"Saved: {out_stem}.pdf/.png")


# ── scatter plot: peaks vs nonpeaks pearsonr ──────────────────────────────────


# %%
def plot_pearsonr_scatter(
    df: pd.DataFrame, selection: pd.DataFrame, out_stem: Path
) -> None:
    """
    Scatter plot of peaks Pearson r (x) vs nonpeaks Pearson r (y).
    Each point is one (bias, fold) pair, coloured by bias model.
    Selected (fold, bias) pairs are outlined in navy.
    Threshold lines mark ChromBPNet pass/warn/fail boundaries.
    """
    biases = sorted(df["bias"].unique())
    selected_pairs = {
        (str(row["fold"]), row["selected_bias"])
        for _, row in selection.reset_index().iterrows()
    }

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for bias in biases:
        grp = df[df["bias"] == bias]
        color = BIAS_COLORS.get(bias, "gray")
        for _, row in grp.iterrows():
            is_sel = (str(row["fold"]), bias) in selected_pairs
            ax.scatter(
                row["peaks_pearsonr"],
                row["nonpeaks_pearsonr"],
                color=color,
                s=55,
                zorder=4 if is_sel else 3,
                edgecolors="navy" if is_sel else "none",
                linewidth=1.5 if is_sel else 0,
            )
            ax.text(
                row["peaks_pearsonr"] + 0.003,
                row["nonpeaks_pearsonr"],
                f"f{row['fold']}",
                fontsize=7,
                va="center",
                color="black",
            )

    ax.axhline(
        NONPEAKS_PEARSONR_PASS,
        color="black",
        lw=0.8,
        ls="--",
        alpha=0.5,
        label="nonpeaks threshold (0)",
    )
    ax.axvline(
        PEAKS_PEARSONR_WARN,
        color="#ff7f0e",
        lw=0.8,
        ls="--",
        alpha=0.7,
        label="peaks warn (-0.3)",
    )
    ax.axvline(
        PEAKS_PEARSONR_FAIL,
        color="#d62728",
        lw=0.8,
        ls=":",
        alpha=0.7,
        label="peaks fail (-0.5)",
    )

    legend_bias = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=BIAS_COLORS.get(b, "gray"),
            markersize=7,
            label=f"bias_{b}",
        )
        for b in biases
    ]
    legend_sel = mlines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markersize=8,
        markeredgecolor="navy",
        markeredgewidth=1.5,
        label="Selected per fold",
    )
    legend_thresh = [
        mlines.Line2D(
            [0], [0], color="black", lw=0.8, ls="--", label="nonpeaks threshold (0)"
        ),
        mlines.Line2D(
            [0], [0], color="#ff7f0e", lw=0.8, ls="--", label="peaks warn (-0.3)"
        ),
        mlines.Line2D(
            [0], [0], color="#d62728", lw=0.8, ls=":", label="peaks fail (-0.5)"
        ),
    ]
    ax.legend(
        handles=legend_bias + [legend_sel] + legend_thresh,
        frameon=False,
        fontsize=8,
        loc="upper left",
    )

    ax.set_xlabel("Pearson r (peaks)", fontsize=10)
    ax.set_ylabel("Pearson r (nonpeaks)", fontsize=10)
    ax.set_title("Bias model counts fit\npeaks vs nonpeaks Pearson r", fontsize=10)

    n = len(df)
    ax.text(
        0.97, 0.97, f"n={n}", transform=ax.transAxes, fontsize=8, ha="right", va="top"
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)

    out_data = df[["bias", "fold", "peaks_pearsonr", "nonpeaks_pearsonr"]].copy()
    out_data["selected"] = out_data.apply(
        lambda r: (str(r["fold"]), r["bias"]) in selected_pairs, axis=1
    )
    out_data.to_csv(
        out_stem.parent / (out_stem.name + "_data.tsv"), sep="\t", index=False
    )
    print(f"Saved: {out_stem}.pdf/.png")


# ── console summary ────────────────────────────────────────────────────────────


# %%
def print_summary(selection: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("BIAS MODEL SELECTION SUMMARY")
    print("=" * 70)
    cols = [
        "selected_bias",
        "status",
        "nonpeaks_pearsonr",
        "peaks_pearsonr",
        "peaks_median_jsd",
        "peaks_median_norm_jsd",
    ]
    print(selection[cols].to_string(float_format="{:+.4f}".format))
    print("=" * 70)
    print("\nRecommended bias model per fold:")
    for fold, row in selection.iterrows():
        flag = {"pass": "✓", "warn": "⚠", "fail": "✗"}[row["status"]]
        print(
            f"  fold {fold}: bias_{row['selected_bias']}  {flag}  "
            f"[np_r={row['nonpeaks_pearsonr']:+.3f}  "
            f"pk_r={row['peaks_pearsonr']:+.3f}  "
            f"nJSD={row['peaks_median_norm_jsd']:.3f}]"
        )
    winner_counts = selection["selected_bias"].value_counts()
    top_bias = winner_counts.index[0]
    top_n = winner_counts.iloc[0]
    print(
        f"\nOverall: bias_{top_bias} wins {top_n}/{len(selection)} folds → "
        f'set bias_suffix="_{top_bias}" in config.sh\n'
    )


# %%
def parse_args():
    p = argparse.ArgumentParser(description="Select ChromBPNet bias model per fold")
    p.add_argument("--core-path", required=True, help="Project root directory")
    p.add_argument("--biases", nargs="+", default=["05", "06", "07", "08"])
    p.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. igvf6_definitive_endoderm)",
    )
    p.add_argument("--peak-type", default="all")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <core-path>/results/plots/bias_model_selection/<dataset>)",
    )
    p.add_argument("--save-plots", action="store_true", default=True)
    p.add_argument(
        "--fold-bias",
        nargs="*",
        metavar="FOLD:BIAS",
        default=[],
        help=(
            "Override automated selection for specific folds with values from "
            "dataset_config.sh. Format: fold:bias, e.g. --fold-bias 0:07 4:07."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(args.core_path)
        / "results"
        / "plots"
        / "bias_model_selection"
        / args.dataset
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(
        core_path=Path(args.core_path),
        biases=args.biases,
        dataset=args.dataset,
        peak_type=args.peak_type,
        folds=args.folds,
    )

    if df.empty:
        print("No metrics found. Check that bias models have been evaluated.")
        return

    overrides = {}
    for item in args.fold_bias or []:
        fold, bias = item.split(":")
        overrides[fold] = bias.lstrip("_")
    if overrides:
        print(f"Applying fold-bias overrides from dataset_config.sh: {overrides}")

    selection = build_selection_table(df, overrides=overrides or None)
    comparison = build_comparison_table(df, selection)

    df.to_csv(out_dir / "all_bias_metrics.tsv", sep="\t", index=False)
    selection.to_csv(out_dir / "selected_bias_per_fold.tsv", sep="\t")
    comparison.to_csv(out_dir / "bias_comparison_table.tsv", sep="\t")
    print(f"Tables written to {out_dir}/")

    explanation = generate_explanation(
        df, selection, args.dataset, args.biases, args.folds
    )
    (out_dir / "bias_selection_explanation.txt").write_text(explanation)
    print(f"Explanation written to {out_dir}/bias_selection_explanation.txt")

    if args.save_plots:
        plot_boxplot(df, selection, out_dir / "bias_model_pearsonr_boxplot")
        plot_boxplot_by_fold(df, out_dir / "bias_model_pearsonr_boxplot_by_fold")
        plot_metrics(df, selection, out_dir / "bias_model_metrics")
        plot_selection_heatmap(df, selection, out_dir / "bias_model_selection_heatmap")
        plot_pearsonr_scatter(df, selection, out_dir / "bias_model_pearsonr_scatter")

    print_summary(selection)
    print(f"All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
