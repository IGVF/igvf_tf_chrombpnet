#!/usr/bin/env python3
"""Is the bias threshold factor a per-fold quantity? docs/bias-factor-per-fold.md, run.

Reads 03.1's all_bias_metrics.tsv for every dataset given and answers the doc's
three questions with 03.1's own selector (select_best / classify_row imported
from src/select_bias_model.py, so the counterfactual cannot drift from it):

  Q1  do the folds agree?   distinct per-fold winners per dataset
  Q2  signal or noise?      spread of the selection score across FACTORS within
                            a fold vs across FOLDS at a fixed factor
  Q3  the counterfactual    each fold scored at the dataset's modal winner instead
                            of its own: relative score loss vs the 2% tie epsilon
                            select_best already uses, and any pass/warn/fail change

and prints the decision rule fixed in advance in the doc:

  any status change                        -> keep the full grid
  every |delta| <= 2% and no status change -> switch 03.0 to two stages
  otherwise                                -> keep the grid

It also counts winners on an edge of the swept range (03.1's sweep_edge), which
the doc's HEP3B run showed can make a "shared" factor the wrong one to share.
CPU-only and small (one TSV per dataset), so it is safe on a login node.

    python workflows/dcai/bias_factor_per_fold.py \\
        /dcai/projects/iu_0109/datasets/amsc/chrombpnet/amsc_*/plots/bias_model_selection/*/all_bias_metrics.tsv

Run it from the `qc` environment (pandas): `pixi run -e qc python ...`.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lib" / "python"))
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

import select_bias_model as sbm  # noqa: E402
from utils import log  # noqa: E402

logger = log.get_logger(__name__)


def score(rows: pd.DataFrame) -> pd.Series:
    """select_best's ranking score."""
    return rows["peaks_median_norm_jsd"] - rows["peaks_median_jsd"] / 100


def analyse(name: str, df: pd.DataFrame) -> dict:
    df = df.copy()
    df["fold"] = df["fold"].astype(str)
    df["bias"] = df["bias"].astype(str)
    df["score"] = score(df)
    df["status"] = df.apply(sbm.classify_row, axis=1)
    biases = sbm.sweep_order(df["bias"].unique())

    # Q1: per-fold winners, by 03.1's own rule.
    winners = {fold: sbm.select_best(group) for fold, group in df.groupby("fold")}
    counts = Counter(winners.values())
    # The modal winner; a tie goes to the factor with the best mean score.
    top = max(counts.values())
    modes = [b for b, n in counts.items() if n == top]
    shared = max(modes, key=lambda b: df.loc[df["bias"] == b, "score"].mean())

    # Q2: spread across factors within a fold vs across folds at a factor.
    within = df.groupby("fold")["score"].agg(lambda s: s.max() - s.min()).mean()
    across = df.groupby("bias")["score"].agg(lambda s: s.max() - s.min()).mean()

    # Q3: the counterfactual.
    rows = []
    for fold, own in winners.items():
        g = df[df["fold"] == fold].set_index("bias")
        if shared not in g.index:
            rows.append((fold, own, shared, float("nan"), g.loc[own, "status"], "missing"))
            continue
        s_own, s_shared = g.loc[own, "score"], g.loc[shared, "score"]
        delta = (s_own - s_shared) / abs(s_own) if s_own else 0.0
        rows.append((fold, own, shared, delta, g.loc[own, "status"], g.loc[shared, "status"]))
    q3 = pd.DataFrame(rows, columns=["fold", "own", "shared", "rel_delta", "status_own", "status_shared"])

    edge = sum(1 for b in winners.values() if b in (biases[0], biases[-1]))
    return {
        "dataset": name,
        "factors": len(biases),
        "folds": len(winners),
        "distinct_winners": len(counts),
        "winners": " ".join(f"{f}:{b}" for f, b in sorted(winners.items())),
        "shared": shared,
        "edge_winners": edge,
        "spread_within_fold": within,
        "spread_across_folds": across,
        "max_rel_delta": q3["rel_delta"].abs().max(),
        "status_changes": int((q3["status_own"] != q3["status_shared"]).sum()),
        "q3": q3,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("metrics", nargs="+", type=Path, help="03.1's all_bias_metrics.tsv, one per dataset")
    ap.add_argument("--out", type=Path, default=None, help="also write the per-dataset table here (TSV)")
    log.add_logging_args(ap)
    args = ap.parse_args(argv)
    log.setup_from_args(args)

    eps = sbm.NORM_JSD_SCORE_TIE_REL_EPS
    results = []
    for path in args.metrics:
        name = path.parent.name
        df = pd.read_csv(path, sep="\t", dtype={"bias": str, "fold": str})
        if df["fold"].nunique() < 2:
            logger.warning("%s has %d fold(s); per-fold questions need at least 2, skipped", name, df["fold"].nunique())
            continue
        results.append(analyse(name, df))
    if not results:
        logger.error("no dataset with at least two folds")
        return 1

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "q3"} for r in results])
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(table.drop(columns="winners").round(5).to_string(index=False))
        print()
        for r in results:
            print(f"{r['dataset']}: winners {r['winners']}  (shared {r['shared']})")
    if args.out:
        table.to_csv(args.out, sep="\t", index=False)

    n_change = int(table["status_changes"].sum())
    worst = float(table["max_rel_delta"].max())
    print()
    print(f"Q1  datasets whose folds all agree: {(table['distinct_winners'] == 1).sum()} of {len(table)}")
    print(f"Q2  mean spread across folds / within folds: "
          f"{table['spread_across_folds'].mean():.5f} / {table['spread_within_fold'].mean():.5f}"
          f"  (ratio {table['spread_across_folds'].mean() / table['spread_within_fold'].mean():.2f})")
    print(f"Q3  max relative loss {100 * worst:.2f}% (tie epsilon {100 * eps:.0f}%), status changes: {n_change}")
    print(f"    winners on an edge of the swept range: {int(table['edge_winners'].sum())} of {int(table['folds'].sum())} folds")
    if n_change:
        verdict = "KEEP THE FULL GRID (a fold changes status under the shared factor)"
    elif worst <= eps:
        verdict = "SWITCH 03.0 TO TWO STAGES (every loss within the tie epsilon, no status change)"
    else:
        verdict = "KEEP THE GRID (loss beyond the tie epsilon, but no status change)"
    print(f"Decision rule: {verdict}")
    if int(table["edge_winners"].sum()):
        print("Caveat: some winners sit on an edge of the sweep -- the doc's HEP3B note says widen the grid first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
