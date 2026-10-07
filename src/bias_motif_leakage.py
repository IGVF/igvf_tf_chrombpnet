#!/usr/bin/env python3
"""Has a selected bias model learned TF motifs, and does that reach the full model?

A bias model should learn the enzyme's cut preference (Tn5 or DNase) and base
composition, not transcription-factor motifs. 03.1 cannot see the difference:
its selection score (the profile head's normalised JSD) rewards a bias model
that has absorbed real TF signal as readily as one that has not, and it tends
to prefer the top of the factor sweep, where that happens most. 03.3 runs
TF-MoDISco on every fold's selected bias model; this reads those results.

For every fold (its bias model, from --fold-bias) and both heads, each
positive TF-MoDISco pattern's consensus -- the argmax base at each position of
its contribution-weight matrix where the position carries at least
--min-importance of the peak position's importance -- is tested on both
strands for the TF sites in TF_SITES. The figure per fold and head is the share
of the head's positive seqlets that sit in such patterns. A clean bias model is
at 0.

Why the consensus and not 03.3's report labels: tomtom-lite matches every
pattern to its closest database motif, and on bias models it puts TF names on
GC-rich stretches, Alu fragments and poly-A runs, so a label screen makes
most counts heads look contaminated. Testing the sequence for the sites
themselves does not.

Where 04.5 has run, the full models' own TF-MoDISco results are read the same
way. If a leaky bias model had absorbed a TF's effect, the full model trained
on top of it would under-represent that TF; the summary compares the full
models' TF-site shares on folds whose bias model is leaky (counts head at
--threshold or more) with folds whose bias model is clean. Two caveats, both
in the summary: a share of seqlets is a proxy for contribution strength, and
TF_SITES matches full NFI palindromes, while full models often learn NFI
half-sites.

Output, in --out-dir:
  bias_motif_leakage.tsv          one row per fold x head x model (bias, full):
                                  positive seqlets, TF-site share, seqlets per site
  bias_motif_leakage_summary.txt  the leaky folds per head, and the comparison

    python src/bias_motif_leakage.py --bias-models-dir <results>/bias_models \\
        --full-model-dir <results>/full_models --dataset <name> --bias-dataset <name> \\
        --peak-type all --folds 0 1 2 3 4 --fold-bias 0:_07 1:_07 ... --out-dir <dir>

Run from the chrombpnet environment (h5py, pandas); 03.4 does.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import log  # noqa: E402

logger = log.get_logger(__name__)

# Core sites, tested on both strands of a pattern's consensus. Patterns are
# called by their first matching site in this order.
TF_SITES = {
    "AP-1": r"TGA[CG]TCA",
    "NFI": r"TGGC.{4,6}GCCA",
    "CEBP": r"TT[GA]C[GA][CT]AA",
    "CTCF": r"CC[AG][GC][CT]AG[AG][GT]GG",
    "TEAD": r"CATTCC",
}
_RC = str.maketrans("ACGT", "TGCA")
HEADS = ("counts", "profile")


def cwm_consensus(cwm: np.ndarray, min_importance: float = 0.3) -> str:
    """Argmax base of each position of a (length x 4) contribution-weight
    matrix, over the span between the first and last position whose total
    absolute weight is at least min_importance of the largest."""
    imp = np.abs(cwm).sum(axis=1)
    if not imp.size or imp.max() <= 0:
        return ""
    keep = np.where(imp >= min_importance * imp.max())[0]
    return "".join("ACGT"[i] for i in np.asarray(cwm)[keep.min() : keep.max() + 1].argmax(axis=1))


def tf_site(consensus: str) -> str:
    """The first TF site in TF_SITES found on either strand, or ""."""
    rc = consensus.translate(_RC)[::-1]
    for name, pattern in TF_SITES.items():
        if re.search(pattern, consensus) or re.search(pattern, rc):
            return name
    return ""


def site_counts(h5_path: Path, min_importance: float = 0.3) -> tuple[int, collections.Counter]:
    """Positive seqlets in a TF-MoDISco results file, and how many sit in
    patterns whose consensus carries each TF site."""
    total, sites = 0, collections.Counter()
    with h5py.File(h5_path, "r") as h5:
        if "pos_patterns" not in h5:
            return 0, sites
        for pattern in h5["pos_patterns"].values():
            n = int(np.asarray(pattern["seqlets"]["n_seqlets"][()]).ravel()[0])
            total += n
            site = tf_site(cwm_consensus(pattern["contrib_scores"][()], min_importance))
            if site:
                sites[site] += n
    return total, sites


def collect(args) -> pd.DataFrame:
    picks = dict(item.split(":", 1) for item in args.fold_bias)
    rows = []
    for fold in args.folds:
        suffix = picks.get(fold)
        if not suffix:
            logger.warning("fold %s: no bias suffix given, skipped", fold)
            continue
        bias_prefix = f"{args.bias_dataset}_{args.peak_type}_fold_{fold}"
        full_prefix = f"{args.dataset}_{args.peak_type}_fold_{fold}"
        for head in HEADS:
            files = {
                "bias": args.bias_models_dir / f"bias_model{suffix}" / bias_prefix / "evaluation"
                / f"motif_qc_{head}" / f"{bias_prefix}_bias_modisco_results.h5",
                "full": args.full_model_dir / full_prefix / "evaluation"
                / f"motif_qc_{head}" / "chrombpnet_nobias_modisco_results.h5",
            }  # fmt: skip
            for model, path in files.items():
                if not path.exists():
                    if model == "bias":
                        logger.warning("fold %s %s: no 03.3 results at %s", fold, head, path)
                    continue
                total, sites = site_counts(path, args.min_importance)
                rows.append({
                    "fold": fold, "head": head, "model": model, "bias_suffix": suffix,
                    "pos_seqlets": total,
                    "tf_site_share": sum(sites.values()) / total if total else 0.0,
                    "sites": ";".join(f"{k}:{v}" for k, v in sites.most_common()),
                })  # fmt: skip
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, threshold: float) -> str:
    lines = []
    bias = df[df["model"] == "bias"]
    for head in HEADS:
        b = bias[bias["head"] == head]
        if b.empty:
            continue
        leaky = b[b["tf_site_share"] >= threshold]
        lines.append(
            f"{head} head, bias models: {len(leaky)} of {len(b)} fold(s) at {100 * threshold:.0f}% or more "
            f"TF-site seqlets (max {100 * b['tf_site_share'].max():.0f}%)"
        )
        for r in leaky.itertuples():
            lines.append(
                f"  fold {r.fold} ({r.bias_suffix}): {100 * r.tf_site_share:.0f}%  {r.sites}"
            )

    full = df[(df["model"] == "full") & (df["head"] == "counts")]
    counts_bias = bias[bias["head"] == "counts"].set_index("fold")["tf_site_share"]
    if full.empty:
        lines.append("full models: no 04.5 results yet; rerun after 04.5 for the comparison")
        return "\n".join(lines) + "\n"
    lines.append("")
    lines.append(
        "full models (04.5, counts head), TF-site share by site, on folds whose bias model is"
    )
    groups = {
        "leaky": [f for f in full["fold"] if counts_bias.get(f, 0) >= threshold],
        "clean": [f for f in full["fold"] if counts_bias.get(f, 0) < threshold],
    }
    for name, folds in groups.items():
        part = full[full["fold"].isin(folds)]
        if part.empty:
            lines.append(f"  {name}: no fold")
            continue
        per_site = collections.defaultdict(list)
        for r in part.itertuples():
            counts = dict(kv.split(":") for kv in r.sites.split(";") if kv)
            for site in TF_SITES:
                per_site[site].append(
                    int(counts.get(site, 0)) / r.pos_seqlets if r.pos_seqlets else 0.0
                )
        shares = "  ".join(
            f"{s} {100 * min(v):.1f}-{100 * max(v):.1f}%" for s, v in per_site.items()
        )
        lines.append(f"  {name} (folds {', '.join(folds)}): {shares}")
    lines.append(
        "If a site's share is clearly lower on leaky folds, the bias models took that TF's effect;"
        " similar ranges mean the leakage did not reach the full models. The share of seqlets is a proxy"
        " for contribution strength, and NFI is matched as a full palindrome (full models often learn"
        " half-sites), so a 0 there says nothing either way."
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bias-models-dir", type=Path, required=True)
    ap.add_argument("--full-model-dir", type=Path, required=True)
    ap.add_argument("--dataset", required=True, help="dataset name in the full models' prefix")
    ap.add_argument("--bias-dataset", required=True, help="dataset name in the bias models' prefix")
    ap.add_argument("--peak-type", default="all")
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--fold-bias", nargs="+", required=True, metavar="FOLD:SUFFIX",
                    help="each fold's selected bias suffix, as fold_bias_suffix in config.yaml")  # fmt: skip
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="a bias model is leaky at this TF-site share of its counts head (default 0.10)")  # fmt: skip
    ap.add_argument("--min-importance", type=float, default=0.3,
                    help="positions kept in a pattern's consensus, as a fraction of its peak (default 0.3)")  # fmt: skip
    ap.add_argument("--out-dir", type=Path, required=True)
    log.add_logging_args(ap)
    args = ap.parse_args(argv)
    log.setup_from_args(args)

    df = collect(args)
    if df.empty or not (df["model"] == "bias").any():
        logger.error("no 03.3 results found for any fold")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "bias_motif_leakage.tsv", sep="\t", index=False)
    summary = summarise(df, args.threshold)
    (args.out_dir / "bias_motif_leakage_summary.txt").write_text(summary)
    for line in summary.splitlines():
        logger.info(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
