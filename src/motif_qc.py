#!/usr/bin/env python3
"""
motif_qc.py
Per-fold motif QC on one head's DeepLIFT scores: TF-MoDISco 2.5.2 on a small
seqlet budget, its descriptive report, and a MEME export of the patterns.
03.3 (bias model) and 04.5 (full model) run it once per head.

Why a small budget (-n 5000 by default, where chrombpnet's pipeline uses
50000): this is QC, whose question is "which motifs did this fold's model
learn?", and TF-MoDISco's runtime is dominated by clustering, which grows with
the seqlet count. On d0's bias model (fold 0, 30K regions, 4 shared CPUs)
TF-MoDISco took 63 min at -n 50000 on the profile head (2.0.7), and 6 min at
-n 5000 on the counts head (2.5.2) -- which still separated the GC-rich
positive pattern from the AT-rich negative ones cleanly. Analysis-grade motifs
come from 08.0 at the full budget on the fold average.

Why both heads: chrombpnet's pipeline runs TF-MoDISco on profile scores only.
The profile head explains the SHAPE of the cut distribution, where Tn5 lives;
the counts head explains HOW MANY cuts a region gets, and that is where a bias
model absorbs accessibility (GC content on d0) without any of it showing in
the profile patterns.

Why not per-seqlet annotation (tangermeme recursive seqlets + tomtom-lite,
as cherimoya's pipeline does): tried and dropped. Annotated seqlet by seqlet
against chrombpnet's DB, 1% of the d0 bias model's seqlets matched Tn5, while
the same model's TF-MoDISco patterns were 90% Tn5 -- the Tn5 motif is long and
weak per position, and recognisable only in aggregate.

The algorithm is unchanged from the chrombpnet env: modiscolite 2.5.2's
core/affinitymat/cluster/extract_seqlets are identical to 2.0.7. 2.5.2 adds
the descriptive report, `modisco meme`, and the window size recorded in the h5.

Stages, each skipped when its output exists (--force re-runs them):
  modisco  `modisco motifs -i <scores.h5> -n <max-seqlets> -w <window>`
  report   `modisco report -l` (descriptive, tomtom-lite matching against
           --motif-db) and `modisco meme -t PFM`

Input:
  --scores   chrombpnet's DeepLIFT h5 for one head (shap/seq hypothetical,
             raw/seq one-hot, both (N, 4, L)) -- 03.2 or 04.4 output
Output (under --output-dir):
  <prefix>modisco_results.h5, report/report.html, <prefix>modisco_motifs.meme,
  <prefix>motif_qc.json (settings, per-stage seconds, pattern counts)

Usage:
  python motif_qc.py --scores .../d0_all_fold_0_bias.counts_scores.h5 \\
      --motif-db chrombpnet-1.0.1.motifs.meme.txt \\
      --output-dir .../evaluation/motif_qc_counts --prefix d0_all_fold_0_bias
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Make lib/python importable without an install step (works under the cluster
# conda envs, under pixi, and under a bare python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "python"))

from utils import log, metadata  # noqa: E402

logger = log.get_logger(__name__)


def _tool(name: str) -> str:
    """A console script of THIS interpreter's env, whether or not it is on PATH."""
    local = Path(sys.executable).parent / name
    return str(local) if local.exists() else name


def parse_args():
    p = argparse.ArgumentParser(description="TF-MoDISco motif QC on one head's DeepLIFT scores.")
    p.add_argument("--scores", required=True, help="chrombpnet DeepLIFT scores h5 (one head)")
    p.add_argument("--motif-db", required=True, help="MEME motif database for the report")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="", help="File-name prefix, e.g. d0_all_fold_0_bias")
    p.add_argument(
        "--window", type=int, default=400, help="Central bp TF-MoDISco runs over (modisco -w)"
    )
    p.add_argument(
        "--max-seqlets",
        type=int,
        default=5000,
        help="modisco -n, per metacluster (chrombpnet's pipeline: 50000)",
    )
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--force", action="store_true", help="Re-run stages whose outputs exist")
    log.add_logging_args(p)
    args = p.parse_args()
    # numba (modiscolite, memelite) sizes its pool to the cores the kernel
    # reports -- the HOST's on a container or molab box (20+), not the
    # allocation. Pin it before anything imports it; modisco inherits it.
    for var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    return args


def _modisco(*argv: str) -> None:
    logger.info("  $ modisco %s", " ".join(argv))
    subprocess.run([_tool("modisco"), *argv], check=True)


def _n_patterns(h5: Path) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401  # modisco 2.5 may write filter-compressed h5

    with h5py.File(h5, "r") as f:
        return {g: len(f[g]) if g in f else 0 for g in ("pos_patterns", "neg_patterns")}


def main():
    args = parse_args()
    log.setup_from_args(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    px = f"{args.prefix}_" if args.prefix else ""
    f = {
        "modisco": out_dir / f"{px}modisco_results.h5",
        "report": out_dir / "report",
        "meme": out_dir / f"{px}modisco_motifs.meme",
        "summary": out_dir / f"{px}motif_qc.json",
    }
    summary = json.loads(f["summary"].read_text()) if f["summary"].exists() else {}
    summary.update(
        scores=str(args.scores),
        window=args.window,
        max_seqlets=args.max_seqlets,
        motif_db=str(args.motif_db),
    )
    seconds = summary.setdefault("seconds", {})

    def stage(name, done, fn):
        if done and not args.force:
            logger.info("[%s] outputs exist, skipping", name)
            return
        logger.info("[%s] start", name)
        t0 = time.monotonic()
        fn()
        seconds[name] = round(time.monotonic() - t0, 1)
        logger.info("[%s] done in %.1f s", name, seconds[name])
        f["summary"].write_text(json.dumps(summary, indent=2) + "\n")

    def _motifs():
        cmd = ["motifs", "-i", str(args.scores), "-n", str(args.max_seqlets)]
        cmd += ["-w", str(args.window), "-o", str(f["modisco"])]
        _modisco(*cmd, *(["-v"] if args.verbose else []))

    def _report():
        cmd = ["report", "-i", str(f["modisco"]), "-o", f"{f['report']}/", "-s", "./"]
        _modisco(*cmd, "-m", str(args.motif_db), "-l")
        _modisco("meme", "-i", str(f["modisco"]), "-t", "PFM", "-o", str(f["meme"]), "-q")

    stage("modisco", f["modisco"].exists(), _motifs)
    stage("report", (f["report"] / "report.html").exists() and f["meme"].exists(), _report)
    summary.update(_n_patterns(f["modisco"]))
    f["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("motif QC complete -> %s", out_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        metadata.report_peak_rss()  # TF-MoDISco runs as a child; its peak counts
