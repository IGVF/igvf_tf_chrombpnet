#!/usr/bin/env python3
"""Second stage of a two-stage 03.0 sweep, fed into a running box's GPU queue.

docs/bias-factor-per-fold.md: when the bias factor is shared across folds, the
sweep can run every factor on ONE fold and train only that fold's winner on the
others -- 9 trainings per dataset at five factors instead of 25. Bias models
stay per fold; only the factor is chosen once.

The box's queue holds stage one (the sweep fold's tasks). This watcher runs
beside it, inside the box (`srun --jobid <box> --overlap`), and for each dataset
whose sweep-fold models have all finished it:

  1. loads their metrics with 03.1's load_metrics and picks the winner with
     03.1's select_best, so stage two cannot drift from 03.1's rule;
  2. appends the winner's task for every other fold (index fold_idx * n_factors
     + factor_idx, 03.0's mapping) to the queue, under the same flock the box's
     workers take before popping;
  3. records the pick, its status and whether it sits at an end of the sweep
     in --record.

A sweep-fold task that failed (a non-zero exit in the box's status.tsv) counts
as finished: the winner is chosen from the models that exist, and the record
says how many that was. 03.1 still runs at the end and fold_bias_suffix stays
the person's call; this only decides which models get trained.

    python workflows/dcai/two_stage.py --box-dir <box> --configs <configs> \\
        --record <box>/two_stage.tsv [--sweep-fold 0]

CPU-only and light (one small JSON per model); polls every --interval seconds
and exits once every dataset is queued.
"""

import argparse
import fcntl
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lib" / "python"))
sys.path.insert(0, str(REPO / "src"))

import select_bias_model as sbm  # noqa: E402
from utils import log  # noqa: E402

logger = log.get_logger(__name__)


def yaml_list(text: str, key: str) -> list[str]:
    m = re.search(rf"^{key}:\s*\[(.*)\]\s*$", text, re.M)
    return re.findall(r'"([^"]*)"', m.group(1)) if m else []


def yaml_scalar(text: str, key: str) -> str:
    m = re.search(rf'^{key}:\s*"?([^"\n]*?)"?\s*$', text, re.M)
    return m.group(1) if m else ""


def sweep_of(config: Path) -> dict:
    """What 03.0 reads from a generated config: names, folds, factors, tags."""
    text = config.read_text()
    precision = yaml_scalar(text, "bias_precision")
    patience = yaml_scalar(text, "bias_patience")
    tag = (f"_{precision}" if precision and precision != "default" else "") + (
        f"_p{patience}" if patience and patience != "5" else ""
    )
    return {
        "name": yaml_scalar(text, "dataset_name"),
        "output_dir": Path(yaml_scalar(text, "output_dir")),
        "peak_type": yaml_scalar(text, "peak_type") or "all",
        "folds": yaml_list(text, "folds"),
        "labels": [s.lstrip("_") + tag for s in yaml_list(text, "bias_suffixes_sweep")],
    }


def finished_indices(status: Path, name: str) -> set[int]:
    """03.0 task indices of `name` the box has finished, whatever their exit."""
    done = set()
    if status.exists():
        for line in status.read_text().splitlines():
            cols = line.split("\t")
            m = re.match(r"03\.0\.train_bias_model\[(\d+)\]", cols[1]) if len(cols) > 1 else None
            if cols[0] == name and m:
                done.add(int(m.group(1)))
    return done


def append_tasks(queue: Path, lines: list[str]) -> None:
    with open(f"{queue}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(queue, "a") as q:
            q.writelines(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--box-dir",
        type=Path,
        required=True,
        help="the running box's directory (gpu_queue.tsv, status.tsv)",
    )
    ap.add_argument(
        "--configs", type=Path, nargs="+", required=True, help="the datasets' config.yaml files"
    )
    ap.add_argument("--record", type=Path, required=True, help="TSV of the picks, appended")
    ap.add_argument("--sweep-fold", default="0")
    ap.add_argument("--interval", type=int, default=120)
    log.add_logging_args(ap)
    args = ap.parse_args(argv)
    log.setup_from_args(args)

    queue, status = args.box_dir / "gpu_queue.tsv", args.box_dir / "status.tsv"
    done = set()
    if args.record.exists():
        done = {line.split("\t")[0] for line in args.record.read_text().splitlines()[1:]}
    else:
        args.record.write_text("dataset\tsweep_fold\twinner\tstatus\tedge\tmodels\ttime\n")
    pending = {sweep_of(c)["name"]: c for c in args.configs} if args.configs else {}
    pending = {n: c for n, c in pending.items() if n not in done}
    logger.info("%d dataset(s) waiting for their fold-%s sweep", len(pending), args.sweep_fold)

    while pending:
        for name, config in list(pending.items()):
            s = sweep_of(config)
            fold_idx = s["folds"].index(args.sweep_fold)
            n = len(s["labels"])
            sweep_idx = {fold_idx * n + j for j in range(n)}
            if not sweep_idx <= finished_indices(status, name):
                continue
            df = sbm.load_metrics(s["labels"], name, s["peak_type"], [args.sweep_fold],
                                  bias_models_dir=s["output_dir"] / "bias_models")  # fmt: skip
            if df.empty:
                logger.error(
                    "%s: no fold-%s model finished; stage two not queued", name, args.sweep_fold
                )
                pick, row_status, edge = "", "none", ""
            else:
                df["status"] = df.apply(sbm.classify_row, axis=1)
                pick = sbm.select_best(df)
                row_status = df.set_index("bias").loc[pick, "status"]
                edge = (
                    "low" if pick == s["labels"][0] else "high" if pick == s["labels"][-1] else ""
                )
                j = s["labels"].index(pick)
                lines = [
                    f"{config}\t{k * n + j}\n"
                    for k, f in enumerate(s["folds"])
                    if f != args.sweep_fold
                ]
                append_tasks(queue, lines)
                logger.info("%s: fold %s picks %s (%s%s); queued %d task(s)", name, args.sweep_fold, pick,
                            row_status, f", {edge} edge" if edge else "", len(lines))  # fmt: skip
            with open(args.record, "a") as rec:
                rec.write(f"{name}\t{args.sweep_fold}\t{pick}\t{row_status}\t{edge}\t{len(df)}/{n}\t"
                          f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")  # fmt: skip
            del pending[name]
        if pending:
            time.sleep(args.interval)
    logger.info("every dataset's second stage is queued")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
