#!/usr/bin/env python3
"""Bias-model QC (03.2 then 03.3) for every fold's selected bias model, inside a
running box, as soon as each model exists.

The box's GPU workers only run 03.0. This runs beside them (`srun --jobid <box>
--overlap`) and, for every config's folds, waits for the model fold_bias_suffix
names, then runs 03.2 (predictions + DeepLIFT, on a GPU) and 03.3 (TF-MoDISco
motif QC, on CPUs). A fold whose 03.3 summaries both exist is done and skipped,
so a restart resumes.

Sharing the node: while 03.0 trainings are running, each GPU gets at most
--gpu-slots-busy 03.2 runs and --cpu-jobs-busy 03.3 runs share the CPUs with
the trainings' loaders; once no training is left the limits rise to
--gpu-slots-idle / --cpu-jobs-idle.

Threads: every job runs under a CPU mask (taskset) -- 16 cores for 03.2,
--modisco-threads for 03.3 -- because XLA sizes its thread pools to the cores a
process may use: an unmasked JAX process holds ~750 threads on a 224-core node,
masked to 16 it holds 75, and the box's trainings already hold ~445 each against
a per-user limit of 32768 (ulimit -u, threads included). A job that cannot
start (EAGAIN) goes back in the queue instead of stopping this.

Keeping the box: the box's workers exit once the queue is empty and
<box>/.chains_done exists. This removes that file at start, so the box stays up
for the QC, and puts it back when every fold is done or failed and a grace
period (--grace-min) has passed and no <box>/.hold* file is left (full_chain.sh
holds the box with .hold.<dataset>; a person can with .hold) -- the box then
finishes as it would have (03.1, summary, exit).

Record: <box>/bias_qc.tsv (dataset, fold, step, exit, seconds); logs under
<box>/logs/<dataset>/03.{2,3}.<fold>.log.

    python workflows/dcai/bias_qc_box.py --box-dir <box> --configs <config.yaml ...>

Environment: the caller's (workflows/dcai/env.sh plus CHROMBPNET_REPO /
CHROMBPNET_PIXI_ENV / BIAS_FACTORS_FROM_SCAN as the box ran 03.0); TMPDIR and
the JAX cache are put on the box's scratch.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def yaml_scalar(text: str, key: str) -> str:
    m = re.search(rf'^{key}:\s*"?([^"\n]*?)"?\s*$', text, re.M)
    return m.group(1) if m else ""


def targets(config: Path) -> list[dict]:
    """One entry per fold: the selected bias model and where its QC lands."""
    text = config.read_text()
    name, out = yaml_scalar(text, "dataset_name"), Path(yaml_scalar(text, "output_dir"))
    peak_type = yaml_scalar(text, "peak_type") or "all"
    folds = re.findall(r'"([^"]*)"', re.search(r"^folds:\s*\[(.*)\]", text, re.M).group(1))
    block = re.search(r"^fold_bias_suffix:\n((?:[ \t]+.*\n?)*)", text, re.M)
    picks = (
        dict(re.findall(r'^[ \t]+"(\d+)":[ \t]*"([^"]*)"', block.group(1), re.M)) if block else {}
    )
    rows = []
    for idx, fold in enumerate(folds):
        suffix = picks.get(fold, "")
        if not suffix:
            continue
        prefix = f"{name}_{peak_type}_fold_{fold}"
        d = out / "bias_models" / f"bias_model{suffix}" / prefix
        rows.append({
            "config": config, "name": name, "fold": fold, "index": idx,
            "ready": d / "evaluation" / f"{prefix}_bias_metrics.json",
            "done": [d / "evaluation" / f"motif_qc_{h}" / f"{prefix}_bias_motif_qc.json" for h in ("counts", "profile")],
        })  # fmt: skip
    return rows


def trainings_running() -> int:
    try:
        r = subprocess.run(
            ["pgrep", "-fc", "03.0.train_bias_model.sh"], capture_output=True, text=True
        )
    except OSError:  # no thread to spare for pgrep: assume the trainings are still there
        return 1
    return int(r.stdout.strip() or 0)


ALLOWED = sorted(os.sched_getaffinity(0))


def cpu_mask(offset: int, n: int) -> str:
    return ",".join(str(ALLOWED[(offset + i) % len(ALLOWED)]) for i in range(n))


def launch(script: str, mask: str, env: dict, log: Path):
    """The step under a CPU mask, or None when no process can be created now."""
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log, "w") as fh:
            return subprocess.Popen(["taskset", "-c", mask, "bash", str(REPO / "workflows/SLURM" / script)],
                                    cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)  # fmt: skip
    except OSError as e:
        print(
            f"[{time.strftime('%F %T')}] could not start {script}: {e}; retrying later", flush=True
        )
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--box-dir", type=Path, required=True)
    ap.add_argument("--configs", type=Path, nargs="+", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-slots-busy", type=int, default=1)
    ap.add_argument("--gpu-slots-idle", type=int, default=4)
    ap.add_argument("--cpu-jobs-busy", type=int, default=5)
    ap.add_argument("--cpu-jobs-idle", type=int, default=6)
    ap.add_argument("--modisco-threads", type=int, default=32)
    ap.add_argument("--grace-min", type=int, default=30)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument(
        "--dry-run", action="store_true", help="print the folds and their state, run nothing"
    )
    args = ap.parse_args(argv)

    box = args.box_dir
    record = box / "bias_qc.tsv"
    if not record.exists() and not args.dry_run:
        record.write_text("dataset\tfold\tstep\texit\tseconds\n")
    todo = [t for c in args.configs for t in targets(c)]
    todo = [t for t in todo if not all(p.exists() for p in t["done"])]
    print(f"[{time.strftime('%F %T')}] {len(todo)} fold(s) to QC", flush=True)
    if args.dry_run:
        for t in todo:
            print(
                f"  {t['name']} fold {t['fold']}: model {'ready' if t['ready'].exists() else 'not yet'}"
            )
        return 0

    (box / ".chains_done").unlink(missing_ok=True)
    scratch = Path(
        f"/dev/shm/{os.environ.get('USER', 'user')}/box.{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    env = dict(os.environ, TMPDIR=str(scratch / "tmp"), TMP=str(scratch / "tmp"), TEMP=str(scratch / "tmp"),
               JAX_COMPILATION_CACHE_DIR=str(scratch / "jax_cache"), XLA_PYTHON_CLIENT_PREALLOCATE="false",
               JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0", JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0")  # fmt: skip
    (scratch / "tmp").mkdir(parents=True, exist_ok=True)

    gpu_jobs, cpu_jobs = {}, {}  # Popen -> (target, gpu, t0) / (target, t0)
    waiting_cpu = []
    n_started = 0
    while todo or gpu_jobs or cpu_jobs or waiting_cpu:
        busy = trainings_running() > 0
        # finished jobs
        for p, (t, _gpu, t0) in list(gpu_jobs.items()):
            if p.poll() is not None:
                del gpu_jobs[p]
                with open(record, "a") as r:
                    r.write(
                        f"{t['name']}\t{t['fold']}\t03.2\t{p.returncode}\t{int(time.time() - t0)}\n"
                    )
                if p.returncode == 0:
                    waiting_cpu.append(t)
        for p, (t, t0) in list(cpu_jobs.items()):
            if p.poll() is not None:
                del cpu_jobs[p]
                with open(record, "a") as r:
                    r.write(
                        f"{t['name']}\t{t['fold']}\t03.3\t{p.returncode}\t{int(time.time() - t0)}\n"
                    )
        # start 03.3 runs
        while waiting_cpu and len(cpu_jobs) < (args.cpu_jobs_busy if busy else args.cpu_jobs_idle):
            t = waiting_cpu[0]
            e = dict(env, DATASET_CONFIG=str(t["config"]), SLURM_ARRAY_TASK_ID=str(t["index"]),
                     SLURM_CPUS_PER_TASK=str(args.modisco_threads), CUDA_VISIBLE_DEVICES="")  # fmt: skip
            mask = cpu_mask(args.modisco_threads * n_started, args.modisco_threads)
            p = launch(
                "03.3.modisco_selected_bias.sh",
                mask,
                e,
                box / "logs" / t["name"] / f"03.3.{t['fold']}.log",
            )
            if p is None:
                break
            waiting_cpu.pop(0)
            n_started += 1
            cpu_jobs[p] = (t, time.time())
        # start 03.2 runs on the least-loaded GPU
        limit = args.gpu_slots_busy if busy else args.gpu_slots_idle
        for t in [t for t in todo if t["ready"].exists()]:
            load = {g: 0 for g in range(args.gpus)}
            for _, gpu, _ in gpu_jobs.values():
                load[gpu] += 1
            gpu = min(load, key=load.get)
            if load[gpu] >= limit:
                break
            e = dict(env, DATASET_CONFIG=str(t["config"]), SLURM_ARRAY_TASK_ID=str(t["index"]),
                     SLURM_CPUS_PER_TASK="8", OMP_NUM_THREADS="8", NUMBA_NUM_THREADS="8",
                     CUDA_VISIBLE_DEVICES=str(gpu))  # fmt: skip
            p = launch("03.2.qc_selected_bias.sh", cpu_mask(16 * n_started, 16), e,
                       box / "logs" / t["name"] / f"03.2.{t['fold']}.log")  # fmt: skip
            if p is None:
                break
            todo.remove(t)
            n_started += 1
            gpu_jobs[p] = (t, gpu, time.time())
        # a model that can no longer appear: its 03.0 queue is empty and nothing trains
        if todo and not busy and not (box / "gpu_queue.tsv").stat().st_size and not gpu_jobs:
            for t in [t for t in todo if not t["ready"].exists()]:
                todo.remove(t)
                with open(record, "a") as r:
                    r.write(f"{t['name']}\t{t['fold']}\tno-model\t1\t0\n")
        print(f"[{time.strftime('%F %T')}] waiting {len(todo)}, 03.2 running {len(gpu_jobs)}, "
              f"03.3 queued {len(waiting_cpu)} running {len(cpu_jobs)}, trainings {'on' if busy else 'off'}", flush=True)  # fmt: skip
        time.sleep(args.interval)

    print(
        f"[{time.strftime('%F %T')}] every fold QC'd; releasing the box in {args.grace_min} min once no {box}/.hold* is left",
        flush=True,
    )
    deadline = time.time() + 60 * args.grace_min
    while time.time() < deadline or any(box.glob(".hold*")):
        time.sleep(30)
    (box / ".chains_done").touch()
    print(f"[{time.strftime('%F %T')}] released (.chains_done)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
