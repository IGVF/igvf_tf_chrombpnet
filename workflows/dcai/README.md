# workflows/dcai — whole-node runs on the DCAI cluster

The DCAI cluster (`hpc.ite.dcai.dk`, partition `defq`) hands out a whole node per
job: 216 usable cores, ~2 TB RAM and 8x H100 80GB, billed by node-time. Submitting
the `workflows/SLURM/` steps one dataset at a time would rent a node for a
4-core job. Like `../molab/`, this directory changes only how the step files are
launched. It runs many datasets' steps at once inside one exclusive allocation;
the steps themselves are the same files.

| File | What it does |
|---|---|
| `env.sh` | site paths: pixi, `CHROMBPNET_REPO`, `REFERENCE_ROOT`, `DATASET_ROOT`, `SBATCH_ACCOUNT` |
| `run_box.sbatch` | one exclusive node (`--exclusive --gres=gpu:8 --cpus-per-task=216 --mem=0`) running `run_box.sh` |
| `run_box.sh` | phase 0: shifts + configs; wave 1: `run_chain.sh` for the first datasets at once, their 03.0 tasks on GPU workers; wave 2: the rest, sized from wave 1's memory; then 03.1; `checks.sh` and a resource sampler alongside |
| `run_chain.sh` | one dataset: `{00.0.call_peaks ∥ 00.0.prepare_signal} → 00.1 → 01.0 → 02.0`, then (with `bias`) queue its 03.0 tasks |
| `detect_shifts.py` | Tn5 shift of every fragments file, sampled genome-wide; scPrinter's detector cross-checked with chrombpnet's |
| `make_configs.py` | one `config.yaml` per fragments file, shift pinned |
| `checks.sh` | GPU/driver probe; figwig == numpy bigwig on a full library; MACS3 PR #756 == the 3.0.5 release (and how 3.0.4 differs) |
| `compare_bigwigs.py` | per-base comparison of two bigWigs |

## Setup, once

```bash
cd /path/to/igvf_tf_chrombpnet && source workflows/dcai/env.sh
CONDA_OVERRIDE_CUDA=13.0 pixi install --locked --manifest-path "$CHROMBPNET_REPO/pyproject.toml" -e cuda13
pixi install -e preprocess && pixi install -e peaks      # peaks builds MACS3 from source
pixi run -e preprocess python src/cli.py download-references
```

## Run

```bash
source workflows/dcai/env.sh
BOX_FIRST_N=10 BOX_STAGES=prep,bias BOX_BACKFILL=prep \
    sbatch --chdir "$DATASET_ROOT/chrombpnet" workflows/dcai/run_box.sbatch
```

| Variable | Default | Meaning |
|---|---|---|
| `BOX_FIRST` / `BOX_FIRST_N` | the 10 largest libraries | wave 1: named file stems, or the N largest |
| `BOX_STAGES` | `prep,bias` | wave 1: CPU steps 00.0–02.0, then 03.0 on the GPUs |
| `BOX_BACKFILL` | `prep` | wave 2, the remaining datasets: `none`, `prep` or `prep,bias` |
| `BOX_GPU_SLOTS_PER_GPU` | 2 | concurrent 03.0 trainings per GPU |
| `BOX_MEM_FRACTION` | 0.85 | share of RAM wave 2 may plan to fill |

03.1 runs at the end for every dataset whose sweep ran. It writes
`selected_bias_per_fold.tsv`; copying the winners into the config's
`fold_bias_suffix` is the pipeline's deliberate review point before 03.2/04.0.

Each fragments file `<stem>.fragments.tsv.gz` under `$DATASET_ROOT/fragments` becomes
dataset `amsc_<stem>`. Outputs:

- `$DATASET_ROOT/peaks/<dataset>/<dataset>.narrowPeak.gz`, with `call_peaks.json`
  (recipe, shift, counts at each filter, MACS time and RSS) and `reps/`
- `$DATASET_ROOT/bigwigs/<dataset>.unstranded.bw` → the prepared bigwig
- `$DATASET_ROOT/chrombpnet/<dataset>/`: the steps' `output_dir`
- `$DATASET_ROOT/chrombpnet/configs/<dataset>/config.yaml` and `shifts.json`
- `$DATASET_ROOT/chrombpnet/box/<job id>/`: `status.tsv` (dataset, step, exit,
  seconds; 03.0 rows carry `[index]@gpu<N>`), `logs/<dataset>/<step>.log`,
  `resources.tsv` (every 30 s: memory, `/dev/shm`, load, CPU busy %, per-GPU
  utilisation and memory), `sizing.txt` (how wave 2 was sized), `checks/summary.tsv`

Resubmitting resumes: every step skips work whose outputs exist. Phase 0 reuses
`shifts.json` once it exists; delete it to re-detect.

## Sizing

Wave 1 runs all its datasets at once. A chain's peak is its three concurrent
MACS3 `callpeak` runs plus the pseudoreplicate insertion files in scratch (`/raid`
if the node has it, else `/dev/shm`, which counts against the job's memory).
Wave 2's concurrency is then computed, not guessed: the peak memory over wave 1
(from `resources.tsv`, minus what was in use before it) per chain, against
`BOX_MEM_FRACTION` of the node minus what is in use when wave 2 starts -- which
includes the GPU tasks running by then. Every step's run metadata also records
its peak RSS and wall time, and `call_peaks.json` the per-call MACS numbers.

GPU tasks are one 03.0 training each, `BOX_GPU_SLOTS_PER_GPU` per device
(`CUDA_VISIBLE_DEVICES` pinned). chrombpnet disables JAX's memory preallocation,
so several share an 80 GB H100; `resources.tsv` shows whether more would fit.
