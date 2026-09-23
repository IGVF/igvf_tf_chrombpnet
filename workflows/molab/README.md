# workflows/molab/

Running this pipeline on a **molab** box (marimo cloud): one GPU, no SLURM, no
conda, no `/oak`. `workflows/SLURM/` stays the reference implementation — the
step scripts here are the *same files*, run through a different launcher.

```bash
bash workflows/molab/setup_molab.sh      # once per box
source workflows/molab/env.sh            # once per shell
bash workflows/molab/run_step.sh 00.0.prepare_signal.sh
```

---

## Environments: pixi and Apptainer. No conda.

**There is no conda on a molab box, and none is installed.** Two environments,
and that is all:

| | provides | used by |
|---|---|---|
| **pixi `preprocess`** | python 3.13, pyranges1, pybigtools, pysam, pyfaidx, pybigwig, click | steps 00.0, 00.1, 02.0, and `cli.py download-references` |
| **Apptainer container** | chrombpnet, tensorflow, bedtools, modisco | steps 01.0, 03.0, 03.1, 03.2, 04.0, 04.1 |

The pixi environment is defined in `pixi.toml` (`[feature.preprocess]`) and
mirrors `envs/preprocess.yml` dependency for dependency — keep the two in step.

chrombpnet **cannot** move to pixi, which is why the container stays: chrombpnet
1.0.1 pins `tensorflow==2.8.0`, `numpy==1.23.4` and `protobuf==3.20` against
CUDA 11 wheels, and no such wheel supports this GPU. The container ships a
working TF 2.12 build instead.

`lib/bash/common.sh` still *names* its four environment variables after conda,
because that is what the cluster uses. `env.sh` overrides every one of them —
their defaults are paths under another user's Sherlock home, and nothing here
may reach those. Only `PREPROCESS_ENV` names a real directory; the other three
get a self-describing sentinel (`apptainer:…`, `unconfigured:molab`) and are
never dereferenced, because `CONDA_INIT=""` makes `activate_env` return before
it looks at the path.

## What else is different from the cluster

| | Sherlock | molab |
|---|---|---|
| scheduler | `sbatch`, `--array=N-M` | `run_step.sh --array N-M`, looped in sequence |
| environments | four conda envs | pixi + Apptainer (above) |
| references | `$OAK/engreitz/Data` | `$REFERENCE_ROOT` (default `/marimo/data/references`) |
| modules (`ml cuda/…`) | Lmod | no-ops — `load_*_modules` skips when `ml` is absent |

## How the SLURM steps run without SLURM

Nothing is copied or translated: `run_step.sh` runs the very same file the
cluster submits. That works because **`#SBATCH` lines are bash comments** —
SLURM reads them before executing, and `bash step.sh` ignores them and runs the
body.

The steps read exactly three things from sbatch, and `run_step.sh` sets all
three:

| variable | on the cluster | here |
|---|---|---|
| `SLURM_SUBMIT_DIR` | cwd at submit time | `workflows/SLURM` — the bootstrap uses it to find `REPO_ROOT` |
| `SLURM_ARRAY_TASK_ID` | one value per array task | looped over `--array`, in sequence |
| `SLURM_CPUS_PER_TASK` | from `--cpus-per-task` | `$MOLAB_CPUS` |

### What is silently lost

`--mem`, `--cpus-per-task`, `--gres`, `--partition`, `--time` and
`--constraint` are **ignored, and nothing enforces or reports them.** Several
steps request more than this box has (32 GB):

| step | requests |
|---|---|
| `01.0.preprocess_nonpeaks` | 100G |
| `03.2.qc_selected_bias` | 64G |
| `04.0.train_full_model` | 128G |

Those are padded scheduling requests rather than measured usage — `03.0`'s own
header records a *measured* peak of 23.2 GB against what used to be a 128 GB
request. But nothing here checks, so a step that genuinely exceeds 32 GB is
OOM-killed by the kernel. Per CLAUDE.md a SIGKILL writes no metadata record at
all, so **the missing record is the signal**: a step that dies leaving no JSON
under `${metadata_dir}` was almost certainly out of memory.

Likewise `--gres=gpu:1` grants nothing: the GPU is simply visible. Running two
GPU steps at once will contend, which is why `run_step.sh` loops array indices
in sequence rather than in parallel.

## The files

- **`setup_molab.sh`** — idempotent, run once. Runs `apt-get update`,
  generates the `en_US.UTF-8` locale, installs Apptainer + its
  userspace mount helpers, downloads the container from the private bucket
  `gs://${GCP_BUCKET}` (7.8 GiB, md5 checked against the bucket), unpacks it to
  a sandbox directory, **deletes the `.sif`** the moment the unpack returns,
  checks the sandbox against the image's listing (counted before extraction),
  then bakes the low-memory one-hot encoder (`lib/python/utils/onehot.py`)
  into it and verifies it byte-for-byte against upstream inside the container.
  It then installs the pixi `qc` environment (and
  clears pixi's package cache), and fetches the d0 test data into
  `/marimo/data/test_data_d0/{config,inputs}/` and the shared references
  (reading `reference_root` from `$DATASET_CONFIG`).
  `--skip-references` if you already have them. Needs `GCP_BUCKET` and
  `GCP_SA_JSON` (the service-account key as inline JSON) in a `.env`; there is
  no gcloud — the token is minted with `openssl`. Disk: ~24 GB at the end of
  the unpack, ~16 GB after it, ~22 GB with the references. Everything is also
  written to `/marimo/setup_molab.log` (override with `MOLAB_SETUP_LOG`), which
  is where to look if the session dies. A rerun sees the sandbox's
  `.igvf_molab/unpacked` marker and never touches the image again; to rebuild,
  `rm -rf` the sandbox.
- **`env.sh`** — every variable in one place. Source it per shell.
  **`git push` needs it too.** The credential helper it installs expands
  `${GITHUB_TOKEN}` at push time rather than storing it, so a shell that has
  not sourced `env.sh` (or `.env`) pushes an empty password and GitHub replies
  `Invalid username or token` — which reads exactly like a bad token and is
  not one. `set -a; source workflows/molab/.env; set +a` is enough.
- **`run_step.sh`** — runs ONE step in the right environment. Knows which steps
  need which, emulates `--array`, writes a log per index to `$MOLAB_LOG_DIR`.
  `--list` prints every step with its environment, `--dry-run` shows what would
  run without running it.
- **`run_all.sh`** — steps 00.0 → 04.1 in order, written out as literal
  commands you can read or copy one line at a time. Stops at 03.1 for the
  manual bias-model review; `--after-bias` resumes.
- **`sync_to_gcs.sh`** — copies `results/`, a `repo.bundle` of every branch,
  a `MANIFEST.txt` and the dataset config to
  `gs://${GCP_BUCKET}/chrombpnet/test_data_d0/` (override the prefix with
  `MOLAB_GCS_PREFIX`). **Run it after every step**: a session that dies
  keeps only part of `/marimo`, and the commits on this box exist nowhere
  else. Files the bucket already holds byte-for-byte (size + md5) are
  skipped, uploads are md5-checked, and nothing remote is ever deleted.
  `--bundle-only` for just the commits, `--dry-run` to list. Needs a service
  account allowed to write under the prefix; no gcloud.
- **`gcs.sh`** — the bucket helpers `setup_molab.sh` and `sync_to_gcs.sh`
  share: an OAuth token minted from `GCP_SA_JSON` with `openssl` (scope
  `read_only` for setup, `read_write` for the sync), downloads, uploads,
  listings. Sourced, never run.

### Where are the step scripts?

Not here, on purpose. They live in **`workflows/SLURM/`** and are shared with
the cluster — that directory is the one definition of what each step *does*,
and molab only changes how it is *launched*. Copying them would mean two
versions to keep in step. To see them:

```bash
bash workflows/molab/run_step.sh --list
```

```
STEP                               ENV        ARRAY
00.0.prepare_signal.sh             pixi       —
00.1.preprocess_peaks.sh           pixi       —
01.0.preprocess_nonpeaks.sh        container  —
02.0.qc_training_data.sh            pixi       —
03.0.train_bias_model.sh           container  0-19
...
```

The `ARRAY` column is the step's own `#SBATCH --array` default, which assumes
5 folds × 4 bias factors. `run_all.sh` derives the real range from your config
instead (1 fold × 4 factors → `0-3` for `d0`).

## Why the container is unpacked instead of run as a `.sif`

molab runs under **gVisor**, which has no loop devices and no kernel squashfs,
and where `fuse-overlayfs` fails outright (`cannot read lower dirs: Function
not implemented`). Apptainer therefore cannot mount a `.sif` at all. Three
consequences, all handled by `setup_molab.sh`:

1. The image is unpacked with `unsquashfs` into a plain directory, which
   Apptainer execs using underlay bind mounts only.
2. `apptainer.conf` is switched to `enable overlay = no` / `enable underlay = yes`.
3. `unsquashfs` exits 2 because gVisor forbids `mknod`, so device nodes cannot
   be created. That is harmless — but it makes `apptainer build --sandbox`
   delete its own output, which is why the image is unsquashed directly.

## The GPU

An RTX PRO 6000 (Blackwell, **compute capability 12.0**). The container's
TensorFlow has no compiled kernels for it and JIT-compiles from PTX on first
use. Two things make that bearable, both set by `run_step.sh`:

- The image sets `CUDA_CACHE_DISABLE=1`, which silently throws away every
  JIT'd kernel — so the cost would be re-paid on *every* run. It is overridden
  to `0`, with `CUDA_CACHE_PATH` pointed at a persistent directory.
- With a cold cache the first training step takes ~120 s and the first
  `predict` ~20 s; warm, ~8 s and ~3 s. Steady-state step time (~0.6 s) is
  unaffected either way. **The first run on a fresh box is slow. That is
  expected, not a hang.**

Note `03.0` and `04.0` carry `#SBATCH --constraint="GPU_CC:7.0|…|8.6"`, which
excludes this hardware by design. It is inert off SLURM, but it is why the
pipeline "doesn't run on this GPU" as written.

## The box lies about its resources

`nproc` and `free` report the host (20+ cores, 160 GB), not the slice you get
(4 CPUs, 32 GB). Unpinned, TensorFlow and OpenMP size their thread pools to the
phantom count and thrash. `env.sh` sets `MOLAB_CPUS=4` and `run_step.sh`
propagates it to `OMP_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS` and friends.
**Change `MOLAB_CPUS` if your box differs.**

## Order to run

`run_all.sh` does all of this; the explicit form is below so you can run any
single step. QC comes *after* the negatives — `02.0` compares peaks against
them.

```bash
source workflows/molab/env.sh

bash workflows/molab/run_step.sh 00.0.prepare_signal.sh
bash workflows/molab/run_step.sh 00.1.preprocess_peaks.sh
bash workflows/molab/run_step.sh 01.0.preprocess_nonpeaks.sh
bash workflows/molab/run_step.sh 02.0.qc_training_data.sh      # advisory, never fails the run

# 1 fold x 4 bias factors -> indices 0-3. See `cli.py config validate` for the range.
bash workflows/molab/run_step.sh --array 0-3 03.0.train_bias_model.sh
bash workflows/molab/run_step.sh 03.1.select_bias.sh
#   ... then copy the winners from selected_bias_per_fold.tsv into
#   fold_bias_suffix in config/<dataset>/config.yaml. This hand-off is
#   deliberate: the plots are meant to be reviewed.
bash workflows/molab/run_step.sh --array 0 03.2.qc_selected_bias.sh
bash workflows/molab/run_step.sh --array 0 04.0.train_full_model.sh
bash workflows/molab/run_step.sh 04.1.qc_run_full_model.sh
```

## Adding a dataset

```bash
cp -r config/example_dataset config/my_dataset
$EDITOR config/my_dataset/config.yaml
pixi run -e preprocess python src/cli.py config validate --dataset my_dataset
export DATASET=my_dataset
```

`config/d0/config.yaml` is a worked example for this box: absolute paths
outside the checkout, `reference_root` pointing at the local reference tree.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `conda init script not found: /home/groups/engreitz/...` | `env.sh` not sourced. It sets `CONDA_INIT=""`, which makes `activate_env` a no-op. |
| `TypeError: 'type' object is not subscriptable` | A `src/` script using `list[str]` on the container's python 3.8. Needs `from __future__ import annotations`. |
| First GPU step hangs for ~2 min | sm_120 PTX JIT on a cold cache. See **The GPU**. |
| `bedtools: command not found` in 00.x | Step dispatched to pixi but needs the container — check the table in `run_step.sh`. |
| `FileExistsError: ..._auxiliary/` in 01.0 | A killed `chrombpnet prep nonpeaks`; 01.0 clears a stale one, so re-run the step. |
| `found pyproject.toml without tool.pixi section at directory /marimo` | pixi run from `/marimo`, which holds marimo's own `pyproject.toml`. Run pixi from the checkout, or pass `--manifest-path`. |
| Session dies during setup's unpack | Read `/marimo/setup_molab.log`; each line records the space used on `/`. Rerun: the image and a partial sandbox are cleared and redone. |
