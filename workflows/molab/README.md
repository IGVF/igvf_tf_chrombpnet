# workflows/molab/

Running this pipeline on a **molab** box (marimo cloud): one GPU, no SLURM, no
conda, no `/oak`. `workflows/SLURM/` stays the reference implementation — the
step scripts here are the *same files*, run through a different launcher.

```bash
cp workflows/molab/.env.example workflows/molab/.env   # once: bucket credentials
bash workflows/molab/setup_molab.sh      # once per box, and after every new session
source workflows/molab/env.sh            # once per shell
bash workflows/molab/run_step.sh 00.0.prepare_signal.sh
```

---

## Environments: pixi, and nothing else

**There is no conda and no container on a molab box.** Every step enters its
own environment through `activate_env` (`lib/bash/common.sh`), exactly as it
does on the cluster, and every default there is a pixi environment:

| environment | comes from | used by |
|---|---|---|
| **pixi `preprocess`** | this repo's `pixi.toml` | 00.0, 00.1, 02.0, and `cli.py download-references` |
| **chrombpnet 2.x**, pixi `cuda13` | the chrombpnet checkout `/marimo/chrombpnet-igvf` (`NNFC-GMD/chrombpnet` at `CHROMBPNET_REV`), from its own `pyproject.toml` and `pixi.lock` | 01.0, and 03.0 through 08.0 — TF-MoDISco (03.3, 04.5, 08.0) included |
| **pixi `motif-compendium`** | this repo's `pixi.toml` | 09.0 |
| **pixi `finemo`** | this repo's `pixi.toml` | 10.0, 11.0 |
| **pixi `qc`** | this repo's `pixi.toml` | no step; tests and plotting from a shell |

`bash workflows/molab/run_step.sh --list` shows the same mapping, read from each
step's own `activate_env` line.

The chrombpnet environment is installed from **chrombpnet's lock file**, the one
the Keras 3 / JAX port was validated with, not re-solved here. Its commit,
`CHROMBPNET_REV`, is pinned in `lib/bash/common.sh`; `activate_env` warns when the
checkout is anywhere else. The checkout is a separate clone on purpose: setup
moves its HEAD to the pin, so it must not be `/marimo/chrombpnet` or any other
checkout someone develops in (setup refuses to move one that is on a branch or
has local changes).

`env.sh` leaves `CHROMBPNET_ENV`, `PREPROCESS_ENV`, `FINEMO_ENV` and
`MOTIF_COMPENDIUM_ENV` unset, so `common.sh`'s pixi defaults are what runs. It only
says where the chrombpnet checkout is (`CHROMBPNET_REPO`, `CHROMBPNET_PIXI_ENV`).

## Setup

1. **Credentials.** `setup_molab.sh` needs `GCP_BUCKET` and `GCP_SA_JSON` (the
   service-account key as inline JSON) in a `.env` — see `.env.example`. There is
   no gcloud: `gcs.sh` mints the token with `openssl`.
2. **`bash workflows/molab/setup_molab.sh`**, as root. Idempotent: everything
   already present is checked and skipped, so re-run it after every new session.
   In order, it:
   - installs `git`, `curl`, `openssl` and CA certificates with apt when any is
     missing (running `apt-get update` first only then), and generates the
     `en_US.UTF-8` locale;
   - checks the bucket credentials, before anything slow;
   - installs pixi 0.81.0 into `/marimo/igvf-scratch/bin` when the box has no
     pixi or an older one (`env.sh` puts that directory first on `PATH`);
   - clones `https://github.com/NNFC-GMD/chrombpnet` into `$CHROMBPNET_REPO`, or
     fetches it, and runs `git checkout --detach $CHROMBPNET_REV`;
   - `pixi install --locked` of chrombpnet's `$CHROMBPNET_PIXI_ENV` (with
     `CONDA_OVERRIDE_CUDA=13.0`, so it installs even without a visible GPU), then of
     this repo's `preprocess`, `qc`, `finemo` and `motif-compendium`;
   - checks the chrombpnet environment: chrombpnet's `versions` task, then
     `bedtools`, `samtools`, `tomtom` and `modisco`;
   - checks the GPU: `nvidia-smi` (and the driver against the 580 that CUDA 13
     needs), then chrombpnet's `gpu-check` task;
   - downloads the d0 test data into `/marimo/data/test_data_d0/{config,inputs}/`
     and the shared references into `$REFERENCE_ROOT` (`--skip-references` to
     skip those);
   - prints the disk used by the environments and caches.

   Exit 0 means ready — a missing GPU is a loud warning, not a failure, because
   00.0–02.0 need none. Non-zero means something failed: an install, a
   download, or the tool check.
   Everything is also written to `/marimo/setup_molab.log` (override with
   `MOLAB_SETUP_LOG`), which is where to look if the session died mid-setup.
3. **`source workflows/molab/env.sh`** in every shell that runs a step.
4. **For a chrombpnet 2.x run, start a fresh results tree** — see the next section.

A box set up for chrombpnet 1.x still has `/marimo/containers/` (the unpacked
container). Nothing uses it any more; setup says so, and `rm -rf` frees the space.

## Fresh results for chrombpnet 2.x

Do not point a 2.x run at 1.x results. `run_step.sh` skips an array index when a
run-metadata record says it finished with its outputs intact, and before running
it restores from the bucket every result missing locally. So with the 1.x
`output_dir`, or the 1.x bucket prefix, the 1.x outputs count as done and the 2.x
steps never run. Change **both**:

- **`output_dir`** — in a *copy* of the dataset config, pointed at by
  `DATASET_CONFIG`. Not in the fetched `config/config.yaml` itself: setup
  re-fetches that from the bucket and replaces a local edit.
- **`MOLAB_GCS_PREFIX`** — where `sync_to_gcs.sh` writes results and
  `--restore` reads them (default `chrombpnet/test_data_d0`).

```bash
cp /marimo/data/test_data_d0/config/config.yaml /marimo/data/test_data_d0/config/config.cbp2.yaml
$EDITOR /marimo/data/test_data_d0/config/config.cbp2.yaml      # a new output_dir
# then, in the .env:
#   DATASET_CONFIG=/marimo/data/test_data_d0/config/config.cbp2.yaml
#   MOLAB_GCS_PREFIX=chrombpnet/test_data_d0_cbp2
```

## How the SLURM steps run without SLURM

Nothing is copied or translated: `run_step.sh` runs the very same file the
cluster submits, as `bash <step>` from `workflows/SLURM/`. That works because
**`#SBATCH` lines are bash comments** — SLURM reads them before executing, and
`bash step.sh` ignores them and runs the body. The step then activates its own
environment, so `run_step.sh` sets only what sbatch would have:

| variable | on the cluster | here |
|---|---|---|
| `SLURM_SUBMIT_DIR` | cwd at submit time | `workflows/SLURM` — the bootstrap uses it to find `REPO_ROOT` |
| `SLURM_ARRAY_TASK_ID` | one value per array task | looped over `--array`, in sequence |
| `SLURM_CPUS_PER_TASK` | from `--cpus-per-task` | `$MOLAB_CPUS` |

plus the thread caps below and `PYTHONUNBUFFERED=1`, so a step's log fills as it
runs. It also unsets the notebook's `PYTHONPATH`, `PYTHONHOME`, `PYTHONSAFEPATH`
and `VIRTUAL_ENV` (marimo's kernel exports its own venv to every terminal it
starts); `activate_env` does the same, and also drops `LD_LIBRARY_PATH`.

### What is silently lost

`--mem`, `--cpus-per-task`, `--gres`, `--partition`, `--time` and
`--constraint` are **ignored, and nothing enforces or reports them.** Several
steps request more memory than this box has (32 GB) — check a step's
`#SBATCH --mem` line. Those are scheduling requests, not measured usage, but
nothing here checks, so a step that genuinely exceeds 32 GB is OOM-killed by the
kernel. Per CLAUDE.md a SIGKILL writes no metadata record at all, so **the
missing record is the signal**: a step that dies leaving no JSON under
`${metadata_dir}` was almost certainly out of memory.

Likewise `--gres=gpu:1` grants nothing: the GPU is simply visible. Two GPU steps
at once would contend, which is why `run_step.sh` loops array indices in sequence
rather than in parallel. The `--constraint=GPU_CC:…` lines some steps carry are
inert off SLURM.

## The files

- **`setup_molab.sh`** — the setup above. Run once per box, and after every new
  session.
- **`env.sh`** — every variable in one place. Source it per shell. It loads the
  `.env`, sets `CHROMBPNET_REPO` (default `/marimo/chrombpnet-igvf`) and
  `CHROMBPNET_PIXI_ENV` (default `cuda13`), puts the pinned pixi on `PATH`, points
  every cache at `/marimo/igvf-scratch/cache` (below), sets `MOLAB_CPUS`,
  `REFERENCE_ROOT` and `DATASET_ROOT`, and defines `molab_config`, which resolves a
  path through `lib/bash/config.sh` the way the steps do.
  **`git push` needs it too.** The credential helper it installs expands
  `${GITHUB_TOKEN}` at push time rather than storing it, so a shell that has
  not sourced `env.sh` (or `.env`) pushes an empty password and GitHub replies
  `Invalid username or token` — which reads exactly like a bad token and is
  not one. `set -a; source workflows/molab/.env; set +a` is enough.
- **`run_step.sh`** — runs ONE step: emulates `--array`, writes a log per index
  to `$MOLAB_LOG_DIR` (default: the dataset's own `log_dir`). `--list` prints
  every step with its environment and `#SBATCH --array` default, `--dry-run`
  prints the exact command each index would run. **It never redoes finished work
  and never leaves results only on the box.** Before running, it restores from
  the bucket any result missing locally (`sync_to_gcs.sh --restore`, which never
  overwrites a local file). Each array index is then skipped if `step_done.py`
  finds a run-metadata record with `run_status: ok` whose declared outputs are
  all on disk with their recorded md5. After each index that succeeds, it
  uploads with `sync_to_gcs.sh`. A killed run leaves outputs but no record, so
  it is rerun, not trusted. `--force` reruns anyway: a record cannot tell that
  the config changed. It only skips this check, though: each step's own
  file-exists rules still apply, so finished work is not redone (04.0 takes
  `RETRAIN=1` to retrain over a finished model). `--no-bucket` skips the restore and the uploads.
- **`step_done.py`** — that check, stdlib only (it runs on the bare `python3`).
  `step_done.py <metadata_dir> <step> <index>` exits 0 and prints the record
  when the step is done, 1 with the reason otherwise.
- **`run_all.sh`** — steps 00.0 → 08.0 for one dataset, in order, written out
  as literal commands you can read or copy one line at a time. Stops at 03.1 for
  the manual bias-model review; `--after-bias` resumes. Leaves out 04.2 and
  09.0–11.0, which compare or pool datasets; run those by hand.
- **`sync_to_gcs.sh`** — copies `results/`, a `repo.bundle` of every branch,
  a `MANIFEST.txt` and the dataset config to
  `gs://${GCP_BUCKET}/${MOLAB_GCS_PREFIX}/` (default prefix
  `chrombpnet/test_data_d0`); `--restore` is the reverse, missing files only.
  `run_step.sh` runs it after every step. It is needed because a session that dies
  keeps only part of `/marimo`, and the commits on this box exist nowhere
  else. Files the bucket already holds byte-for-byte (size + md5) are
  skipped, uploads are md5-checked, and nothing remote is ever deleted.
  `--bundle-only` for just the commits, `--dry-run` to list. Needs a service
  account allowed to write under the prefix; no gcloud.
- **`gcs.sh`** — the bucket helpers `setup_molab.sh` and `sync_to_gcs.sh`
  share: an OAuth token minted from `GCP_SA_JSON` with `openssl` (scope
  `read_only` for setup, `read_write` for the sync), downloads, uploads,
  listings. Sourced, never run.

### Caches and disk

`/marimo` survives a new session and `$HOME` does not, so `env.sh` keeps the
pinned pixi and every cache under `MOLAB_SCRATCH` (default `/marimo/igvf-scratch`):

| variable | default |
|---|---|
| `PIXI_CACHE_DIR` | `$MOLAB_SCRATCH/cache/rattler` — shared by this repo's environments and chrombpnet's |
| `UV_CACHE_DIR` | `$MOLAB_SCRATCH/cache/uv` |
| `JAX_COMPILATION_CACHE_DIR` | `$MOLAB_SCRATCH/cache/jax` — compiled XLA programs, reused across runs |
| `KERAS_HOME`, `MPLCONFIGDIR`, `NUMBA_CACHE_DIR` | `$MOLAB_SCRATCH/cache/{keras,matplotlib,numba}` |

The box shows no disk quota (`df` reports 8.0E), so setup prints what the
environments and caches use. `pixi clean cache` reclaims the package cache; the
installed environments do not need it.

### Where are the step scripts?

Not here, on purpose. They live in **`workflows/SLURM/`** and are shared with
the cluster — that directory is the one definition of what each step *does*,
and molab only changes how it is *launched*. To see them:

```bash
bash workflows/molab/run_step.sh --list
```

```
STEP                                 ENV                ARRAY
00.0.prepare_signal.sh               preprocess         —
00.1.preprocess_peaks.sh             preprocess         —
01.0.preprocess_nonpeaks.sh          chrombpnet         —
02.0.qc_training_data.sh             preprocess         —
03.0.train_bias_model.sh             chrombpnet         0-19
...
```

The `ARRAY` column is the step's own `#SBATCH --array` default, which assumes
5 folds (× 4 bias factors for 03.0). `run_all.sh` derives the real ranges from
your config instead, through `lib/bash/config.sh`.

## The GPU

The box has an RTX PRO 6000 Blackwell (compute capability 12.0). chrombpnet 2.x
runs on it natively — its CHANGELOG: "runs natively on CUDA 13 GPUs (H100, B200,
RTX PRO 6000 Blackwell)" — with the CUDA libraries coming from JAX's pip wheels
inside the environment: no system CUDA, no `module load`, and none of the 1.x
container's PTX JIT on first use or the CUDA cache settings that softened it.
JAX still compiles each program the first time it runs; `JAX_COMPILATION_CACHE_DIR`
keeps those under `/marimo` for the next run.

- **Driver:** the `cuda13` environment needs NVIDIA driver >= 580. Setup checks
  it; on an older driver use `CHROMBPNET_PIXI_ENV=cuda12`.
- **Check it:** setup runs chrombpnet's `gpu-check` task. By hand:
  `pixi run --frozen --manifest-path /marimo/chrombpnet-igvf/pyproject.toml -e cuda13 gpu-check`.
  A molab session recreated after a crash can come back **without its GPU**, and
  JAX then quietly runs on the CPU; a GPU step should stop rather than crawl
  (`require_gpu` in `common.sh`, `--device gpu` on chrombpnet's training
  commands), but check first.
- **Memory:** JAX allocates on demand (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, set
  by `env.sh` and by chrombpnet itself). `XLA_PYTHON_CLIENT_MEM_FRACTION` is left
  unset, so a step may use the whole card. When the GPU is shared with other jobs,
  set it in the `.env` (e.g. `0.25`): it is a fraction of the card's **total**
  memory, and DeepSHAP sizes its batches from what is free.

## The box lies about its resources

`nproc` and `free` report the host (20+ cores, 160 GB), not the slice you get
(4 CPUs, 32 GB). Unpinned, OpenMP, BLAS and numba size their thread pools to the
phantom count and thrash. `env.sh` sets `MOLAB_CPUS=4` and `run_step.sh`
propagates it to `SLURM_CPUS_PER_TASK`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`NUMEXPR_MAX_THREADS` and `NUMBA_NUM_THREADS`. Numba is the one that gets missed:
TF-MoDISco sized its pool from the host's cores and ran 24 threads on 4 CPUs
until it was capped. **Change `MOLAB_CPUS` if your box differs.**

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

# 03.0's array is (bias-sweep folds x bias factors) - 1; run_all.sh prints it.
bash workflows/molab/run_step.sh --array 0-N 03.0.train_bias_model.sh
bash workflows/molab/run_step.sh 03.1.select_bias.sh
#   ... then copy the winners from selected_bias_per_fold.tsv into
#   fold_bias_suffix in the dataset config. This hand-off is deliberate: the
#   plots are meant to be reviewed.

# Per-fold steps take --array 0-(folds - 1); 04.3, 06.0-08.0 index the dataset (0).
bash workflows/molab/run_step.sh --array 0 03.2.qc_selected_bias.sh
bash workflows/molab/run_step.sh --array 0 03.3.modisco_selected_bias.sh
bash workflows/molab/run_step.sh --array 0 04.0.train_full_model.sh
bash workflows/molab/run_step.sh 04.1.qc_run_full_model.sh
bash workflows/molab/run_step.sh 04.3.generate_predictions.sh
bash workflows/molab/run_step.sh --array 0 04.4.qc_full_model_interpret.sh
bash workflows/molab/run_step.sh --array 0 04.5.modisco_full_model.sh
bash workflows/molab/run_step.sh --array 0 05.0.get_contrib_scores.sh
bash workflows/molab/run_step.sh 06.0.average_contrib_scores.sh
bash workflows/molab/run_step.sh 07.0.contribs_to_bigwig.sh
bash workflows/molab/run_step.sh 08.0.run_modisco.sh
```

## Adding a dataset

```bash
cp -r config/example_dataset config/my_dataset
$EDITOR config/my_dataset/config.yaml
pixi run --frozen -e preprocess python src/cli.py config validate --dataset my_dataset
export DATASET=my_dataset DATASET_CONFIG=$PWD/config/my_dataset/config.yaml
```

The d0 config setup fetches, `/marimo/data/test_data_d0/config/config.yaml`, is a
worked example for this box: absolute paths outside the checkout, and
`reference_root` pointing at the local reference tree.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `WARNING: /marimo/chrombpnet-igvf is at <sha>, not CHROMBPNET_REV=<sha>` | The checkout moved, or `common.sh`'s pin changed since setup. Re-run `setup_molab.sh`, which detaches it at the pin. Set `CHROMBPNET_REV` only to try another commit on purpose. |
| `ERROR: pixi is not on PATH (needed for pixi:…)` | `env.sh` not sourced in this shell (it puts `/marimo/igvf-scratch/bin` on `PATH`), or setup has not run on this box. |
| `ERROR: no pixi manifest at '/pyproject.toml'` | `CHROMBPNET_REPO` is empty: `env.sh` not sourced. |
| setup: `… is on branch '…', not at CHROMBPNET_REV` or `… has local changes` | `CHROMBPNET_REPO` points at a working copy. Use a dedicated checkout (the default `/marimo/chrombpnet-igvf`). |
| setup: `pixi install of … failed` on a lock-file check | The manifest changed without its `pixi.lock`. `--locked` never re-solves on the box: update the lock where the manifest was edited, and pull. |
| JAX lists only CPU devices, or fails to load CUDA | A system CUDA/cuDNN on `LD_LIBRARY_PATH` shadowing JAX's own wheels. Steps and setup unset it; a `python` started by hand in a shell that sets it is not protected — `unset LD_LIBRARY_PATH`. If `nvidia-smi` lists no GPU either, the session came back without one: restart it on a GPU machine. |
| A 2.x run skips steps it never ran | It is reading 1.x results: see **Fresh results for chrombpnet 2.x**. |
| A step died and left no run-metadata JSON | SIGKILL, almost always out of memory; see **What is silently lost**. |
| `FileExistsError: ..._auxiliary/` in 01.0 | A killed `chrombpnet prep nonpeaks`; 01.0 clears a stale one, so re-run the step. |
| `found pyproject.toml without tool.pixi section at directory /marimo` | pixi run from `/marimo`, which holds marimo's own `pyproject.toml`. Run pixi from the checkout, or pass `--manifest-path`. |
