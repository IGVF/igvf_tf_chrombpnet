# igvf_tf_chrombpnet

ChromBPNet pipeline for the IGVF TF collaboration: trains bias-factorised models on
ATAC-seq pseudobulks, averages contribution scores across 5 folds, and discovers
motifs (TF-MoDISco → MotifCompendium → Fi-NeMo).

**This is a SLURM pipeline, not a package.** There is no `pyproject.toml` of its
own, no installable module and no CI. There *is* a test suite (`tests/`, `pixi run -e qc
test`) covering the pure-Python helpers in `lib/python/utils` and the selection
logic in `src/`; it never touches the cluster. It is a numbered sequence of `sbatch`
scripts under `workflows/SLURM/` plus the Python they call, all of which assume
Stanford Sherlock: `/oak/stanford/groups/engreitz/...` data and references, `pixi`
on `PATH`, and GPU nodes whose NVIDIA driver can run CUDA 13 (>= 580). Every
environment a step runs in is a pixi environment (see *Environments*); no step
loads an `ml` module. On a laptop, `pixi` runs the lint, syntax checks, tests and
QC plotting; the steps themselves need the cluster, or the molab box
(`workflows/molab/`).

`README.md` is the user-facing reference (dataset table, setup, quick start). This
file is the contributor-facing complement: don't duplicate the README, point at it.
`workflows/README.md` is a `[work in progress]` stub with a stage overview and the
rules for adding a step.

## Layout
Four top-level roles: `workflows/` orchestrates, `src/` does one job per file,
`lib/` is shared code, and `config/<dataset>/` is configuration.

The repo is checked out **as the collaboration root** by default:
`DATASET_ROOT` defaults to `${REPO_ROOT}`, so `<dataset>/data` and
`<dataset>/results` sit beside `workflows/`. `09.0.cross_dataset_compendium.sh`
reads each dataset's MoDISco H5 under `${data_root}` (= `DATASET_ROOT`) and writes
the compendium to `${REPO_ROOT}/results/compendium/`, which is where
`10.0.run_finemo_unified.sh` reads it; 10.0's per-dataset inputs come through
`config.sh` like any other step's. Only `igvf3_cardiomyocyte/`'s `.gitkeep`s are
tracked (its config is `config/igvf3_cardiomyocyte/config.yaml`); the other three
dataset directories exist on the cluster only.

- `workflows/SLURM/` — one `sbatch` script per step. See the step table below.
- `workflows/molab/` — a launcher that runs the same step files on a molab box (one
  GPU, no SLURM). Its README covers setup; it is not a second implementation.
- `workflows/nextflow/` — empty placeholder for the Nextflow port. `SLURM/` stays the
  reference implementation until it isn't.
- `src/` — the atomic Python scripts the steps call, one concern each. Nothing in
  `src/` imports anything from another file in `src/`; shared code goes to `lib/`.
- `lib/bash/common.sh` — settings and helpers with no `DATASET_DIR`: the environment
  specs (`chrombpnet_env`, `preprocess_env`, `finemo_env`, `motif_compendium_env`)
  and the chrombpnet pin, algorithm thresholds, and the helpers — `bootstrap_python`,
  `load_bias_sweep`, `activate_env` (with its pixi branch `activate_pixi_env`),
  `gpu_env`, `require_gpu`, `metadata_start`/`metadata_emit`, `set_signal_args`,
  `require_input`/`preflight_check`. Its own comment says only the ones that are
  used exist. Steps still write progress with inline `echo "[$(date)] ..."` and check
  inputs with `[[ -f ... ]] || { echo ...; exit 1; }`; adding a
  `log()`/`require_file()` means converting those ~40 call sites in the same
  change, not adding a second way.
- `lib/bash/references.sh` — evals `lib/python/utils/references.py export`, so the
  reference paths the steps read (`genome_fa`, `ref_db_meme`, …) are the ones
  `cli.py download-references` writes.
- `lib/bash/config.sh` — the per-dataset layer. Sources `common.sh`, then renders
  `config/<dataset>/config.yaml` (see *Configuration*), then derives every output
  path.
- `lib/python/utils/` — the importable helper package: `intervals` (pyranges1 ops
  replacing bedtools), `references` (reference paths, pins and fetching),
  `compression` (bgzip/tabix), `config` (the YAML loader), `folds` (reads
  `folds/*.json`), `log`, `metadata` (run records), `palettes`, `pileup`
  (fragments -> cut-site bigwig), `plotting`, `qc`, `regions`.
  Two import rules, both load-bearing: nothing here may import `chrombpnet`,
  `keras`/`jax`, `torch`, `finemo` or `MotifCompendium`, and **only `intervals` may
  import `pyranges1`** (only the `preprocess` and `qc` envs install it, while
  `log`, `metadata` and friends are imported from every environment, the
  chrombpnet, finemo and motif-compendium ones included). The package is called
  `utils` and sits at the front of `sys.path`, so it would shadow any third-party
  top-level `utils` — check before adding a dependency that ships one.
- `folds/fold_{0..4}.json` — the 5-fold chromosome splits (`train`/`valid`/`test`),
  shipped with the repo and passed to chrombpnet as `-fl`.
- `pixi.toml` / `pixi.lock` — every environment except chrombpnet's (see
  *Environments*).
- `cli.py download-references` — one-time, idempotent fetch of genome /
  chrom.sizes / blacklist / both motif DBs into `${REFERENCE_ROOT}` (the shared lab
  `Data/` folder by default), md5-checked.
- `lib/python/utils/shift.py` — vendored from scPrinter (Ruochi Zhang). `00.0`
  (`cli.py prepare-bigwig`) uses it to detect the Tn5 shift when the config sets
  none, and to compute the delta to chrombpnet's +4/-4 (ATAC) or 0/+1 (DNASE).

### Configuration
**One YAML file per dataset — `config/<dataset>/config.yaml` — is the whole
configuration.** There is no second "site"/"env" file: the machine-specific
values (`CHROMBPNET_REPO`, the `*_ENV` overrides, `REFERENCE_ROOT`,
`DATASET_ROOT`) are environment variables with defaults in `common.sh`, because
they are identical across datasets and must not be committed. `config/README.md`
has the table.

Selected by `DATASET=<name>`, or `DATASET_CONFIG=/path/to/config.yaml`.
`dataset_dir` defaults to `${data_root}/<dataset>`, so the tracked config stays
small while the data lives wherever you like; `config.sh` re-exports
`DATASET_DIR` afterwards for the steps that reference it.

**The YAML reaches bash through `lib/python/utils/config.py`**, which renders it
to shell assignments that `config.sh` evals (`${...}` left intact so the shell
still interpolates). That loader **must run before any environment is active**,
on whatever python >= 3.9 `bootstrap_python` finds, so it imports nothing
third-party and nothing else from `utils` — don't add imports to it. It uses
PyYAML when importable and a strict built-in subset parser otherwise; the
subset covers scalars, flow/block lists and
one nesting level and **raises on anything else rather than guessing**.
`tests/test_config.py` asserts the two parsers agree on every config in the repo,
which is what stops the fallback drifting from real YAML. Validate a config with
`cli.py config validate --dataset <name>`.

`#SBATCH` directives stay literals — SLURM parses them before any of our code
runs, so they cannot read a variable. Override at submit time.

### Preconditions
Steps declare inputs with the step that produces them and stop before doing work:

```bash
require_input "${peaks_file}"     00.1.preprocess_peaks.sh
require_input "${negatives_file}" 01.0.preprocess_nonpeaks.sh
preflight_check
```

`preflight_check` reports **every** missing input at once with a copy-pasteable
`sbatch` line per producer — being told about one missing file per GPU submission
is the failure mode it exists to prevent. It must run **before** `activate_env` /
`gpu_env` / `require_gpu` so the check is cheap; every step that has a preflight
is ordered that way deliberately, so don't move the environment setup back above
it. `activate_env` itself fails with a clear message — and, for the chrombpnet
checkout, the commands that create it — when `pixi`, the manifest or the conda
init script is missing.
`workflows/SLURM/status.sh` runs the same checks across all steps.

### How a step starts
Every step opens with a **byte-identical bootstrap block** that sets `REPO_ROOT`, then
one source line. Copy it from an existing step; don't hand-write a variant.

```bash
# --- bootstrap: locate the repo root ... --- (14 lines, identical everywhere)
source "${REPO_ROOT}/lib/bash/config.sh" || exit 1   # per-dataset steps
source "${REPO_ROOT}/lib/bash/common.sh" || exit 1   # cross-dataset steps
```

The bootstrap tries `REPO_ROOT`, then walks up from `SLURM_SUBMIT_DIR`, then from
`BASH_SOURCE`, looking for `lib/bash/common.sh`. **`sbatch` copies the script to a
node-local spool directory**, so `BASH_SOURCE` is useless under SLURM and
`SLURM_SUBMIT_DIR` (the cwd at submit time) is what actually works — which is why
submitting happens from `workflows/SLURM/`, or with `REPO_ROOT` exported.

Dataset-specific things (signal path, reference overrides, `bias_factors`,
`fold_bias_suffix`) live in `config/<dataset>/config.yaml`; shared things live in
`common.sh`; derived output paths live in `config.sh`.

### Steps
`--array` index means something different per step — check before overriding it.

| Script | Array index | Per-dataset (needs `DATASET` / `DATASET_CONFIG`) |
|---|---|---|
| `00.0.prepare_signal.sh` | — | yes |
| `00.1.preprocess_peaks.sh` | — | yes |
| `01.0.preprocess_nonpeaks.sh` | — | yes |
| `02.0.qc_training_data.sh` | — | yes |
| `03.0.train_bias_model.sh` | `fold_idx * n_factors + factor_idx` | yes |
| `03.1.select_bias.sh` | no SBATCH header — run with `bash` | yes |
| `03.2.qc_selected_bias.sh` | fold | yes |
| `03.3.modisco_selected_bias.sh` | fold | yes |
| `04.0.train_full_model.sh` | fold | yes |
| `04.1.qc_run_full_model.sh` | — | yes |
| `04.2.qc_combined_boxplot.sh` | — | no (discovers `config/*/config.yaml` plus `DATASET_CONFIG`) |
| `04.3.generate_predictions.sh` | dataset | yes |
| `04.4.qc_full_model_interpret.sh` | fold | yes |
| `04.5.modisco_full_model.sh` | fold | yes |
| `05.0.get_contrib_scores.sh` | fold | yes |
| `06.0.average_contrib_scores.sh` | dataset | yes |
| `07.0.contribs_to_bigwig.sh` | dataset | yes |
| `08.0.run_modisco.sh` | dataset | yes |
| `09.0.cross_dataset_compendium.sh` | — | no (hardcoded `h5_map` under `${data_root}`) |
| `deprecated/motif_compendium.sh` | — | yes |
| `10.0.run_finemo_unified.sh` | dataset | yes |
| `11.0.postprocess_finemo.sh` | dataset | yes |
| `qc_datasets.sh` | — | no (sources `common.sh`; paths from `core_path` in `src/qc_datasets.py`) |

All step scripts live in `workflows/SLURM/`. The Python they call lives in `src/`
and is referenced as `${src_dir}/<name>.py`, never by a relative path:
`cli.py` (00.0 `prepare-bigwig`, 00.1 `preprocess-peaks`, 02.0 `qc-signal`, and
`download-references`), `predict_bias_metrics.py` (03.0),
`select_bias_model.py` (03.1), `run_bias_qc.py` (03.2), `motif_qc.py` (03.3, 04.5), `chrombpnet_train.py`
(03.0, 04.0), `qc_full_model.py` (04.1 and 04.2-combined), `predict_and_avg.py` (04.3),
`run_full_model_qc.py` (04.4), `average_contrib_scores.py` (06),
`contribs_to_bigwig.py` (07), `motif_compendium.py` (09 and
`deprecated/motif_compendium.sh`), `qc_datasets.py` (`qc_datasets.sh`), and
`emit_metadata.py` (every bash step's metadata trap). 01.0, 05.0 and 08.0 call
chrombpnet's and modisco's own CLIs, and 10.0/11.0 `finemo`, directly.

## Commands
Everything local goes through `pixi` (0.76+). It covers lint and syntax checks, plus a
`qc` environment that can run the tests and the plotting scripts on results copied
off the cluster.

```bash
pixi install
pixi run check           # check-bash + check-py + lint
pixi run check-bash      # bash -n, then shellcheck -S warning, over workflows/SLURM/*.sh,
                         # workflows/molab/*.sh and lib/bash/*.sh (not deprecated/)
pixi run check-py        # byte-compile lib/python and src
pixi run lint            # ruff check
pixi run fmt             # ruff format (fmt-check for a dry run)
pixi run hooks-install   # enable the pre-commit hook
pixi run hooks           # pre-commit run --all-files
pixi run -e qc test      # pytest (needs the qc env: pyranges1, Python >= 3.12)

pixi run -e qc python src/qc_full_model.py --help   # QC scripts import without the cluster
```

**Commit from inside `pixi shell`, or the hook fails.** The `ruff` and `shellcheck`
hooks are `language: system` so that the hook and `pixi run check` are the same
binaries reading the same config — which means they need `ruff`/`shellcheck` on
`PATH`. A `git commit` from a bare terminal fails with `ruff: command not found`;
that is the hook working as configured, not a broken hook.

On the cluster, every step enters a pixi environment through `activate_env`, so
`pixi` must be on `PATH` wherever a step runs. Once per cluster, from a login
node (the same commands are in `lib/bash/common.sh`, under *Software
environments*, and in `activate_env`'s error when the checkout is missing):

```bash
export CHROMBPNET_REPO=/path/to/chrombpnet    # a checkout used only by this pipeline
git clone https://github.com/NNFC-GMD/chrombpnet "$CHROMBPNET_REPO"
(cd "$CHROMBPNET_REPO" && git checkout --detach "$CHROMBPNET_REV")   # the SHA in common.sh
CONDA_OVERRIDE_CUDA=13.0 pixi install --locked \
    --manifest-path "$CHROMBPNET_REPO/pyproject.toml" -e cuda13

pixi install --locked -e preprocess          # and finemo, motif-compendium; from this checkout
pixi run -e preprocess python src/cli.py download-references

export DATASET=igvf3_cardiomyocyte
cd workflows/SLURM && sbatch 00.1.preprocess_peaks.sh
```

`CHROMBPNET_REPO` has no default and must be exported for every job (put it in
your profile; `sbatch` passes the environment on). `CONDA_OVERRIDE_CUDA` is what
lets a login node without a GPU install a CUDA environment; `activate_env` sets
it too (default `13.0`), so CPU-only steps can enter the chrombpnet environment.

### Environments
All pixi, and they are not interchangeable. A step names its environment in
exactly one place, its `activate_env` line, as `pixi:<manifest>#<environment>`:

| Environment | Steps | Comes from |
|---|---|---|
| chrombpnet 2.x, `cuda13` (`chrombpnet_env`) | 01.0, 03.0–04.5, 05.0–08.0 — training, interpretation, predictions, TF-MoDISco (modisco 2.5.2), the QC plots of 03.1/04.1/04.2 | the chrombpnet checkout's `pyproject.toml` + `pixi.lock`, at `CHROMBPNET_REV` |
| `preprocess` (`preprocess_env`) | 00.0, 00.1, 02.0, `qc_datasets.sh`, `cli.py download-references` | `pixi.toml` |
| `finemo` (`finemo_env`) | 10.0, 11.0 — Fi-NeMo 0.41 on torch 2.14 (CUDA 13.0 wheel) | `pixi.toml` |
| `finemo-cu126` | none by default; for 10.0 on drivers below 580 or GPU_CC 7.0 (`FINEMO_ENV`, see 10.0's header) | `pixi.toml` |
| `motif-compendium` (`motif_compendium_env`) | 09.0, `deprecated/motif_compendium.sh` — MotifCompendium v1.0.19, CPU only | `pixi.toml` |
| `default` / `qc` | no step: lint, tests, local plotting | `pixi.toml` |

**chrombpnet stays out of this repo's `pixi.toml` on purpose.** It is installed
from a pinned checkout of the NNFC-GMD fork (branch `pipeline-hooks`: PR
kundajelab/chrombpnet#284 plus `-bw` on the training commands, `pipeline
--skip-interpretation` and the lookup-table one-hot encoder), from that repo's own
lock file — the one the Keras 3 / JAX port was validated against — rather than
re-solved here. `activate_env` warns when the checkout is not at `CHROMBPNET_REV`;
it does not stop, so a newer chrombpnet can be tried on purpose, and the run
record carries `chrombpnet_commit`. `CHROMBPNET_PIXI_ENV=cuda12` is the fallback
for drivers below 580. The environments `pixi.toml` does define each have their own
solve group, so Fi-NeMo's torch and MotifCompendium's stack cannot move each
other's pins. Any `*_ENV` may still be a plain conda prefix, which `activate_env`
enters through `CONDA_INIT`.

**No step loads a CUDA module, and none should.** JAX and torch bring CUDA as pip
wheels inside their environments; a system CUDA on `LD_LIBRARY_PATH` shadows them
and breaks their start-up, which is why `activate_pixi_env` unsets
`LD_LIBRARY_PATH` and `PYTHONPATH` before `pixi shell-hook --frozen` (`--frozen`:
install from the lock if missing, never re-solve, so a job cannot rewrite a lock
file). What gates a GPU node is now its NVIDIA driver, not a module.

**What the tests do and do not cover.** `tests/` asserts against the pure-Python
helpers — intervals, pileup, config parsing, metadata, palettes, QC metrics,
bias selection — several of them differentially, against bedtools or chrombpnet's
own output. What they cannot touch is anything needing a GPU, chrombpnet, or
cluster data: every `sbatch` step is verified only by `bash -n`, `shellcheck` and
reading the tool's argument parser. **Don't claim a step ran when it didn't** —
say which of the two kinds of verification a change actually got.

## Conventions
- **Indentation: 4 spaces**, bash and Python alike; no tabs anywhere. Python style is
  enforced by `ruff` — config in `.ruff.toml` (not `pyproject.toml`; there is no
  package to hang it off), read by the CLI, the pixi tasks and the pre-commit hook
  alike, so they cannot disagree. `E501` is off (the formatter owns line length) and
  `B905` is off with a note; see the file.
- **Commit messages: sentence-case imperative**, no prefix — `Add …`, `Fix …`,
  `Move …`, `Renumber …`, `Split …`. Bodies explain the *why* at length (see
  `a249519`, `b6baa4b`). Don't switch to a tag/prefix style.
- **Docs accuracy is a hard rule**: every concrete detail (paths, defaults, array
  ranges, step numbers, partition names) must be confirmable from the source in this
  repo. If you can't verify it, omit it. This file is the one place the step numbering
  is correct — keep it that way.
- **`src/` scripts import `lib/python` through a three-line `sys.path` shim**, not
  through `PYTHONPATH` and not through an install. It works identically under every
  pixi environment (chrombpnet's included), a conda prefix and a bare `python`,
  which is the point — and `activate_env`'s pixi branch unsets `PYTHONPATH` anyway. The
  shim sits between the stdlib imports and the third-party ones, so `E402` is ignored
  for `src/*.py`. Copy it verbatim into any new script.
- **Shared code goes in `lib/`, one job per file in `src/`.** If two `src/` scripts
  need the same thing, it moves to `lib/python/utils/` — they must not import each
  other. Same rule for bash: repeated setup belongs in `lib/bash/common.sh`.
- **Every script header is a mini-doc**: `# NN.name.sh`, `# Purpose:`, why-this-way
  prose, `# Input:` / `# Output:`, `# Usage:`, `# Prerequisites:`. New steps follow
  the same shape, and an edit that changes behaviour updates the header in the same
  commit.
- **Never commit data or results.** `.gitignore` covers `*.tsv`, `*.gz`, `*.h5`,
  `*.bw`, `*.bed`, `results/`, `*/results/`, `*.ipynb` and `.claude/`. A dataset
  directory holds only `.gitkeep`s; its config lives in `config/<dataset>/`. Note
  `*.tsv` is ignored, so pipeline output tables cannot be committed even
  deliberately.
- **Python QC scripts are notebook-shaped**: `# %%` cell markers, `plotting.apply_style()`
  (which sets the `Agg` backend and the rcParams), and `plotting.save_fig` writing
  `.pdf` + `.png`.
- **Plots must be colourblind-safe, and this is enforced.** Categorical series use
  **Okabe-Ito**; continuous scales use **cividis** (`palettes.SEQUENTIAL_CMAP`).
  `tests/test_palettes.py` fails the build if any plotting source hardcodes a hex
  literal or names `viridis`/`jet`/`rainbow`/`magma`/`inferno`/`plasma` as a `cmap`,
  so **all colour lives in `utils/palettes.py`** — add a named entry there rather
  than a literal at the call site. The pass/warn/fail heatmap also labels each cell
  with its metric value; that text is the fallback for readers colour cannot serve,
  so keep it.
- **Every step emits a run-metadata JSON.** One record per invocation under
  `${metadata_dir}` (`<dataset>/results/metadata/<ts>_<step>_<runid>.json`, a flat directory;
  cross-dataset steps use `${REPO_ROOT}/results/metadata`). Schema version 2.
  Python steps use `with metadata.record(...)`; bash steps call
  `metadata_start "<step>"` and append to `metadata_inputs`/`metadata_outputs`/
  `metadata_params`/`metadata_metrics`/`metadata_tools`, and an EXIT trap emits via
  `src/emit_metadata.py`. **Both paths produce the same schema** — they share
  `StepMetadata`, so don't add fields to one without the other.
  - **Everything a run consumed, produced, was configured with, or measured is
    ONE long-format list, `parameters`.** `parameter_type` tells the four kinds
    apart: `input` and `output` are files, `param` is a setting that controlled
    the run, `metric` is a quantity it measured. One `UNNEST` answers every
    question about a run; `queries.sql` exposes `run_parameters`, `run_files`,
    `run_settings` and `run_metrics` over it.
  - **Field names are ENCODE/IGVF-flavoured and unambiguous on their own**,
    because UNNEST flattens these structs into one table where a bare `key`,
    `value`, `name` or `path` says nothing: `parameter_name`/`parameter_value`,
    `software_name`/`software_version`, `file`/`filepath`/`file_size`/`md5sum`,
    `uuid`, `date_created`/`date_completed`. Run outcome is `run_status`, NOT
    `status` — on the portal `status` means an object's lifecycle (released /
    in progress / archived), not whether a process exited 0. No `@id`, `@type`
    or `accession` is emitted: those are portal-assigned, and inventing them
    would make a local record look like a registered IGVF object.
  - **`parameter_name` says WHAT the data is; `file_format` says how it is
    encoded.** Neither borrows the other's vocabulary. `file_format` is
    DERIVED from the extension by `metadata.file_format()` and never
    hand-written — hand-written formats drift into the semantic name, which is
    how the old vocabulary ended up with roles like `qc_tsv`, `counts_h5` and
    `prepared_bigwig` that answered both questions in one string.
  - Values are strings *on purpose*: a plain object would give every step a
    different STRUCT and DuckDB's union would go ragged. `command` keeps the
    full argv.
  - Outputs are declared early but hashed at exit, so a failed run still records
    what it meant to produce with `file_exists: false`.
  - md5 is on by default, streamed in 8 MiB chunks; `METADATA_CHECKSUMS=0`
    turns it off and the record says `md5_skipped: "disabled"` rather than going
    silently null.
  - A SIGKILL (OOM, hard preemption) writes nothing. The missing record is the
    signal; the trap only survives SIGTERM.
  - Emission never fails a step, and `script_url` is misleading when `git.dirty`
    is true — filter on it. Query recipes are in `queries.sql`.
  - A version string cannot say which chrombpnet ran: 2.x reports `2.0.0.dev0` at
    every commit, and MotifCompendium comes from a git pin. So the tool list also
    carries `<name>_commit` for any tracked distribution whose PEP 610 record
    names a git commit or a local checkout (asked with git, `-dirty` appended
    when tracked files differ). Query `chrombpnet_commit`, not the version.
- **00.1 can drop peaks by their own signal** (`peak_min_signal_quantile`, off
  by default). The threshold is a quantile of the experiment's own genome-wide
  windows, so it is depth-independent; blacklist regions are excluded from the
  background sample but peaks are not, because it is the experiment's
  distribution rather than a background model. It matters out of proportion to
  the peaks it removes: chrombpnet anchors every bias threshold in the 03.0
  sweep to `quantile(peak_counts, 0.01)`, and on d0 the weakest 1% of peaks sat
  at the 40.6th percentile of genome windows, so dropping 1.81% of them moved
  that anchor from 4 to 17. Needs 00.0's bigwig. Outputs go to
  `preprocessing/peaks/` with a `peaks.json` sidecar recording what each filter
  stage cost, and a plot in `plots/peaks_qc/`.
- **QC runs on the artifacts, not the inputs.** `02.0` reads the prepared
  bigwig, the filtered narrowPeak and the negatives, so every number describes
  what ChromBPNet will actually see after all filtering. It is advisory and
  never fails the pipeline. Two halves:
  - *individual* — `tss_enrichment` (near 1 means the signal is not
    accessibility, or does not match the annotation) and
    `frac_peaks_zero_signal` (peaks with no evidence under them, usually
    meaning peaks and signal came from different samples).
  - *comparative* — `auroc_peaks_vs_nonpeaks`, how well signal alone separates
    peaks from their own GC-matched background. That is the task training
    poses, so a value near 0.5 means no GPU time is worth spending. Both sides
    are summed over one FIXED window (`qc_compare_window`, default 1000 =
    ChromBPNet's output window) via `qc.window_totals`, because the negatives
    are all 2114bp and summing peaks over their own called widths would decide
    the comparison on peak-caller settings rather than on signal.
- **QC comes after the negatives (02 after 01), not before.** The comparative
  half needs the negatives to exist, so the order is
  `00.0 signal → 00.1 peaks → 01.0 non-peaks → 02.0 QC → 03/04 GPU`. QC still
  runs before anything expensive.
- **`01.0` deliberately calls `chrombpnet prep nonpeaks` rather than a port.**
  The negatives *are* training data — half of what every model in 03 and 04
  sees — and they come out of a seeded Python RNG, so a reimplementation that
  differed in the order it consumed `random` would silently change every model
  and surface months later as unexplained metric drift. Contrast `00.0`, whose
  bigwig is verifiable byte-for-byte against ChromBPNet's output, which is why
  a faster path there is free. Consequences to know:
  - ChromBPNet does `os.makedirs(prefix + "_auxiliary/", exist_ok=False)`, so a
    killed job leaves that directory and every retry dies on `FileExistsError`
    before doing any work. `01.0` clears a stale one before each fold.
  - The genome-wide GC scan (~3M windows, pure Python) is redone per fold,
    because the auxiliary directory is per-prefix. That is the cost of running
    their code unchanged; it is CPU-only and paid once per dataset.
  - It needs the FULL `chrom_sizes`, not `chrom_sizes_main`: the blacklist
    carries non-main contigs and `bedtools slop` errors on a contig it cannot
    find. The fold JSON already confines sampling to main chromosomes.
  - It shells out to bedtools, so it activates `${chrombpnet_env}`, not
    `${preprocess_env}`.
- **Not SnapATAC2, deliberately.** `snapatac2.metrics.tsse` needs an AnnData from
  `import_fragments` and reports per-cell scores. This pipeline trains on
  pseudobulk, so a per-cell distribution does not answer "is this worth
  training on", and building a cell x bin matrix to reach the library aggregate
  is a lot of machinery for one number. SnapATAC2 is the right tool for per-cell
  QC *upstream* of pseudobulking — a different question.
- **Use `logging`, never `print`.** `utils/log.py` configures it: scripts call
  `log.setup_from_args(args)` in `main()` and use a module-level
  `logger = log.get_logger(__name__)`; library modules in `utils/` only ever call
  `logging.getLogger(__name__)` and never add a handler. Logs go to **stderr** with
  the same `[%F %T]` prefix the bash steps use, so a SLURM `.log` reads consistently.
  **stdout is left free for data** — nothing pipes a script's stdout today, and the
  old print-everything convention is what made that impossible. Every script takes
  `-v/--verbose` and `-q/--quiet`, and `LOG_LEVEL` sets the level for a whole
  job through `sbatch --export=ALL`.

## Gotchas

- **Step files are `NN.M.name.sh`, uniformly, so `ls` order is execution order.**
  That uniformity is the point: when some steps were `NN.` and others `NN.M.`,
  a `00.1.` step sorted *above* the `00.` one, and the listing misrepresented
  the order things run in. Keep the `.M` even when a stage
  has only one step. `workflows/SLURM/deprecated/` is outside the sequence.


- **`deprecated/motif_compendium.sh` is superseded, deliberately.** `a249519` renamed it with
  the underscore to mark it as "not part of the main numbered sequence": it builds a
  *per-dataset* compendium into `${modisco_compiled_dir}`, while
  `09.0.cross_dataset_compendium.sh` pools all four datasets, and `10.0.run_finemo_unified.sh`
  explicitly reads `09`'s output. Don't renumber it back in.

- **chrombpnet 2.x results need a fresh results tree.** Every skip rule looks at
  files, not at which chrombpnet wrote them — 03.0's model + metrics, 04.0's model +
  footprints, 05.0's scores, 06/07's helpers, 08.0 per head — and
  `load_model_wrapper` reads 1.x model files happily, so a 2.x run pointed at a 1.x
  `output_dir` silently reuses 1.x models and everything derived from them. On molab
  it is worse: `run_step.sh` first restores every missing result from the bucket and
  then skips any index `step_done.py` calls finished, and `--force` bypasses only
  `step_done.py`, not the steps' own checks, so it does not retrain. Give a 2.x run
  its own `output_dir` (on molab, in a copy of the config, plus its own
  `MOLAB_GCS_PREFIX`; `workflows/molab/README.md` has the recipe).

- **`${ref_db_meme}` is a versioned file now:
  `MotifCompendium-1.0.19-Database-Human.meme.txt`.** The unversioned URL pointed at
  MotifCompendium's `main`, whose database changed at v1.0.17, so two installs of
  the same pipeline could annotate 08.0's report and 09.0's compendium against
  different motif sets. `references.py` now pins the v1.0.19 commit the
  `motif-compendium` environment installs and checks the md5. An install fetched
  before the pin keeps its unversioned file, neither overwritten nor mistaken for
  the pinned one — which means **every existing `REFERENCE_ROOT` needs one more
  `cli.py download-references`** before 08.0/09.0. Both preflight the file and
  name that command.

- **Untracked `dataset_config.sh` copies on the cluster still set `folds_dir`.**
  No tracked config does — `lib/bash/common.sh` sets it to `${REPO_ROOT}/folds`.
  The three cluster-only datasets (igvf6, igvf11, igvf_endothelial) carried
  `folds_dir="${SCRIPT_DIR}/../folds"`, which stopped resolving when the steps
  moved. `config.sh` no longer reads a `dataset_config.sh` at all, only a
  `config.yaml`, so those datasets need one (none is tracked), and it must not
  carry that line over. `config.sh` still checks the directory exists and fails at
  once rather than several minutes into a GPU job (its message still names
  `dataset_config.sh`).

- **Absolute `opushkar` paths remain in two files**: `core_path` in
  `src/qc_datasets.py`, and `queries.sql` — the `runs` view's glob (point it at
  your own metadata) and the user in one example query. The conda env defaults in
  `lib/bash/common.sh` that used to point there are gone; every default is now a
  pixi environment.

- **The endothelial dataset is named two ways and laid out differently.**
  `qc_datasets.py` calls it `igvf17_endothelial`; `README.md` and `09` call it
  `igvf_endothelial`, and `utils/palettes.py` carries both keys. In `09`'s `h5_map`
  its MoDISco H5 sits at `${data_root}/igvf_endothelial/results/contrib_scores/modisco/...`
  while the other three use `${data_root}/<dataset>/results/contrib_scores/<dataset>/modisco/...`.
  Both are load-bearing paths; changing either without checking the cluster layout
  will silently drop the dataset (the loop only `[WARN]`s on a missing H5).

- **`src/qc_full_model.py`'s leftovers from another pipeline are gone, on
  purpose.** `--datasets` has no default: the names are real directory names and
  a wrong guess silently produces an empty plot, so they come from the config via
  the step. Its hardcoded `TEST_CHROMS` is gone too — it calls
  `utils.folds.test_chroms()`, which reads the same `folds/fold_<n>.json`
  chrombpnet trained against; the two were verified identical for all five folds
  before the swap. `d0` (the molab test dataset) still appears in the usage
  examples of `predict_and_avg.py`, `run_full_model_qc.py` and `motif_qc.py`.

- **Steps 06/07/08 run both heads; steps 09/10 consume `counts` only.**
  `score_types=("counts" "profile")` in 06, 07 and 08. Counts used to be left out
  because the 1.x results tree already had it, but a 2.x results tree starts
  empty, and counts is the head downstream needs: `09.0.cross_dataset_compendium.sh`
  hardcodes `modisco_counts_results.h5` and `10.0.run_finemo_unified.sh` hardcodes
  `_average_shaps.counts.h5`. 06/07's helpers and 08.0 skip a head whose output
  exists, so a rerun only fills in what is missing. Dropping `"counts"` from
  `score_types` again — assuming the downstream steps follow it — is not safe.

- **The GPU compute-capability constraint is not uniform.** `03.0`, `03.2`, `04.0`,
  `04.3` and `04.4` pin `--constraint="GPU_CC:8.0|GPU_CC:8.6"`: the only classes the
  chrombpnet 2.x wheels have been used on, not a known limit. What gates a node is
  its NVIDIA driver (the `cuda13` environment needs >= 580, `cuda12` >= 525); the
  old upper bound came from the `cuda/11.5` module, which no step loads any more.
  `03.0`/`04.0` leave out 7.0/7.5 for speed (the 1.x timings are in 03.0's header).
  `10.0` pins `GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0`, following the
  torch 2.14 CUDA 13.0 wheel (sm_75 and up); its header gives the `finemo-cu126`
  fallback and its constraint. `05.0` has no constraint at all and relies on
  `require_gpu jax`, which stops a job whose JAX sees no GPU but does not check the
  class. If a GPU step fails oddly on `owners`, check the driver first —
  `require_gpu`'s error says so.

- **`03.2` is the GPU half and `03.3` the CPU half of the selected-bias QC.**
  `pipelines.bias_model_qc()` runs predictions, DeepLIFT interpretation, then
  TF-MoDISco and its reports in one call. Only the first two need a GPU;
  TF-MoDISco is CPU-only and is the long pole, so running them together left a
  GPU idle for hours. `src/run_bias_qc.py` is 03.2: `--stage` accepts only `gpu`,
  since the old `modisco`/`all` stages read the `motifs.html` that modisco 2.5's
  `modisco report` no longer writes, and `--device gpu` (the default) makes
  interpretation fail rather than fall back to the CPU. 03.3 runs
  `src/motif_qc.py` -- see below. Same reasoning as 08.0 below.

- **The full model gets the same split: 04.0 trains, 04.4/04.5 interpret.**
  `chrombpnet pipeline`, after training, predictions and marginal footprinting,
  runs DeepLIFT on a 30K peak subsample and TF-MoDISco on the profile scores
  inside the same GPU job. 04.0 passes chrombpnet's own `--skip-interpretation`
  (from the pinned `pipeline-hooks` commit), which stops after the marginal
  footprints and writes the train-mode report. `src/chrombpnet_train.py` patches
  nothing any more: it runs chrombpnet in-process to record the training
  process's peak RSS, after checking that the `prepared_bigwig.json` 00.0 wrote
  beside the `-bw` bigwig records the configured signal path, md5 and assay — a
  mismatch stops the job, because there is no fallback conversion left and a
  stale bigwig would otherwise train on the wrong signal silently. 03.0 goes
  through the same launcher. `src/run_full_model_qc.py --stage gpu` then runs the
  interpretation as pipeline would have, as 04.4 — except that it scores BOTH
  heads (the pipeline's counts run is commented out upstream) — and 04.5 finds
  the motifs with `src/motif_qc.py`. The 30K subsample is
  `utils.regions.subsample_regions`, chrombpnet's own rule (seed 1234), shared
  with the bias QC; DeepSHAP's references take chrombpnet's default seed (1234).
  Nothing downstream waits on 04.4/04.5: they are per-fold
  QC, while 05 gives analysis-grade scores on all peaks and 08 motifs on the
  fold average -- which can hide a bad fold, which is what 04.5 is for.

- **05.0 seeds DeepSHAP per fold: `--shap-seed $((1234 + fold))`.** chrombpnet 2.x
  seeds the 20 dinucleotide-shuffled references from `--shap-seed` and each
  sequence's content, so one seed for every fold would score a peak against the
  same references in all five models, and 06.0's average would stop averaging out
  reference noise the way the unseeded 1.x runs did. The seed is in the run
  metadata and in `interpretation.interpret.args.json`.

- **Per-fold motif QC (03.3, 04.5) is `src/motif_qc.py`, not chrombpnet's
  modisco: TF-MoDISco 2.5.2 at `-n 5000` on BOTH heads, in the chrombpnet 2.x
  env** (which ships modisco 2.5.2 and memelite, so there is no separate `motifs`
  env any more). The changes, each for a measured reason (d0, chrombpnet 1.x bias
  model, fold 0, bias `_065`, 2026-09-23):
  - *Small budget.* Runtime is dominated by clustering, which grows with the
    seqlet count: 63 min at chrombpnet's `-n 50000` (profile head, modisco-lite
    2.0.7) against 6 min at 5000 (counts head, 2.5.2 at its CLI defaults, before
    the flags below were pinned). This is QC; analysis-grade motifs come from
    08.0 at the full budget on the fold average. `motif_qc_max_seqlets` in
    config.yaml raises it.
  - *Both heads.* chrombpnet scores and clusters the profile head only. On d0
    the profile head was Tn5 (90% of seqlets in `TN5_*`-matching patterns),
    while the counts head -- the one behind the bias model's r of 0.56 on
    peaks vs 0.40 on non-peaks -- was GC-rich (positive) and AT-rich
    (negative) composition, invisible from profile alone. So 03.2 and 04.4
    score both heads (04.4 departs from the pipeline to do it).
  - *1.x's settings, passed explicitly:* `-l 2 -z 20 -f 5 -t 20 -g 5 -j 0`, as
    chrombpnet 2.x's own `evaluation/modisco/run.py` passes them; 08.0 passes the
    same. 2.5.2 keeps 2.0.7's `core`/`affinitymat`/`cluster`/`extract_seqlets`
    byte for byte but not its CLI defaults: 2.0.7 hard-coded a 20-bp seqlet core
    and 5-bp flank and trimmed patterns to 20 bp plus 5, while 2.5.2 defaults to
    `-t 30 -g 10`, which makes every pattern 50 bp instead of 30. The flags go
    into `motif_qc.json`, and a summary that lacks them counts as stale, so QC
    made at 2.5.2's defaults is redone once. One difference cannot be set from
    the CLI: `merging_max_seqlets_subsample` rose from 300 to 1000.
  - *The report.* 2.5.2's `modisco report` writes a descriptive `report.html`,
    so chrombpnet's `*_profile.pdf` and pipeline-mode HTML report (which read
    `motifs.html`) are gone from 03.3/04.5. (08.0 calls `modisco report-simple`,
    the one 2.0.7 called `report`, which does write `motifs.html`.)
  - The report matches against chrombpnet's own `motifs.meme.txt`
    (`chrombpnet_motifs_meme`, fetched pinned to v1.0.1 by
    `download-references`), because MotifCompendium has no Tn5 or DNase bias
    motifs. tomtom-lite still assigns broad GC and Alu-repeat patterns to
    `TN5_*` at tiny p-values; read the logos, not the labels.
  - Per-seqlet annotation (tangermeme recursive seqlets + tomtom-lite, as
    cherimoya does) was tried and dropped: 1% of seqlets matched Tn5 on a
    model whose patterns were 90% Tn5.
  - The script pins numba/OpenMP threads to `--threads` (the step passes
    `SLURM_CPUS_PER_TASK`) before anything imports numba, which otherwise
    sizes its pool to the host's cores. chrombpnet's scores h5 is
    filter-compressed: reading it needs `import hdf5plugin`.

- **`08.0.run_modisco.sh` is CPU-only on `engreitz` with `--qos=high_p`, on purpose.**
  tfmodisco-lite doesn't use a GPU, and the default QOS caps walltime at 2 days for
  this account regardless of the partition ceiling; `high_p` (7-day MaxWall) is what
  actually gets the longer runs some datasets need. Don't "fix" it back to `gpu`.
  03.3 and 04.5 (per-fold TF-MoDISco) use the same partition and QOS for the same
  reason.

- **`set -euo pipefail` is the exception, not the rule** — only `00.1`, `01.0` and
  `03.1` use it. `00.1` and `01.0` set it right after sourcing `config.sh`, so they
  enter their pixi environment under `set -u`; `03.1` sets it *after*
  `activate_env`, which is the placement `activate_env`'s own comment asks for
  (neither conda's nor pixi's activation scripts are written against `set -u`).
  Everything else relies on explicit `[[ -f ... ]]` guards and `$?` checks.
  `activate_env` deliberately does not set it. Keep the placement where it is unless
  you've tested the move on the cluster, and don't convert the guard-based scripts
  wholesale.
  **`02.0` does not have it** — this note used to claim it did. It does not, and the
  gap was real: a failed `cli.py qc-signal` printed "QC written to" and exited 0, and
  because this QC is advisory and nothing waits on it, an empty `plots/signal_qc/`
  reads as "QC was fine" rather than "QC never ran". `00.0` had the identical hole.
  Both now carry an explicit `$?` + output-file guard after their python call; if you
  add a step that shells out to `cli.py`, copy that guard.

- **Idempotency markers are per-step and sometimes not the obvious file.** `04.0`
  requires *both* `models/chrombpnet_nobias.h5` and
  `auxiliary/chrombpnet_nobias_footprints.h5` before skipping, because the second is
  the last file pipeline writes before the interpretation 04.0 stops at (after the
  predictions and max-bias-response 04.1/04.3 read) — a preempted job leaves the
  model but not that, and the script `rm -rf`s the directory and retrains. (It
  used to key off `evaluation/chrombpnet_nobias_profile.pdf`, the TF-MoDISco report,
  back when 04.0 ran interpretation itself; that is 04.5's output now.) That `rm -rf` on a
  seemingly-complete model is intentional. chrombpnet 2.x keeps that order: it moves
  the footprints into `auxiliary/` after the predictions and renders the report
  after that. `03.0` likewise needs the model *and* `*_bias_metrics.json`, because
  Keras' `ModelCheckpoint(save_best_only=True)` writes the model after epoch 1.
  `05.0` needs both `interpretation.{counts,profile}_scores.h5` and
  `interpretation.profile_scores.bw`, the last file `contribs_bw` writes (the h5s
  go through `.partial` + rename; a bigwig killed mid-write is truncated and counts
  as done — delete it). `10.0` keys off `hits.bed.gz`, which it now writes under a
  temporary name and moves into place only after `bgzip` and `tabix` both succeed:
  the old redirect left an empty `hits.bed.gz` that made every rerun skip the
  dataset. Others: `motif_report.tsv` (11.0), `*_negatives.bed` (01.0), per head
  `modisco_{head}_results.h5` and `{head}_report/motifs.html` (08.0), and
  `motif_qc.json`'s recorded settings as well as the outputs (03.3, 04.5).

- **`04.3.generate_predictions.sh` hardcodes `for fold in "0" "1" "2" "3" "4"`** rather
  than iterating `"${folds[@]}"`, and collects whichever fold models happen to exist.
  Deliberate for the "average over available folds" behaviour, but it means a dataset
  configured with a fold subset still scans all five.

- **`03.0`'s `--array=0-19` assumes 5 folds × 4 bias factors.** A dataset with a
  different `bias_factors` length needs the range overridden at submit time
  (`igvf11_h7_hesc` has 6 → `0-29`). Nothing in the script validates this; an
  out-of-range index just exits 0.

- **Per-fold bias selection is a manual hand-off.** `03.1` writes
  `selected_bias_per_fold.tsv`; a human copies the winners into `fold_bias_suffix` in
  the dataset's `config.yaml`; `03.2` and `04.0` read that map and fail loudly if a fold is
  missing. The loop is intentionally not closed automatically — the plots are meant to
  be reviewed.

- **`03.1` flags a winner at an end of the swept range** (`sweep_edge` in
  `selected_bias_per_fold.tsv`, plus a log warning and a section in
  `bias_selection_explanation.txt`). `select_best` answers "best of the factors
  we tried"; it cannot see past the range it was given, and the metrics look
  identical whether or not the real optimum lies outside. Note
  `bias_factors: ["0.5", ...]` puts chrombpnet's recommended ATAC start at the
  *floor* of the sweep, while its guidance when a bias model regresses TF motifs
  is to *reduce* the factor — so the default grid can only move the wrong way.
  Ordering is numeric, not lexical: `"10"` sorts before `"5"` as a string.

- **Whether the bias factor needs sweeping per fold is an open question**, with
  the mechanism, the decisive analysis and a decision rule fixed in advance in
  `docs/bias-factor-per-fold.md`. The data to settle it is already on the
  cluster and needs no GPU. Don't shrink the `03.0` grid before running it.

- **`chrombpnet contribs_bw` silently drops peaks whose 2114 bp window runs off a
  chromosome end**, so the averaged H5 from 06 has fewer rows than
  `*_peaks_no_blacklist.narrowPeak`. That's why `07` and `10` take regions from
  `interpretation.interpreted_regions.bed` (fold `${folds[0]}` — all folds filter
  identically) and not from the raw peaks file. Swapping it back produces a silent
  row-count mismatch, not an error.

- **`predict_bias_metrics.py` and `run_bias_qc.py` exist because the `chrombpnet bias qc`
  CLI can't be pointed at a directory `chrombpnet bias train` already populated** — it
  recreates `auxiliary/` and `evaluation/` with `exist_ok=False` and crashes. Both call
  into chrombpnet's Python API directly, reusing the filtered beds `bias train` wrote
  into `auxiliary/`, and load the model through chrombpnet's `load_model_wrapper`,
  which reads 1.x (TF-Keras) and 2.x (Keras 3) model files alike. The observed
  signal is passed explicitly as `--bigwig`: 00.0's prepared bigwig, which 2.x
  trains from in place, so there is no copy in `auxiliary/` to reuse. The split is
  also a cost decision: 03.0 runs only the fast prediction metrics across the whole
  sweep, and the expensive QC runs on the selected model alone — DeepLIFT in 03.2,
  TF-MoDISco in 03.3.

- **`.shellcheckrc` disables SC2154 globally, and that is load-bearing.** Every step
  reads variables that come from the dataset's `config.yaml`, which `config.sh`
  renders and `eval`s at runtime, so shellcheck cannot follow them and would fire
  ~76 times on `datasets`, `peak_type`, `folds`, `genome_fa` and friends. The
  `source-path` entries and the `# shellcheck source=` directives on each source line
  are what make the *resolvable* sources work; don't remove either.

- **The `# shellcheck disable=SC2218` directives work around a shellcheck 0.11.0
  false positive** on functions that are defined before use: in `01.0`, `02.0`,
  `04.2` and `09.0`, and in `workflows/molab/`'s `gcs.sh`, `run_all.sh`,
  `run_step.sh` and `setup_molab.sh`. Retest without them when shellcheck is next
  upgraded.

- **`ruff format` has been applied to `lib/python` and `src`.** The first run
  reformatted 9 of 14 files, so any diff against a pre-split revision is dominated by
  formatting. Run `pixi run fmt` before committing, or let the pre-commit hook do it.

- **10 `B905` findings (`zip()` without `strict=`) are ignored, not fixed.** They all
  pre-date the split. `strict=True` would change behaviour and `strict=False` is
  noise, so each call site needs a look. Re-enable the rule in `.ruff.toml` when
  that's done.
