# igvf_tf_chrombpnet

ChromBPNet pipeline for the IGVF TF collaboration: trains bias-factorised models on
ATAC-seq pseudobulks, averages contribution scores across 5 folds, and discovers
motifs (TF-MoDISco → MotifCompendium → Fi-NeMo).

**This is a SLURM pipeline, not a package.** There is no `pyproject.toml`, no
installable module and no CI. There *is* a test suite (`tests/`, `pixi run -e qc
test`) covering the pure-Python helpers in `lib/python/utils` and the selection
logic in `src/`; it never touches the cluster. It is a numbered sequence of `sbatch`
scripts under `workflows/SLURM/` plus the Python they call, all of which assume
Stanford Sherlock: `ml` modules, `/oak/stanford/groups/engreitz/...` data, and
`/home/groups/engreitz/...` conda environments. Nothing here runs on a laptop except
lint and syntax checks, which `pixi` provides.

`README.md` is the user-facing reference (dataset table, setup, quick start). This
file is the contributor-facing complement: don't duplicate the README, point at it.
`workflows/README.md` is a `[work in progress]` stub with a stage overview and the
rules for adding a step.

## Layout
Four top-level roles: `workflows/` orchestrates, `src/` does one job per file,
`lib/` is shared code, and `<dataset>/` is configuration.

The repo is checked out **as the collaboration root**: `09.0.cross_dataset_compendium.sh`
and `10.0.run_finemo_unified.sh` resolve per-dataset results under `${REPO_ROOT}` and
expect `<dataset>/results/...` as siblings of `workflows/`. Only
`igvf3_cardiomyocyte/` is tracked (its `dataset_config.sh` plus `.gitkeep`s); the
other three dataset directories exist on the cluster only.

- `workflows/SLURM/` — one `sbatch` script per step. See the step table below.
- `workflows/nextflow/` — empty placeholder for the Nextflow port. `SLURM/` stays the
  reference implementation until it isn't.
- `src/` — the atomic Python scripts the steps call, one concern each. Nothing in
  `src/` imports anything from another file in `src/`; shared code goes to `lib/`.
- `lib/bash/common.sh` — settings and helpers with no `DATASET_DIR`: conda env paths,
  `ref_db_meme`, algorithm thresholds, and exactly four helpers —
  `activate_env`, `load_render_modules`, `load_gpu_modules`, `gpu_env`. Steps still
  write progress with inline `echo "[$(date)] ..."` and check inputs with
  `[[ -f ... ]] || { echo ...; exit 1; }`; adding a `log()`/`require_file()` means
  converting those ~40 call sites in the same change, not adding a second way.
- `lib/bash/config.sh` — the per-dataset layer. Sources `common.sh`, then
  `${DATASET_DIR}/dataset_config.sh`, then derives every output path.
- `lib/python/utils/` — the importable helper package: `intervals` (pyranges1 ops
  replacing bedtools), `references` (ENCODE accession -> URL), `compression`
  (bgzip/tabix), `folds` (reads `folds/*.json`), `palettes`, `plotting`, `regions`.
  Two import rules, both load-bearing: nothing here may import `chrombpnet`,
  `tensorflow`, `finemo` or `MotifCompendium`, and **only `intervals` may import
  `pyranges1`** (it needs Python >= 3.12, which only the `preprocess` env has).
  The package is called `utils` and sits at the front of `sys.path`, so it would
  shadow any third-party top-level `utils`; nothing in the four envs ships one.
- `folds/fold_{0..4}.json` — the 5-fold chromosome splits (`train`/`valid`/`test`),
  shipped with the repo and passed to chrombpnet as `-fl`.
- `envs/{chrombpnet,finemo,motif_compendium}.yml` — fully pinned conda exports.
- ``cli.py download-references`` — one-time, idempotent fetch of genome /
  chrom.sizes / blacklist / MotifCompendium into the shared lab `Data/` folder.
- `lib/python/utils/shift.py` — vendored from scPrinter (Ruochi Zhang),
  not wired into any pipeline step.
- `<dataset>/dataset_config.sh` — per-dataset parameters, sourced by `config.sh`.

### Configuration
**One YAML file per dataset — `config/<dataset>/config.yaml` — is the whole
configuration.** There is no second "site"/"env" file: the machine-specific
values (conda paths, `REFERENCE_ROOT`, `DATASET_ROOT`) are environment
variables with defaults in `common.sh`, because they are identical across
datasets and must not be committed. `config/README.md` has the table.

Selected by `DATASET=<name>`, or `DATASET_CONFIG=/path/to/config.yaml`.
`dataset_dir` defaults to `${data_root}/<dataset>`, so the tracked config stays
small while the data lives wherever you like; `config.sh` re-exports
`DATASET_DIR` afterwards for the steps that reference it.

**The YAML reaches bash through `lib/python/utils/config.py`**, which renders it
to shell assignments that `config.sh` evals (`${...}` left intact so the shell
still interpolates). That loader **must run on the bare system python before any
conda env is active**, so it imports nothing third-party and nothing else from
`utils` — don't add imports to it. It uses PyYAML when importable and a strict
built-in subset parser otherwise; the subset covers scalars, flow/block lists and
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
`load_gpu_modules` so the check is cheap; the wiring in every step is ordered that
way deliberately, so don't move the env setup back above it. `load_*_modules` and
`activate_env` no-op or fail with a clear message when `ml`/conda are absent, so a
step reaches its own preflight instead of dying on a missing module system.
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

Dataset-specific things (fragments path, genome refs, `bias_factors`,
`fold_bias_suffix`) live in `dataset_config.sh`; shared things live in `common.sh`;
derived output paths live in `config.sh`.

### Steps
`--array` index means something different per step — check before overriding it.

| Script | Array index | Needs `DATASET_DIR` |
|---|---|---|
| `00.0.prepare_signal.sh` | — (loops internally) | no (hardcoded paths) |
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
| `05.0.get_contrib_scores.sh` | fold | yes |
| `06.0.average_contrib_scores.sh` | dataset | yes |
| `07.0.contribs_to_bigwig.sh` | dataset | yes |
| `08.0.run_modisco.sh` | dataset | yes |
| `09.0.cross_dataset_compendium.sh` | — | no (hardcoded `h5_map`) |
| `deprecated/motif_compendium.sh` | — | yes |
| `10.0.run_finemo_unified.sh` | dataset | yes |
| `11.0.postprocess_finemo.sh` | dataset | yes |
| `qc_datasets.sh` | — | no (sets a dummy one) |

All step scripts live in `workflows/SLURM/`. The Python they call lives in `src/`
and is referenced as `${src_dir}/<name>.py`, never by a relative path:
`predict_bias_metrics.py` (03.0),
`select_bias_model.py` (03.1), `run_bias_qc.py` (03.2), `qc_full_model.py` (04.1 and
04.2-combined), `predict_and_avg.py` (04.2), `average_contrib_scores.py` (06),
`contribs_to_bigwig.py` (07), `motif_compendium.py` (09 and `_10`),
`qc_datasets.py` (`qc_datasets.sh`).

## Commands
Everything local goes through `pixi` (0.76+). It covers lint and syntax checks, plus a
`qc` environment that can run the plotting scripts on results copied off the cluster.

```bash
pixi install
pixi run check           # check-bash + check-py + lint
pixi run check-bash      # bash -n over every script, then shellcheck -S warning
pixi run check-py        # byte-compile lib/python, src, scripts/python
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

On the cluster, nothing uses pixi:

```bash
conda env create -f envs/chrombpnet.yml     # once per cluster
bash `cli.py download-references`    # once per cluster

export DATASET_DIR=/path/to/igvf3_cardiomyocyte
cd workflows/SLURM && sbatch 00.1.preprocess_peaks.sh
```

### Environments
Four, and they are not interchangeable:

| Where | What | Managed by |
|---|---|---|
| `chrombpnet` | training, contribs, predictions, MoDISco, all QC plots | `envs/chrombpnet.yml` |
| `finemo` | steps 10, 11 | `envs/finemo.yml` |
| `motif_compendium` | step 09, `_10` | `envs/motif_compendium.yml` |
| pixi `default` / `qc` | local lint + syntax + plotting only | `pixi.toml` |

**pixi deliberately does not reproduce the three cluster environments.** chrombpnet
1.0.1 pins `tensorflow==2.8.0` / `keras==2.8.0` / `numpy==1.23.4` as pip dependencies
against CUDA 11 wheels; finemo and MotifCompendium pin mutually incompatible `torch`
and `cupy` builds. The `envs/*.yml` files are full `conda env export` output and stay
canonical. Don't "finish the migration" by moving them into `pixi.toml` — the point of
pixi here is that `check-bash`/`check-py`/`lint` run on a laptop in seconds.

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
  through `PYTHONPATH` and not through an install. It works identically under the
  cluster conda envs, under pixi and under a bare `python`, which is the point. The
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
  directory holds only `dataset_config.sh` and `.gitkeep`s. Note `*.tsv` is ignored,
  so pipeline output tables cannot be committed even deliberately.
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
  `metadata_params`/`metadata_tools`, and an EXIT trap emits via
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
  - It shells out to bedtools, so it activates `${CONDA_ENV}` (the chrombpnet
    env), not `${preprocess_conda}`.
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

- **Untracked `dataset_config.sh` copies on the cluster still set `folds_dir`.**
  The tracked `igvf3_cardiomyocyte` one no longer does — `lib/bash/common.sh` sets it
  to `${REPO_ROOT}/folds`. The three cluster-only datasets (igvf6, igvf11,
  igvf_endothelial) still carry `folds_dir="${SCRIPT_DIR}/../folds"`, which stopped
  resolving when the steps moved. `config.sh` checks the directory exists and fails
  with the fix in the message, so this surfaces immediately rather than several
  minutes into a GPU job — but **each of those three files needs the line deleted.**

- **Absolute `opushkar` paths remain in three places**: the three conda env
  defaults in `lib/bash/common.sh` (each overridable: `CHROMBPNET_ENV`,
  `FINEMO_ENV`, `MOTIF_COMPENDIUM_ENV`), `core_path` in `src/qc_datasets.py`,
  and the `runs` view's glob in `queries.sql` (point it at your own metadata;
  the molab notebook rewrites it when it loads the views). They are no longer
  duplicated — `09.0.cross_dataset_compendium.sh` used to re-declare several of
  them "mirroring config.sh" and now sources `common.sh`.

- **The endothelial dataset is named two ways and laid out differently.**
  `qc_datasets.py` calls it `igvf17_endothelial`; `README.md`, `qc_full_model.py` and
  `09` call it `igvf_endothelial`. In `09`'s `h5_map` its MoDISco H5 sits at
  `results/contrib_scores/modisco/...` while the other three use
  `results/contrib_scores/<dataset>/modisco/...`. Both are load-bearing paths; changing
  either without checking the cluster layout will silently drop the dataset (the loop
  only `[WARN]`s on a missing H5).

- **`src/qc_full_model.py` still carries leftovers from another pipeline.**
  `--datasets` defaults to `["d0", …, "d4"]`, and `contribs_to_bigwig.py` /
  `average_contrib_scores.py` use `d0` in their usage examples. Its hardcoded
  `TEST_CHROMS` is gone — it now calls `utils.folds.test_chroms()`, which reads the
  same `folds/fold_<n>.json` chrombpnet trained against. The two were verified
  identical for all five folds before the swap.

- **Steps 06/07/08 run `profile`; steps 09/10 run `counts`.** `score_types=("profile")`
  in 06, 07 and 08 is deliberate — counts was already computed and is left out so
  reruns don't touch it (the helpers also skip an existing output). But
  `09.0.cross_dataset_compendium.sh` hardcodes `modisco_counts_results.h5` and
  `10.0.run_finemo_unified.sh` hardcodes `_average_shaps.counts.h5`, so the compendium
  and Fi-NeMo consume counts only. Adding `"counts"` back to `score_types` is safe;
  the reverse — assuming the downstream steps follow `score_types` — is not.

- **The GPU compute-capability constraint is applied unevenly.** `03.0` and `04.0` pin
  `--constraint="GPU_CC:7.0|7.5|8.0|8.6"` because the loaded `cuda/11.5.0` can't drive
  Ada (8.9) or Hopper (9.0). `03.2`, `04.3.generate_predictions`, `05` and `10` load
  the same cuda/cudnn modules with no constraint. If a GPU step fails oddly on
  `owners`, that's the first thing to check.

- **`03.2` is the GPU half and `03.3` the CPU half of the selected-bias QC.**
  `pipelines.bias_model_qc()` runs predictions, DeepLIFT interpretation, then
  TF-MoDISco and its reports in one call. Only the first two need a GPU;
  TF-MoDISco is CPU-only and is the long pole, so running them together left a
  GPU idle for hours. `src/run_bias_qc.py --stage {gpu,modisco,all}` splits
  them and each stage skips work whose outputs exist, so a failed MoDISco
  re-runs without redoing interpretation. `all` keeps the original behaviour.
  Same reasoning as 08.0 below.

- **`08.0.run_modisco.sh` is CPU-only on `engreitz` with `--qos=high_p`, on purpose.**
  tfmodisco-lite doesn't use a GPU, and the default QOS caps walltime at 2 days for
  this account regardless of the partition ceiling; `high_p` (7-day MaxWall) is what
  actually gets the longer runs some datasets need. Don't "fix" it back to `gpu`.

- **`set -euo pipefail` is the exception, not the rule** — only `01.0` and `03.1`
  use it, and in `03.1` it is placed *after* `activate_env`, not at the top.
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
  `evaluation/chrombpnet_nobias_profile.pdf` before skipping, because the second is the
  last file the evaluation stage writes — a preempted job leaves the model but no
  report, and the script `rm -rf`s the directory and retrains. That `rm -rf` on a
  seemingly-complete model is intentional. Other steps key off `hits.bed.gz` (10),
  `motif_report.tsv` (11), `interpretation.counts_scores.{h5,bw}` (05) or
  `*_negatives.bed` (02).

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
  `dataset_config.sh`; `03.2` and `04.0` read that map and fail loudly if a fold is
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
  `interpretation.interpreted_regions.bed` (fold 0 — all folds filter identically) and
  not from the raw peaks file. Swapping it back produces a silent row-count mismatch,
  not an error.

- **`predict_bias_metrics.py` and `run_bias_qc.py` exist because the `chrombpnet bias qc`
  CLI can't be pointed at a directory `chrombpnet bias train` already populated** — it
  recreates `auxiliary/` and `evaluation/` with `exist_ok=False` and crashes. Both call
  into chrombpnet's Python API directly, reusing the filtered beds and shifted bigwig
  already on disk. The split is also a cost decision: 03.0 runs only the fast
  prediction metrics across the whole sweep, and the expensive DeepLIFT + TF-MoDISco QC
  runs in 03.2 on the selected model alone.

- **`.shellcheckrc` disables SC2154 globally, and that is load-bearing.** Every step
  reads variables that come from `${DATASET_DIR}/dataset_config.sh` — a file outside
  the repo whose path is known only at runtime — so shellcheck cannot follow it and
  fires ~76 times on `datasets`, `peak_type`, `folds`, `genome_fa` and friends. The
  `source-path` entries and the `# shellcheck source=` directives on each source line
  are what make the *resolvable* sources work; don't remove either.

- **Two `# shellcheck disable=SC2218` directives are working around a shellcheck
  0.11.0 false positive** (`09.0.cross_dataset_compendium.sh`,
  ``cli.py download-references``). Both functions are defined before use;
  verified by hand. Retest without them when shellcheck is next upgraded.

- **`ruff format` has been applied to `lib/python` and `src`.** The first run
  reformatted 9 of 14 files, so any diff against a pre-split revision is dominated by
  formatting. Run `pixi run fmt` before committing, or let the pre-commit hook do it.

- **10 `B905` findings (`zip()` without `strict=`) are ignored, not fixed.** They all
  pre-date the split. `strict=True` would change behaviour and `strict=False` is
  noise, so each call site needs a look. Re-enable the rule in `.ruff.toml` when
  that's done.
