# config/

One YAML file per dataset. That is the whole configuration.

```
config/
  example_dataset/config.yaml     annotated template
  igvf3_cardiomyocyte/config.yaml a real, working example
```

## Running a dataset

```bash
export DATASET=igvf3_cardiomyocyte     # -> config/igvf3_cardiomyocyte/config.yaml
cd workflows/SLURM && sbatch 00.1.preprocess_peaks.sh
```

Or point at a file anywhere: `export DATASET_CONFIG=/path/to/config.yaml`.

## Adding a dataset

```bash
cp -r config/example_dataset config/my_dataset
$EDITOR config/my_dataset/config.yaml
pixi run -e qc python src/cli.py config validate --dataset my_dataset
export DATASET=my_dataset
```

## Machine-specific paths are environment variables, not a config file

Where the software and the reference files live is the same for every dataset
and must not be committed, so those are environment variables with defaults
(`lib/bash/common.sh`). Put the ones you need in your shell profile; `sbatch`
passes them on to the job.

Only `CHROMBPNET_REPO` is required (unless `CHROMBPNET_ENV` names the environment
outright): every step from 01.0 to 08.0 runs in the chrombpnet environment, and
its checkout has no default location. `pixi` itself must be on `PATH`.
`REPO_ROOT` is also honoured, but only if it really points at this checkout — the name is generic
enough that another project may have exported it, so the steps verify it and fall
back to locating the repo themselves.

| Variable | Default | What it is |
|---|---|---|
| `DATASET_ROOT` | the repo checkout | where `<dataset>/data` and `<dataset>/results` live |
| `REFERENCE_ROOT` | `/oak/stanford/groups/engreitz/Data` | genome, chrom.sizes, blacklist, motif DBs |
| `CHROMBPNET_REPO` | *(none)* | the chrombpnet 2.x checkout (`NNFC-GMD/chrombpnet`) |
| `CHROMBPNET_REV` | the commit pinned in `common.sh` | the commit that checkout should be at; `activate_env` warns otherwise |
| `CHROMBPNET_PIXI_ENV` | `cuda13` | environment in the checkout's `pyproject.toml`; `cuda12` for NVIDIA drivers below 580 |
| `CHROMBPNET_ENV` | `pixi:${CHROMBPNET_REPO}/pyproject.toml#${CHROMBPNET_PIXI_ENV}` | chrombpnet env (01.0, 03.0–08.0) |
| `PREPROCESS_ENV` | `pixi:<checkout>/pixi.toml#preprocess` | preprocess env (00.0, 00.1, 02.0, `qc_datasets.sh`) |
| `FINEMO_ENV` | `pixi:<checkout>/pixi.toml#finemo` | finemo env (10.0, 11.0); `#finemo-cu126` for drivers below 580 |
| `MOTIF_COMPENDIUM_ENV` | `pixi:<checkout>/pixi.toml#motif-compendium` | motif-compendium env (09.0) |
| `CONDA_INIT` | the Engreitz Sherlock install | `conda.sh` to source — only for a `*_ENV` given as a plain conda prefix |
| `CONDA_OVERRIDE_CUDA` | `13.0` (set by `activate_env`) | lets pixi install or enter a CUDA environment on a node without a GPU |
| `METADATA_DIR` | `<results>/metadata` | run-metadata output |
| `LOG_LEVEL` | `INFO` | logging level |
| `BOOTSTRAP_PYTHON` | the first python >= 3.9 on `PATH`, else this checkout's pixi `preprocess` or `default` env | python used to read the config (and write run metadata) before any environment is active |

A `*_ENV` value is either `pixi:<manifest>#<environment>` or a conda prefix path.

A dataset's `config.yaml` can still override anything derived from these
(`genome_fa`, `chrom_sizes`, `dataset_dir`, …) for that dataset alone.

## How the YAML reaches bash

`lib/bash/config.sh` renders the YAML to shell assignments and `eval`s them:

```bash
$ python3 lib/python/utils/config.py export config/igvf3_cardiomyocyte/config.yaml
...
output_dir="${DATASET_ROOT}/igvf3_cardiomyocyte/results"
folds=( 0 1 2 3 4 )
declare -A fold_bias_suffix=( [0]=_08 [1]=_08 [2]=_06 [3]=_08 [4]=_07 )
datasets=( igvf3_cardiomyocyte )
...
```

`${...}` is left intact so the shell still does the interpolation.

The loader runs on **whatever python >= 3.9 `bootstrap_python` finds, before any
environment is active**, so it imports nothing third-party: it uses PyYAML when
importable and a strict built-in subset parser otherwise. That parser handles scalars, lists
and one level of nesting, and **raises on anything else** rather than guessing —
`tests/test_config.py` asserts it agrees with PyYAML on every config here. If
you need anchors or block scalars, the loader will tell you so.

## What is still not configurable

`#SBATCH` directives are literals in each step: SLURM parses them before any of
our code runs, so they cannot read a variable. Override at submit time:

```bash
sbatch --partition=mypartition --account=myaccount 04.0.train_full_model.sh
```

Nothing is copied: `regions` and `signal_path` point at your data where it
already is, and `00.0.prepare_signal.sh` writes only derived files under
`output_dir`.
