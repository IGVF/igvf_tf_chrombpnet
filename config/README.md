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
cd workflows/SLURM && sbatch 01.preprocess_peaks.sh
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

Conda locations and reference roots are the same for every dataset and must not
be committed, so they are environment variables with defaults. Put the ones you
need in your shell profile:

None are required. `REPO_ROOT` is also honoured, but only if it really points at
this checkout — the name is generic enough that another project may have exported
it, so the steps verify it and fall back to locating the repo themselves.

| Variable | Default | What it is |
|---|---|---|
| `DATASET_ROOT` | the repo checkout | where `<dataset>/data` and `<dataset>/results` live |
| `REFERENCE_ROOT` | `/oak/stanford/groups/engreitz/Data` | genome, chrom.sizes, blacklist, motif DB |
| `CONDA_INIT` | the Engreitz Sherlock install | `conda.sh` to source |
| `CHROMBPNET_ENV` | *(as above)* | chrombpnet env |
| `PREPROCESS_ENV` | *(as above)* | preprocess env (steps 00, 01) |
| `FINEMO_ENV` | *(as above)* | finemo env (steps 10, 11) |
| `MOTIF_COMPENDIUM_ENV` | *(as above)* | motif_compendium env (step 09) |
| `METADATA_DIR` | `<results>/metadata` | run-metadata output |
| `LOG_LEVEL` | `INFO` | logging level |
| `BOOTSTRAP_PYTHON` | `python3` | python used to read the config before conda |

A dataset's `config.yaml` can still override anything derived from these
(`genome_fa`, `chrom_sizes`, `dataset_dir`, …) for that dataset alone.

## How the YAML reaches bash

`lib/bash/config.sh` renders the YAML to shell assignments and `eval`s them:

```bash
$ python3 lib/python/utils/config.py export config/igvf3_cardiomyocyte/config.yaml
datasets=( igvf3_cardiomyocyte )
folds=( 0 1 2 3 4 )
fragments_path="${dataset_dir}/data/fragments"
declare -A fold_bias_suffix=( [0]=_08 [1]=_08 [2]=_06 [3]=_08 [4]=_07 )
```

`${...}` is left intact so the shell still does the interpolation.

The loader runs on **whatever `python3` is on PATH, before any conda env is
active**, so it imports nothing third-party: it uses PyYAML when importable and
a strict built-in subset parser otherwise. That parser handles scalars, lists
and one level of nesting, and **raises on anything else** rather than guessing —
`tests/test_config.py` asserts it agrees with PyYAML on every config here. If
you need anchors or block scalars, the loader will tell you so.

## What is still not configurable

`#SBATCH` directives are literals in each step: SLURM parses them before any of
our code runs, so they cannot read a variable. Override at submit time:

```bash
sbatch --partition=mypartition --account=myaccount 04.0.train_full_model.sh
```

`00.copy_and_prepare_data.sh` is an explicit staging **template** with source
paths at the top; it is the one step you edit rather than configure.
