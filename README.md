# IGVF TF ChromBPNet

> [!NOTE]
> **This is the actively maintained repository for the project.**

ChromBPNet pipeline for the IGVF TF collaboration. Trains bias-factorised deep learning
models on ATAC-seq pseudobulks from IGVF datasets to learn sequence-based chromatin
accessibility and discover TF binding motifs.

See [`workflows/README.md`](workflows/README.md) for full documentation.

---

## Datasets

| Dataset ID | Cell type |
|---|---|
| igvf3_cardiomyocyte | WTC11 cardiomyocyte |
| igvf6_definitive_endoderm | Definitive endoderm |
| igvf11_h7_hesc | H7 hESC |
| igvf_endothelial | iPSC-derived endothelial cells, d3 (Engreitz lab 5-timepoint multiome) |

---

## Repo structure

```
config/
  example_dataset/config.yaml      annotated dataset template
  igvf3_cardiomyocyte/config.yaml  a real, working example
workflows/
  SLURM/                sbatch scripts, one per pipeline step (00–11) + status.sh
  molab/                launcher for running the same steps on a molab box (no SLURM)
  nextflow/             placeholder for the Nextflow port
src/                    cli.py (Click command group) + the remaining argparse tools
lib/
  bash/                 config.sh (per-dataset) + common.sh (shared settings, helpers)
                        + references.sh (reference paths)
  python/utils/         importable helpers shared by src/ (intervals, references,
                        compression, config, folds, log, metadata, palettes,
                        pileup, plotting, qc, regions)
folds/                  5-fold cross-validation chromosome splits
pixi.toml               every environment except chrombpnet's (preprocess, finemo,
                        motif-compendium) plus local dev (lint, tests, QC plotting)
<DATASET_ROOT>/<dataset>/  Data and results (never tracked; DATASET_ROOT defaults
                           to the checkout); a config's output_dir says where results go
```

---

## Setup (one-time per cluster)

Every environment is a [pixi](https://pixi.sh) environment, so `pixi` must be on
`PATH` on the login node and in every job. chrombpnet 2.x is not in this repo's
`pixi.toml`: it comes from a separate checkout of `NNFC-GMD/chrombpnet`, pinned to
the commit in `lib/bash/common.sh` (`CHROMBPNET_REV`) and installed from that
repo's own lock file.

```bash
# 1. chrombpnet 2.x, from its own checkout (keep this clone for the pipeline only)
export CHROMBPNET_REPO=/path/to/chrombpnet          # also in your profile: every job needs it
git clone https://github.com/NNFC-GMD/chrombpnet "$CHROMBPNET_REPO"
# the pinned SHA, read from lib/bash/common.sh (run from this checkout)
export CHROMBPNET_REV=$(REPO_ROOT=$PWD bash -c 'source lib/bash/common.sh >/dev/null && echo "$CHROMBPNET_REV"')
(cd "$CHROMBPNET_REPO" && git checkout --detach "$CHROMBPNET_REV")
CONDA_OVERRIDE_CUDA=13.0 pixi install --locked \
    --manifest-path "$CHROMBPNET_REPO/pyproject.toml" -e cuda13

# 2. This repo's environments, from the checkout
pixi install --locked -e preprocess
pixi install --locked -e finemo
pixi install --locked -e motif-compendium

# 3. Fetch the shared genome / chrom.sizes / blacklist / GENCODE TSSs / motif
#    databases into $REFERENCE_ROOT (idempotent; md5-checks all but the blacklist)
pixi run -e preprocess python src/cli.py download-references
```

`CONDA_OVERRIDE_CUDA` lets a login node without a GPU install the CUDA build. The
`cuda13` environment needs an NVIDIA driver >= 580 on the GPU nodes;
`CHROMBPNET_PIXI_ENV=cuda12` is the fallback for older drivers, and
`FINEMO_ENV="pixi:<this checkout>/pixi.toml#finemo-cu126"` the one for Fi-NeMo
(see `10.0.run_finemo_unified.sh`). [`config/README.md`](config/README.md) lists
every variable.

**Upgrading an install that ran chrombpnet 1.x:** re-run `download-references` —
the MotifCompendium database is now pinned to v1.0.19 under a versioned file
name, and 08.0 and 09.0 stop at their preflight until it is there. And give 2.x
runs a new `output_dir`: the steps skip work whose output files exist, so they
would treat 1.x results in the same tree as done.

Reference paths (genome, chrom.sizes, blacklist, motif DBs) are defined once, in
`lib/python/utils/references.py`, under `REFERENCE_ROOT` (default: the shared
`/oak/stanford/groups/engreitz/Data` copy). A dataset's `config.yaml` can set
`reference_root`, or override single files, for that dataset alone.

---

## Quick start

Most steps run per dataset — set `DATASET` (or `DATASET_CONFIG`) before submitting.
Steps 04.2, 09.0 and qc_datasets span all datasets and need no `DATASET`.
Every step is `NN.M.name.sh`, so listing order is execution order.

**Submit from `workflows/SLURM/`.** The steps locate the repo by walking up from the
directory you ran `sbatch` in; set `REPO_ROOT` to submit from anywhere else.

Every step checks its inputs first and, if any are missing, names the step that
produces them and prints the command to run — so submitting out of order costs you
a few seconds, not a queued GPU job. `bash status.sh` shows the whole picture.

```bash
export DATASET=igvf3_cardiomyocyte   # picks config/igvf3_cardiomyocyte/
cd workflows/SLURM

sbatch 00.0.prepare_signal.sh          # CPU: filter + convert once, so GPU jobs skip it
sbatch 00.1.preprocess_peaks.sh
sbatch 01.0.preprocess_nonpeaks.sh    # GC-matched background, via chrombpnet
sbatch 02.0.qc_training_data.sh        # read this before spending GPU time
sbatch 03.0.train_bias_model.sh
bash   03.1.select_bias.sh             # not a batch job; copy the winners into config.yaml after
sbatch 03.2.qc_selected_bias.sh        # GPU: predictions + DeepSHAP of the selected bias model
sbatch 03.3.modisco_selected_bias.sh   # CPU: its per-fold motif QC
sbatch 04.0.train_full_model.sh
sbatch 04.1.qc_run_full_model.sh
sbatch 04.2.qc_combined_boxplot.sh     # no DATASET needed; run once all datasets finish 04.1
sbatch 04.3.generate_predictions.sh
sbatch 04.4.qc_full_model_interpret.sh # GPU: per-fold DeepSHAP QC (only 04.5 waits on it)
sbatch 04.5.modisco_full_model.sh      # CPU: per-fold motif QC
sbatch 05.0.get_contrib_scores.sh
sbatch 06.0.average_contrib_scores.sh
sbatch 07.0.contribs_to_bigwig.sh
sbatch 08.0.run_modisco.sh
sbatch 09.0.cross_dataset_compendium.sh  # no DATASET needed; run once all datasets finish 08
sbatch 10.0.run_finemo_unified.sh
sbatch 11.0.postprocess_finemo.sh
```

All shared parameters (environments, the chrombpnet pin, algorithm thresholds) are in
[`lib/bash/common.sh`](lib/bash/common.sh); the per-dataset output layout is derived in
[`lib/bash/config.sh`](lib/bash/config.sh). Dataset-specific parameters (signal file,
peak file, output directory, bias sweep values) are in each dataset's
`config/<dataset>/config.yaml` — see [`config/README.md`](config/README.md).

---

## Where am I?

```bash
export DATASET=igvf3_cardiomyocyte
bash workflows/SLURM/status.sh      # what has run, and what to run next
```

Every run also writes a metadata JSON (inputs, outputs, md5s, tool versions, git
commit, GitHub permalink). Load them all into DuckDB with the recipes in
[`queries.sql`](queries.sql).

---

## Local development

The steps need the cluster (or a molab box — see
[`workflows/molab/README.md`](workflows/molab/README.md)). Locally, `pixi` runs the
checks and the tests:

```bash
pixi install
pixi run check           # bash -n + shellcheck + ruff + byte-compile
pixi run -e qc test      # pytest
pixi run hooks-install   # enable the pre-commit hook
```

Which step runs in which environment, and why chrombpnet is installed from its own
checkout rather than from `pixi.toml`, is in `CLAUDE.md` under *Environments*.
