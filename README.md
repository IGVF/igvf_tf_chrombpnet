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
  nextflow/             placeholder for the Nextflow port
src/                    cli.py (Click command group) + the remaining argparse tools
lib/
  bash/                 config.sh (per-dataset) + common.sh (shared settings, helpers)
  python/utils/       importable helpers shared by src/ (intervals, references,
                      compression, folds, log, palettes, plotting, regions)
scripts/bash/           Utilities (download_references.sh: one-time shared-reference setup)
scripts/python/         Standalone tools not wired into the pipeline
envs/                   Conda environment specs (chrombpnet, finemo, motif_compendium)
folds/                  5-fold cross-validation chromosome splits
pixi.toml               Local dev environment (lint + syntax checks + QC plotting)
<data_root>/<dataset>/  Data and results (never tracked; set data_root in site.sh)
  data/fragments/         Fragment files (*.tsv.gz)
  data/peaks/             Peak files (*.bed)
  results/                Model outputs, plots and run metadata
```

---

## Setup (one-time per cluster)

```bash
# 1. Recreate the conda environments from the pinned specs
conda env create -f envs/chrombpnet.yml
conda env create -f envs/finemo.yml
conda env create -f envs/motif_compendium.yml

# 2. Fetch the shared genome / chrom.sizes / blacklist / MotifCompendium references
#    into the lab Data/ folder (idempotent; verifies existing files)
pixi run -e qc python src/cli.py download-references
```

Reference paths (genome, blacklist, motif DB) are set in `lib/bash/common.sh` and each
`dataset_config.sh` and point at the shared `$OAK/engreitz/Data` copies by default.

---

## Quick start

Most steps run per dataset — set `DATASET_DIR` before submitting.
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
sbatch 03.2.qc_selected_bias.sh
sbatch 04.0.train_full_model.sh
sbatch 04.1.qc_run_full_model.sh
sbatch 04.2.qc_combined_boxplot.sh     # no DATASET needed; run once all datasets finish 04.1
sbatch 04.3.generate_predictions.sh
sbatch 05.0.get_contrib_scores.sh
sbatch 06.0.average_contrib_scores.sh
sbatch 07.0.contribs_to_bigwig.sh
sbatch 08.0.run_modisco.sh
sbatch 09.0.cross_dataset_compendium.sh  # no DATASET needed; run once all datasets finish 08
sbatch 10.0.run_finemo_unified.sh
sbatch 11.0.postprocess_finemo.sh
```

All shared parameters (conda envs, references, algorithm thresholds) are in
[`lib/bash/common.sh`](lib/bash/common.sh); the per-dataset output layout is derived in
[`lib/bash/config.sh`](lib/bash/config.sh). Dataset-specific parameters (fragment paths,
peak files, bias sweep values) are in each dataset's `dataset_config.sh`.

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

The cluster stack is not installable on a laptop. `pixi` covers the local checks only:

```bash
pixi install
pixi run check           # bash -n + shellcheck + ruff + byte-compile
pixi run hooks-install   # enable the pre-commit hook
```

The three cluster environments stay canonical in `envs/*.yml` — see `CLAUDE.md`.
