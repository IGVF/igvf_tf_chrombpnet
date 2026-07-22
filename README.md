# IGVF TF ChromBPNet

> [!NOTE]
> **This is the actively maintained repository for the project.**

ChromBPNet pipeline for the IGVF TF collaboration. Trains bias-factorised deep learning
models on ATAC-seq pseudobulks from IGVF datasets to learn sequence-based chromatin
accessibility and discover TF binding motifs.

See [`pipeline/README.md`](pipeline/README.md) for full documentation.

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
pipeline/               Pipeline scripts (steps 00–11) and shared config
scripts/bash/           Utilities (download_references.sh: one-time shared-reference setup)
envs/                   Conda environment specs (chrombpnet, finemo, motif_compendium)
folds/                  5-fold cross-validation chromosome splits
<dataset>/              One folder per dataset (data and results not tracked)
  dataset_config.sh       Dataset-specific parameters (tracked)
  data/fragments/         Fragment files (*.tsv.gz)
  data/peaks/             Peak files (*.bed)
  results/                Model outputs written here by pipeline
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
ml biology samtools bedtools
bash scripts/bash/download_references.sh
```

Reference paths (genome, blacklist, motif DB) are set in `pipeline/config.sh` and each
`dataset_config.sh` and point at the shared `$OAK/engreitz/Data` copies by default.

---

## Quick start

Most steps run per dataset — set `DATASET_DIR` before submitting.
Steps 03.1, 04.2, and 10 process all datasets internally and do not need `DATASET_DIR`.

```bash
export DATASET_DIR=/path/to/igvf3_cardiomyocyte   # set per dataset for steps that need it

sbatch pipeline/00.copy_and_prepare_data.sh
sbatch pipeline/01.preprocess_peaks.sh
sbatch pipeline/02.preprocess_nonpeaks.sh
sbatch pipeline/03.0.train_bias_model.sh
sbatch pipeline/03.1.select_bias.sh        # no DATASET_DIR needed; update fold_bias_suffix in dataset_config.sh after
sbatch pipeline/04.0.train_full_model.sh
sbatch pipeline/04.1.qc_run_full_model.sh
sbatch pipeline/04.2.qc_combined_boxplot.sh  # no DATASET_DIR needed; run once all datasets complete 04.1
sbatch pipeline/05.get_contrib_scores.sh
sbatch pipeline/06.average_contrib_scores.sh
sbatch pipeline/07.contribs_to_bigwig.sh
sbatch pipeline/08.run_modisco.sh
sbatch pipeline/09.generate_predictions.sh
sbatch pipeline/10.motif_compendium.sh       # no DATASET_DIR needed; run once all datasets complete 08
sbatch pipeline/11.run_finemo_unified.sh
```

All shared parameters (conda envs, genome paths, output dirs) are in
[`pipeline/config.sh`](pipeline/config.sh). Dataset-specific parameters
(fragment paths, peak files, bias sweep values) are in each dataset's
`dataset_config.sh`.
