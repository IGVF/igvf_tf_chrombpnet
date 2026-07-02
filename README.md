# IGVF TF ChromBPNet

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
pipeline/               Pipeline scripts (steps 00–04.0) and shared config
folds/                  5-fold cross-validation chromosome splits
igvf3_cardiomyocyte/    Example dataset folder (data not included)
  dataset_config.sh       Dataset-specific parameters
  data/fragments/         Fragment files go here (*.tsv.gz)
  data/peaks/             Peak files go here (*.bed)
  results/                Model outputs written here by pipeline
```

---

## Quick start

```bash
# Set the dataset you want to run
export DATASET_DIR=/path/to/igvf3_cardiomyocyte

# Run steps in order (Sherlock / SLURM)
bash   pipeline/00.copy_and_prepare_data.sh
sbatch pipeline/01.preprocess_peaks.sh
sbatch pipeline/02.preprocess_nonpeaks.sh
sbatch pipeline/03.0.train_bias_model.sh
bash   pipeline/03.1.select_bias.sh        # update fold_bias_suffix in dataset_config.sh after
sbatch pipeline/04.0.train_full_model.sh
```

All shared parameters (conda envs, genome paths, output dirs) are in
[`pipeline/config.sh`](pipeline/config.sh). Dataset-specific parameters
(fragment paths, peak files, bias sweep values) are in each dataset's
`dataset_config.sh`.
