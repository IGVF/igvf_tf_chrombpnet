# ChromBPNet pipeline

ChromBPNet (Chromatin BPNet) is a bias-factorised deep learning model that predicts
per-base ATAC-seq accessibility from DNA sequence. It decomposes the signal into a Tn5
bias model (sequence preferences of the Tn5 transposase) and a ChromBPNet model that
captures true biological accessibility. The bias model is subtracted so that the final
`chrombpnet_nobias` model only learns TF binding motifs and chromatin accessibility
signals, not Tn5 sequence artifacts.

This pipeline trains ChromBPNet across several IGVF datasets using
5-fold cross-validation, then discovers and annotates the regulatory motifs active at
each stage.

All paths and shared parameters live in `config.sh`. Source it at the top of any new
script.

---

## Pipeline overview

```
Stage 1,2   Preprocess data                               00, 01, 02
Stage 3     Train bias models and QC                      03.0 -> 03.1
Stage 4     Train full model with selected bias and QC    04.0 -> 04.1
Stage 5     Contribution scores                           05
Stage 6,7   Fold averaging and BigWig conversion          06, 07
Stage 8     MoDISco on averaged scores                    08
Stage 9     Generate predictions                          09
Stage 10,11 Motif compendium and Fi-NeMo                  10 -> 11
```

---

[work in progress]
