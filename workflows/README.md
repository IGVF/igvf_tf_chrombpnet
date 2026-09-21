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

Steps live in `SLURM/`; the Python they call lives in `../src/`; shared bash settings
and helpers live in `../lib/bash/`. A new per-dataset step starts with the standard
bootstrap block and one `source "${REPO_ROOT}/lib/bash/config.sh"` line — copy an
existing step. Cross-dataset steps source `lib/bash/common.sh` instead.

`nextflow/` is a placeholder: the same steps are to be ported there, with `SLURM/`
remaining the reference implementation until they are.

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

---

## Run metadata

Every step writes one JSON record per invocation describing what it read, what it
wrote (with md5s), the tool versions, the git commit and a GitHub permalink to the
step script. They share one schema, so every run of every step across every
dataset loads as a single DuckDB table:

```sql
SELECT * FROM read_json_auto(
    '<collab-root>/*/results/metadata/**/*.json', union_by_name = true);
```

`queries.sql` at the repo root defines `runs`, `run_files` and `run_params` views
and has worked queries for: latest run per step, failures with a link to the code,
where the wall-clock goes, which run produced a given file, files whose checksum
changed between runs, and runs made from a dirty tree.

There is deliberately no TSV sidecar — a second on-disk format is a second thing to
keep in sync, and DuckDB exports one on demand:

```sql
COPY (SELECT * FROM run_files) TO 'run_files.tsv' (HEADER, DELIMITER '\t');
```
