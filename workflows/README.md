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
remaining the reference implementation until they are. `molab/` runs the same
`SLURM/` step files on a molab box, without SLURM; see its README.

Every step enters its own pixi environment through `activate_env`
(`lib/bash/common.sh`): chrombpnet 2.x from a pinned chrombpnet checkout, and
`preprocess`, `finemo` and `motif-compendium` from this repo's `pixi.toml`.
Which step uses which is in the Environments table in `../CLAUDE.md`.

---

## Pipeline overview

The step table in `../CLAUDE.md` is the authoritative list, with each step's array
index.

```
Preprocess data                                   00.0 -> 00.1 -> 01.0 -> 02.0 (QC)
Train bias models, select, QC the selection       03.0 -> 03.1 -> 03.2 -> 03.3
Train full model with selected bias, QC           04.0 -> 04.1 (04.2 across datasets)
Predictions; per-fold interpretation QC           04.3; 04.4 -> 04.5
Contribution scores on all peaks                  05.0
Fold averaging and BigWig conversion              06.0 -> 07.0
TF-MoDISco on averaged scores                     08.0
Cross-dataset motif compendium                    09.0
Fi-NeMo hit calling and report                    10.0 -> 11.0
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

`queries.sql` at the repo root defines `runs`, `run_parameters`, `run_files`,
`run_settings`, `run_metrics`, `jobs` and `tools` views
and has worked queries for: latest run per step, failures with a link to the code,
where the wall-clock goes, which run produced a given file, files whose checksum
changed between runs, and runs made from a dirty tree.

There is deliberately no TSV sidecar — a second on-disk format is a second thing to
keep in sync, and DuckDB exports one on demand:

```sql
COPY (SELECT * FROM run_files) TO 'run_files.tsv' (HEADER, DELIMITER '\t');
```
