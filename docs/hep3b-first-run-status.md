# First real run of the refactored pipeline — status

**As of 2026-09-21.** Dataset: HEP3B (`config/HEP3B/config.yaml`), data in
`/oak/stanford/groups/engreitz/Users/emattei/projects/nnfc_hep3b`, outputs in
that project's `results/chrombpnet_refactor/`.

This is a resume point, not a design doc. It records what ran, what is still
queued, and the order to pick things up in.

## Where it got to

| Step | State |
|---|---|
| `00.0.prepare_signal` | **done** — bigwig + sidecar written |
| `00.1.preprocess_peaks` | **done** — 155,207 of 155,267 peaks kept |
| `01.0.preprocess_nonpeaks` | **done** — all 5 folds of GC-matched negatives |
| `02.0.qc_signal_peaks` | **done** — QC written, and it passes |
| `03.0.train_bias_model` | **queued, never yet executed** |
| `03.1` – `04.3` | not run; changed but only `bash -n` + source review |

`workflows/SLURM/status.sh` reports the same and prints the next command.

### The QC says the data is worth GPU time

`auroc_peaks_vs_nonpeaks` **0.974** (0.5 would mean don't bother),
`tss_enrichment` **15.6**, `frac_peaks_zero_signal` **0.0018**,
`frac_insertions_in_peaks` 0.47, 104 M insertions,
`enough_depth_for_chrombpnet: true`.

## Job left running

    44588963_0   bias_sweep (03.0, --array=0, fold 0, bias 0.5)   PENDING

Submitted with `--requeue`, `--mem=48G`, and `-C "GPU_CC:7.5|GPU_CC:8.0|GPU_CC:8.6"`.
It is a single-task **pilot**, deliberately: verify chrombpnet runs end to end
before committing 20 GPU jobs.

It is not stuck on anything we control. A trivial 5-minute / 8 GB GPU job with
the same constraint also pended, `sprio` shows real accumulating priority, and
`sacctmgr` shows no limit — the cluster GPU queue was 651 deep on `gpu` and
7,950 on `owners`. Check it with `squeue --me`; the log lands in
`/scratch/users/emattei/hep3b_refactor/logs/bias_sweep_44588963_0.log`.

## Pick up here, in this order

1. **Check the pilot.** If it completed, `models/*_bias.h5` AND
   `evaluation/*_bias_metrics.json` will both exist — the step now requires both
   before it will skip (see below).
2. **Launch the full sweep**: `DATASET=HEP3B sbatch --array=0-19 --requeue
   03.0.train_bias_model.sh`. 5 folds x 4 factors.
3. **`bash 03.1.select_bias.sh`** (no SBATCH header), then read
   `plots/bias_model_selection/HEP3B/` and copy the winners into
   `fold_bias_suffix` in `config/HEP3B/config.yaml`. This hand-off is manual on
   purpose.
4. **`03.2`**, then **`04.0`**. `04.0` still requests 128 G — unlike 03.0 there
   are no MaxRSS records to size it from, because it has never run.

### Environment, if a step fails immediately

Nothing needs exporting any more — `DATASET=HEP3B sbatch <step>` works — but
for the record:

- the `preprocess` env is at
  `/home/groups/engreitz/Users/emattei/.conda/envs/preprocess` (built from
  `envs/preprocess.yml`, Python 3.13). `common.sh` now defaults to it.
- `bootstrap_python()` in `common.sh` finds an interpreter >= 3.9 for the
  config/reference loaders. Sherlock's `/usr/bin/python3` is 3.6.8 and
  `/usr/bin/python` is 2.7.5; `BOOTSTRAP_PYTHON` still overrides.

## Known open, none blocking 03.0

19 findings from the steps-03/04 audit remain unfixed, clustered in
`04.2.qc_combined_boxplot.sh`, `04.3.generate_predictions.sh`, `status.sh` and
two in `common.sh`. The two worth doing before 04 runs:

- **`04.2` dies on a no-data `04.1`.** `qc_full_model.py` writes a 1-byte
  `model_metrics.tsv` when no fold has metrics, and `load_combined_metrics`
  only checks `.exists()` before `read_csv`, so it fails with an uncaught
  parser error. It also declares an output filename it never writes.
- **`common.sh` `dataset_dir`** defaults to `${data_root}/${DATASET}`, which
  with no `DATASET_ROOT` is `$REPO_ROOT/HEP3B` — a path that does not exist.
  It never bit us only because `config/HEP3B/config.yaml` uses absolute paths;
  `status.sh` prints the bogus value as `data:`.

Also outstanding: `shellcheck` has not been run on any of today's changes (it
is not installed on the cluster — `pixi run check-bash` on a laptop).
`pytest` is green: 209 passed, 13 skipped.

## Two things that changed under you

- **Tn5 shift for HEP3B is measured, not assumed: +4/-5**, detected by 00.0 and
  now recorded in the config so later runs skip detection. It matches the
  CellRanger convention these fragments come from.
- **The bias selector now picks `08,08,07,07,08`** for HEP3B, where the July
  run recorded `08,07,08,08,08`. That difference is `219a826`'s three-step
  tie-break, not anything from today — `select_best` was not modified.

## Related

`docs/bias-factor-per-fold.md` is now partially answered, for HEP3B only — see
the analysis at
`nnfc_hep3b/analyses/20260921_bias_factor_per_fold/FINDINGS.md`.
