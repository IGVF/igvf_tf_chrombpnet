# Resource measurements

Measured numbers for sizing `#SBATCH --mem`, `--cpus-per-task` and `--time`.

Everything here was measured on the molab box, which is **not** a cluster node:
gVisor, 4 real CPUs, 32 GB RAM, one RTX PRO 6000 Blackwell (sm_120, 96 GB,
PTX-JIT because the container ships no sm_120 cubin). Treat the GPU numbers as
a ceiling on a slower card and the CPU numbers as measured at 4 cores.

The `duration_s` and `peak_rss_gb` fields in `results/metadata/*.json` are the
running record; this file is the interpretation. Query them with DuckDB rather
than re-deriving by hand:

```sql
SELECT step, median(duration_s), max(duration_s) FROM runs GROUP BY step;
SELECT step, max(parameter_value) FROM run_metrics
 WHERE parameter_name = 'peak_rss_gb' GROUP BY step;
```

## 03.0 train_bias_model — the expensive step

One fold x one factor, d0 (153,347 peaks after the 00.1 signal floor).

| factor | train regions | steps/epoch | epochs | epoch time | **total job** |
|-------:|--------------:|------------:|-------:|-----------:|--------------:|
| 0.6    | 119,444       | 1,867       | 9      | 4.6 min    | **10.3 min**  |
| 1.05   | 164,208       | 2,566       | 10     | 6.9 min    | **13.1 min**  |

Phase breakdown for factor 0.6:

| phase | time | GPU |
|---|---:|---|
| data load (one-hot + bigwig counts) | **~4.4 min** | **idle** |
| training epochs | 4.6 min | 79–82% |
| save + predict on test set | ~0.1 min | busy |
| `predict_bias_metrics` (our QC) | 1.2 min | busy |

**Roughly 45% of the wall time is GPU-idle data loading.** Budget `--time` from
total job time, not from epoch time.

### GPU

- **16 ms/step, ~62 it/s, batch 64** — constant across factors. Throughput is a
  property of the architecture, not the factor; the factor changes only the
  number of steps.
- 10,957 MiB VRAM at batch 64. Any modern card fits this; VRAM is not the
  constraint. A 16 GB card is plenty.
- 314 W of a 600 W budget, 79–82% utilisation. Not saturated — see CPUs.

### CPUs — do not request more than 2

The data loader is **single-threaded**. `data_utils.get_seq`, `get_cts` and
`get_coords` are each a `for i, r in peaks_df.iterrows()` loop over pyfaidx /
pyBigWig, and there is no `workers`, `use_multiprocessing` or `Pool` anywhere
under `training/data_generators/`. The 4.4 min load phase is one core spinning.

So extra cores do **not** shorten the load phase, and the 79–82% GPU
utilisation during training is not something more CPUs will fix either — the
generator hands batches from arrays already in RAM. `--cpus-per-task=2` (one
for the loader, one for TF's op threads) is the honest ask. Anything beyond
that is reserved and idle.

Set `NUMEXPR_MAX_THREADS` to match the allocation; the container otherwise
probes the host's core count and warns.

### Memory — scales linearly with region count

Peak host RSS was **8.7 GB** at 131,938 train+valid regions (factor 0.6).

The dominant transient is inside the one-hot encoder, and it is **not** the
one-hot array. `one_hot.dna_to_one_hot` builds the encoding via
`np.unique(base_vals, return_inverse=True)`, and `return_inverse` is **int64** —
8 bytes per base, for every base of every region at once:

| allocation | bytes/region (inputlen 2114) |
|---|---:|
| `base_inds`, int64, transient | 2114 x 8 = 16.9 KB |
| one-hot output, int8 | 2114 x 4 = 8.5 KB |
| counts from `bw.values()`, float64 | 1000 x 8 = 8.0 KB |
| joined sequence string + `base_vals` | 2114 x 2 = 4.2 KB |

That is ~38 KB/region of array plus TF/CUDA overhead (~2 GB) and allocator
slack, which lands near the 8.7 GB observed.

A usable rule, anchored on the one measurement:

```
--mem  ~=  2 GB  +  70 MB per 1000 train+valid regions
```

Worked examples (train+valid, both peaks and non-peaks):

| regions | predicted | note |
|--------:|----------:|---|
| 132,000 | ~11 GB | factor 0.6; 8.7 GB measured, so this has headroom |
| 180,000 | ~15 GB | factor 1.05 |
| 290,000 | ~22 GB | factor 1.5 — near the 32 GB box limit |

**`--mem=24G` covers the factors we sweep; `--mem=32G` is safe for the top of
the range.** Region count is knowable before submission — it is the
`n_after_cutoff` column of `results/plots/signal_qc/<prefix>_bias_threshold_scan.tsv`
— so the sweep can size its own memory per array index instead of using one
worst-case number for every job.

### Disk

| artifact | size | per |
|---|---:|---|
| `evaluation/*_bias_predictions.h5` | **273 MB** | **each factor** |
| `auxiliary/*_data_unstranded.bw` | 192 MB | hardlinked, counted once |
| `auxiliary/*_filtered.bias_{peaks,nonpeaks}.bed` | ~16 MB | each factor |
| `models/` | 2.6 MB | each factor |

The bigwig is a hardlink (`src/chrombpnet_train.py` uses `os.link`, falling
back to `copy2` only across filesystems), so N factors cost one copy, not N.

The prediction `.h5` is the real per-factor cost and is **kept after the
metrics are computed from it**. A 32-factor sweep is ~8.7 GB of predictions;
the 12-factor sweep we would realistically run is ~3.3 GB. Worth deleting once
`*_bias_metrics.json` exists, but that is a policy call — the h5 is the only
way to recompute a metric without retraining.

## Other steps

Measured on d0, from `results/metadata/`. These are CPU-only and cheap; the
defaults are fine and none is worth tuning.

| step | duration | notes |
|---|---:|---|
| 00.0 prepare_signal | see note | fragments -> bigwig |
| 00.1 preprocess_peaks | 24 s | includes the genome-background floor |
| 01.0 preprocess_nonpeaks | 94 s | GC-matched background |
| 02.0 qc_training_data | 88 s | includes the full bias-factor scan |
| 03.1 select_bias | 4 s | reads the sweep, writes a choice |

**Note on 00.0.** The metadata row reads 1.1 s, which is the wrapper only — the
prepared bigwig was reused via `--prepared-bigwig` and the conversion was
skipped. A cold `prepare_signal` that actually builds the bigwig from a
fragments file is minutes, not seconds, and is not yet measured here. Do not
size a cold run from that row.

## Gaps

- One fold, one dataset (d0), one GPU. No cross-node or cross-card comparison.
- `peak_rss_gb` was added to the metadata writer after the runs above, so the
  8.7 GB figure is an external `ps` observation, not a metadata row. Runs from
  here on record it themselves, and the `--mem` rule should be re-fit once
  several steps have reported it.
- 03.2, 03.3, 04.0, 04.1 have never been run; no numbers exist for them. 04.0
  trains a full ChromBPNet model and should be assumed *more* expensive than
  03.0, not comparable to it.
- Cold-start `prepare_signal` unmeasured, as above.
