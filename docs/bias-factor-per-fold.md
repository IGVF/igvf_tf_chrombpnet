# Is the bias threshold factor a per-fold quantity?

**Status:** open. Everything needed to settle it is already on the cluster; the
analysis needs no GPU and no retraining. Written 2026-09-21.

## The question

`03.0.train_bias_model.sh` trains a full grid — every fold × every
`bias_factors` entry, 20 GPU jobs at `--time=2-0` by default — and `03.1`
selects a winner **per fold**, which a human copies into `fold_bias_suffix`.

Is the per-fold part of that grid buying anything?

## Why it might not be

The factor's only job, from chrombpnet's
`helpers/hyperparameters/find_bias_hyperparams.py:81`:

```python
counts_threshold = np.quantile(peak_cnts, 0.01) * args.bias_threshold_factor
```

Non-peak regions above `counts_threshold` are dropped from bias training. The
fold-dependent term — the 1st percentile of that fold's training peak counts —
is **already recomputed per fold**. The factor multiplying it describes the
enzyme and the protocol, not which chromosomes are held out. There is no
mechanism by which fold 2 wants a genuinely different factor from fold 0.

Per-fold bias *models* must stay per-fold regardless: a bias model that saw
fold k's test chromosomes leaks into fold k's evaluation. Only the
hyperparameter is in question.

If the factor is shared, the grid can become two stages — sweep on one fold
(4 jobs), then train the remaining folds at the winner (4 jobs) — 8 GPU jobs
instead of 20, with the leakage guarantees unchanged.

## How to settle it

The inputs already exist: `03.1` writes `all_bias_metrics.tsv` for every
dataset it has ever run on, at
`${results_path}/plots/bias_model_selection/${bias_dataset}/`. Columns are
`bias`, `fold`, `nonpeaks_pearsonr`, `nonpeaks_spearmanr`, `peaks_pearsonr`,
`peaks_spearmanr`, `peaks_mse`, `peaks_median_jsd`, `peaks_median_norm_jsd`
(`src/select_bias_model.py:1153`). Four datasets × 5 folds × 4–6 factors.

Three questions, in order of how decisive they are.

**Q1 — do the folds already agree?** Run `select_best()` per fold and count
distinct winners per dataset. Cheap, and if every dataset is unanimous the
grid is visibly redundant. Not decisive on its own: agreement could be luck.

**Q2 — is "per-fold winner" signal or noise?** For the selection metric
(`norm_jsd_score = peaks_median_norm_jsd - peaks_median_jsd/100`, as
`select_best` computes it), compare the spread **across factors within a fold**
against the spread **across folds at a fixed factor**. If the second is
comparable to the first, the per-fold winner is fold-to-fold measurement noise
being read as a preference.

**Q3 — the counterfactual, and the one that actually decides.** For each
dataset, take the modal winner across folds. For every fold, compute what
using that shared factor instead of the fold's own winner would cost:

- `Δ norm_jsd_score` — compare against `NORM_JSD_SCORE_TIE_REL_EPS` (0.02), the
  tolerance `select_best` already treats as "not meaningfully different"
- whether `classify_row` status changes (pass → warn, warn → fail, …)

This asks the real question — *would we have shipped a different model?* —
rather than a proxy for it.

### Decision rule, fixed in advance

- **Any fold in any dataset changes status** under the shared factor → keep the
  full grid. The cost is real and 12 extra GPU jobs is the right price.
- **All Δ within 2% and no status changes** → switch `03.0` to two stages.
- **Anything in between** (Δ exceeds the tie epsilon but no status flips) →
  keep the grid, and revisit only if GPU time becomes the binding constraint.
  A metric difference we cannot connect to an outcome is not a reason to act.

Q3 is a re-analysis of TSVs already on disk. It should take minutes.

## If it settles "shared"

Split `03.0` into `03.0` (sweep, one fold, `--array=0-$((n_factors-1))`) and a
second stage that trains the remaining folds at the chosen factor. Things to
get right:

- The sweep fold must be configurable, not hardcoded to 0 — if fold 0's test
  chromosomes are atypical for a dataset, that is the one fold you would not
  want to choose on.
- `03.1` still runs, still writes `selected_bias_per_fold.tsv`, and
  `fold_bias_suffix` stays per-fold in the config. The manual hand-off is
  deliberate (see CLAUDE.md) and the escape hatch for a fold that needs
  something different should survive the change.
- The saving is GPU jobs, not wall-clock: stage two cannot start until stage
  one is reviewed, so the pipeline gains a human checkpoint it does not have
  today. That is a cost, and for a dataset being run unattended it may be the
  deciding one.

## Already done, independent of the outcome

`03.1` now flags a fold whose winner sits at an end of the swept range
(`sweep_edge` in `selected_bias_per_fold.tsv`, a warning in the log and a
section in `bias_selection_explanation.txt`). That matters either way:
`bias_factors: ["0.5", "0.6", "0.7", "0.8"]` puts chrombpnet's recommended ATAC
starting value at the **floor** of the sweep, while chrombpnet's own guidance
when a bias model starts regressing TF motifs is to *reduce* the factor
(`helpers/generate_reports/make_html_bias.py:132`). Until now nothing said when
the answer was pinned against the edge of what was tried, and "best of those
tried" and "best there is" look identical in the metrics.
