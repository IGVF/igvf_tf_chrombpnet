# The bias selector walks to the top of whatever range it is given

**Status:** mechanism established on d0 fold 0, 2026-09-22. Not yet acted on.
Bears directly on `docs/bias-factor-per-fold.md`, which observed the symptom
without identifying the cause.

## Symptom

`03.1` selected `_18` — the highest factor in a sweep of 0.55, 0.8, 1.05, 1.3,
1.55, 1.8 — and fired its own `sweep_edge` warning saying the grid might be
mis-centred. The obvious reading is "widen the grid". That reading is wrong.

HEP3B showed the same thing: three of five folds selected `08`, the top of its
swept range, "with the metric still improving".

## Mechanism

The primary score is `peaks_median_norm_jsd - peaks_median_jsd/100`. On d0 it
does NOT max at the winner:

| factor | score | tied? | peaks_r | nonpeaks_r | non-peaks trained on |
|---|---|---|---|---|---|
| 0.55 | 0.03719 | | -0.0040 | +0.2840 | 11,589 |
| 0.80 | 0.04036 | | -0.2858 | -0.1100 | 36,653 |
| 1.05 | 0.04480 | | +0.0034 | +0.1950 | 52,871 |
| 1.30 | 0.04588 | | +0.2505 | +0.3706 | 75,434 |
| 1.55 | **0.04727** | tied | +0.3692 | +0.3915 | 92,235 |
| 1.80 | 0.04685 | tied | +0.3356 | **+0.4132** | 110,749 |

The score peaks at **1.55**. 1.8 is inside the 2% tie window, and then:

1. tie-break 1 (`peaks_r <= nonpeaks_r`) — both healthy, no decision;
2. tie-break 2 (`|peaks_r|` nearest zero) — 0.336 vs 0.369 are within
   `PEAKS_PEARSONR_NEAR_ZERO_EPS` (0.05) of each other, so it abstains BY
   DESIGN;
3. tie-break 3 — **highest `nonpeaks_pearsonr` wins**, and that is 1.8.

`nonpeaks_pearsonr` rises monotonically with the factor (+0.195, +0.371,
+0.392, +0.413) for a reason that has nothing to do with bias: a larger
background is more training data, and more training data fits itself better.
It is an in-domain fit statistic being used as a quality signal.

So whenever the primary metric plateaus inside its tie window, the last
tie-break selects the largest training set available — i.e. the top of the
range. **Widening the grid does not centre the sweep; it moves the ceiling.**
Extending d0 to 3.0 would very likely select 3.0.

This also explains HEP3B without needing a mis-centring story, and it explains
why `docs/bias-factor-per-fold.md` found "the 2% relative tie window is ... 27%
of the entire observed spread — so on this data the tie-break, not the primary
metric, usually decides the winner". It identified that the tie-break decides.
The missing half is that the deciding step has a monotone preference for more
data.

## Why it matters

`peaks_pearsonr` climbs with the factor too: -0.004, +0.003, +0.251, +0.369,
+0.336. A bias model that predicts peak signal is fitting accessibility, not
Tn5 preference, and the bias-factorised subtraction downstream then removes
real biology. Note chrombpnet's own thresholds only guard against peaks_r
being very NEGATIVE (`PEAKS_PEARSONR_WARN = -0.3`), so nothing stops it
drifting positive.

On the merits 1.05 is the standout: `peaks_r = +0.003` with 52,871 non-peaks —
essentially peak-independent, with 4.5x the background of 0.55. But its
primary score is genuinely lower than 1.55's, so preferring it means
overriding the primary metric, not just the tie-break. That is a disagreement
about what the score should reward, and should be settled deliberately.

## Options

1. **Reverse tie-break 3** — among ties prefer the SMALLEST factor. Parsimony:
   the least background achieving the same profile quality. "More data fits
   itself better" is not evidence of a better bias model.
2. **Shrink `PEAKS_PEARSONR_NEAR_ZERO_EPS`** so tie-break 2 decides when
   peaks_r differs by 0.03. Narrower; leaves the monotone bias latent.
3. **An absolute guard on `peaks_r`**, independent of the tie-breaks.

## What is NOT the cause

Ruled out with measurements, so nobody re-treads them:

* **Not a mis-centred grid.** The range was widened from 0.5-0.8 to 0.05-2.0
  and the winner still landed at the top.
* **Not too many peaks.** Against chrombpnet's own reference set
  (`ENCSR868FGK_relaxed_peaks_no_blacklist.bed`), d0 covers LESS of the genome
  in model input windows (9.91% vs 10.60%) and has a near-identical
  negative:peak ratio (1.79 vs 1.75).
* **Not irregular peaks.** The reference set is far more irregular: 150-4182bp
  widths, 40.1% of peaks overlapping another, 1.74x window redundancy, against
  d0's 0% and 1.08x.
