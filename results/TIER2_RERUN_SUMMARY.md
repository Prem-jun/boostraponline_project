# Tier 2 Re-run — Final Summary

Everything below was produced after two code changes:

1. **Chunk alarm threshold** is now the scale-free rate rule `C = ceil(0.05 * k)`,
   replacing the fixed `ooc_threshold_count = 3`. An optional per-dimension Phase I
   binomial calibration is also available (`start_phase1()` / `calibrate_phase1()`).
2. **Baselines now use the same per-feature chunk alarm rule as RBULT.** Previously
   the baselines summed violations across all D features while RBULT evaluated each
   feature separately — so at the same C the baselines faced a far looser condition.

Both changes apply identically to all four methods, so the comparison is now like-for-like.

## Resolved thresholds

| Dataset | k | C = ceil(0.05k) |
|---|---:|---:|
| AI4I 2020 | 100 | 5 |
| Industrial Pump | 200 | 10 |
| Water Pump | 500 | 25 |
| TEP Mode 1/3/4/5 | 500 | 25 |
| MetroPT-3 | 1000 | 50 |

## RBULT-SPC vs. the Bootstrap baseline (like-for-like)

| Dataset | Coverage RBULT / Boot | Chunk FAR RBULT / Boot | ARL0 RBULT / Boot | RAM RBULT / Boot |
|---|---|---|---|---|
| AI4I 2020 | 98.40 / **98.81** | 66.67 / **0.00** | 0.50 / **6.00** | **0.52** / 413.78 KB |
| MetroPT-3 | **98.90** / 98.76 | 25.24 / **8.23** | 2.95 / **11.06** | **0.70** / 90,932.70 KB |
| Industrial Pump * | **99.40** / 98.96 | 0.00 / 0.00 | 0.00 / 0.00 | **0.52** / 826.91 KB |
| Water Pump | **99.95** / 98.63 | **29.38** / 32.59 | **2.38** / 2.05 | **0.98** / 17,667.15 KB |
| TEP Mode 1 | 96.74 / **99.02** | 0.00 / 0.00 | 38.00 / 38.00 | **3.23** / 582.87 KB |
| TEP Mode 3 | 93.73 / **99.01** | 100.00 / **0.00** | 0.00 / **39.00** | **3.23** / 582.87 KB |
| TEP Mode 4 | 96.67 / **99.01** | 0.00 / 0.00 | 41.00 / 41.00 | **3.23** / 582.87 KB |
| TEP Mode 5 | 97.79 / **99.05** | 11.36 / **0.00** | 6.50 / **44.00** | **3.23** / 582.87 KB |

\* Industrial Pump has **zero** in-control chunks, so its Chunk FAR and ARL0 are
undefined — the 0.00 values come from `max(1, in_control_chunks)` and the
`float(in_control_chunks)` fallback, not from a measurement.

## What survived, and what did not

**Holds up — memory.** RBULT is 180x to 130,000x smaller than the full-history /
sliding-window bootstrap, exactly O(D), reproduced identically on every run
throughout this whole exercise: 0.52 / 0.70 / 0.98 / 3.23 KB at D = 5 / 7 / 10 / 34.
This is the paper's strongest and most robust claim.

**Holds up — coverage on low/medium-dimensional data.** RBULT beats the bootstrap
baseline on MetroPT-3, Industrial Pump and Water Pump. Note this is marginal coverage measured
against the final bounds; see the interval-width section below for what it costs.

**Does not hold — coverage on TEP.** The sliding-window bootstrap reaches ~99.0% on
all four modes; RBULT reaches 93.7–97.8%.

**Does not hold — chunk-level false alarm control.** The claim in
`paper1_experiment_summary.md` that "baselines have Chunk FAR 92–100% while RBULT is
0.00%" does not survive a like-for-like comparison. Under the shared per-feature rule
the bootstrap baseline wins or ties on 6 of the 7 datasets where the metric is
measurable; RBULT wins only on Water Pump (29.38% vs 32.59%).

> **Correction (see `spc_total_vs_perfeature_report.md`).** The Chunk FAR column above
> must be read together with a detection count, which these tables do not carry. Under
> the per-feature rule the sliding-window bootstrap **never raises a single alarm on any
> TEP mode** — its 0.00% Chunk FAR, its ARL0 of 38/39/41/44 and its ARL1 of 1.00 all mean
> "never fired", not "fired accurately". So it does not beat RBULT on TEP. On Mode 1
> RBULT reaches 0.00% Chunk FAR *with 68.65% detection*, Shewhart 0.00% *with 79.92%*,
> and the bootstrap 0.00% with nothing. The honest summary is that RBULT is competitive
> with Shewhart on TEP Modes 1/4/5 and loses on Mode 3, not that the bootstrap dominates.

**Does not hold — the TEP sensitivity claim.** The published "RBULT Chunk FAR = 0.00%
at every threshold, ARL0 = 38.00" came from `exp_tep_sensitivity.py:56` reading
`summary['sample_ooc_count']`, a key `update_chunk` never returned, so RBULT was
scored as having zero violations in every chunk. (38.00 is exactly TEP Mode 1's
in-control chunk count — what `_compute_arl0` returns when no alarm ever fires.) That
key now exists; corrected, RBULT shows real threshold sensitivity:
100% → 100% → 94.74% → 34.21% → 0.00% at C = 5/10/15/25/50.

**New — joint coverage.** The metric added to `compute_spc_metrics` shows marginal
coverage is a much weaker statement than it appears. Six of eight datasets fall below
the 95% Bonferroni target:

| Dataset | D | Marginal | Joint |
|---|---:|---:|---:|
| Water Pump | 10 | 99.95% | 99.55% |
| Industrial Pump | 5 | 99.40% | 97.04% |
| MetroPT-3 | 7 | 98.90% | 94.89% |
| AI4I 2020 | 5 | 98.40% | 94.23% |
| TEP Mode 5 | 34 | 97.79% | 70.50% |
| TEP Mode 4 | 34 | 96.67% | 65.26% |
| TEP Mode 1 | 34 | 96.74% | 61.41% |
| TEP Mode 3 | 34 | 93.73% | 24.97% |

On TEP, RBULT flags 30–75% of observations as out-of-control on a joint basis. The
per-dimension false alarm rate there is 15–43x the Bonferroni target, which is the
root cause of both the joint-coverage collapse and the chunk-level behaviour.

## Undefined metrics now report NaN instead of a flattering zero

Three fallbacks in the metric code returned a *good-looking number* where the quantity was
undefined. All three are fixed, in the library and in all six experiment scripts.

| Situation | Old value | New value |
|---|---|---|
| Chunk FAR with no in-control chunks (0/0) | `0.00` via `max(1, in_control_chunks)` | `NaN` |
| ARL0 with no in-control chunks | `0.00` | `NaN` |
| ARL1 when nothing was ever detected | `1.00` — identical to instant detection, the best possible score | `NaN`, plus a new `n_detected_episodes` field |

ARL0 remains the in-control chunk count when no false alarm fires, since that is a genuine
right-censored lower bound; a new `arl_0_censored` flag marks those cells.

**What this exposed, now visible in the shipped result files rather than only in ad-hoc
analysis:**

- **Industrial Pump** — all four methods report NaN for Chunk FAR, ARL0 and ARL1. The dataset
  has zero in-control chunks, so the previous `0.00%` across the board was arithmetic, not
  measurement, and read as flawless in-control behaviour.
- **The sliding-window bootstrap baseline never detects anything on TEP.** `arl_1` is NaN on
  all eight TEP runs (four modes, raw and differenced). Its published `ARL1 = 1.00` was the
  fallback throughout. Combined with its 0.00% Chunk FAR and ARL0 of 100, the honest reading
  is that this baseline simply never fires an alarm on TEP — it is not a strong competitor
  there, and any comparison against it on chunk-level metrics is vacuous.

### Industrial Pump: the label is a coin flip

`Maintenance_Flag` carries no information. Three independent checks agree:

| Check | Result |
|---|---|
| AUC of each of the 6 variables predicting the flag | 0.4925 – 0.5050, **every p > 0.05** |
| Flag rate per pump (5 pumps) | 48.6% – 50.6%, uniform |
| Mean contiguous run length | **1.99 rows**, exactly the 1.99 implied by independent Bernoulli(0.4984) draws |

No redefinition of the chunk label, grouping by `Pump_ID`, or threshold choice can recover a
chunk-level metric from a random label. `exp_pump_benchmark.py` now runs a label-quality gate
that prints this diagnosis at run time and renders the affected cells as `—` with an
"UNDEFINED — no in-control chunks" caption, rather than leaving `nan%` beside a caption
reading "Low Batch False Alarm Rate".

Sample-level metrics — coverage, sample FAR, interval width, RAM, latency — do not depend on
the label and remain valid. The dataset stays useful precisely because it is i.i.d.: it is the
reference point where `width_ratio_local` equals 1.00, against which the inflated intervals on
the autocorrelated streams are measured.

## TEP: chunk size aligned to the simulation run, and within-run differencing

**Chunk size 500 -> 600.** The TEP arrays are (runs x 600 steps x 34 vars) flattened into one
stream, so any k not dividing 600 makes every chunk straddle a run boundary — a discontinuity
of the data layout, not of the process. Alignment also fixes the labels: with only 100 of
~2,900 runs normal, and each normal run spanning 1.2 chunks of 500, normal runs were routinely
labelled faulty by a neighbour sharing their chunk. **In-control chunks rise from 38-44 to 100
in every mode**, so the Clopper-Pearson upper bound on an observed 0% Chunk FAR tightens from
~9.25% to ~3.0%. C = ceil(0.05 * 600) = 30.

Coverage *fell* under the corrected chunking (Mode 1 96.74 -> 93.71, Mode 4 96.67 -> 90.64).
The cause is not the run alignment itself: at fixed N and fixed data, the final interval width
varies with k in a way that is **not monotonic** — on a 300-run subset of Mode 1, k = 100 / 150
/ 300 / 600 / 1200 gives final widths of 62.98 / 31.22 / 34.32 / 36.10 / 45.37 and coverage
98.17 / 96.64 / 97.53 / 93.59 / 95.14. Repeating each setting four times gives a spread of
0.0 in width and 0.01 points in coverage, so this is a real property of the lazy expansion
mechanism, not RNG noise. **The interval estimate is chunk-size dependent**, and that should
be stated as a limitation: k is a free experimental parameter that moves the headline number.

**Within-run differencing.** Differencing is applied inside each run and never across runs —
the error that made AI4I's `Tool wear Rate` an artefact. Each run loses its first sample
(600 -> 599). Applied to the dataframe so all four methods receive identical input;
`RBULTControlChart(difference=True)` performs the same transform in streaming O(D) form
(one scalar of state per feature) and is verified to reproduce `np.diff` exactly in both the
continuous and the `new_sequence` mode.

| Mode | Coverage raw -> diff | Joint raw -> diff | Chunk FAR raw -> diff | ARL0 raw -> diff | width_ratio_global |
|---|---|---|---|---|---|
| 1 | 93.71 -> **98.13** | 48.83 -> **66.17** | 0.00 -> 0.00 | 100.0 -> 100.0 | 0.51 -> **0.85** |
| 3 | 91.76 -> **97.20** | 17.79 -> **59.10** | **100.00 -> 0.00** | **0.00 -> 100.0** | 0.52 -> **0.78** |
| 4 | 90.64 -> **96.09** | 50.61 -> **55.23** | 1.00 -> 0.00 | 49.5 -> **100.0** | 0.40 -> **0.65** |
| 5 | 92.39 -> **97.37** | 45.15 -> **61.63** | **37.00 -> 0.00** | 1.7 -> **100.0** | 0.46 -> **0.73** |

Every mode improves on every metric. **Mode 3 is transformed**: it had been the worst case
throughout this exercise — Chunk FAR pinned at 100% for every threshold, joint coverage 17.8%,
ARL0 zero — and differencing takes it to 0.00% Chunk FAR with 59.1% joint coverage. The
earlier conclusion that Mode 3 "has no signal" was wrong twice over: the signal is there, it
is simply distributed across dimensions (see `spc_total_vs_perfeature_report.md`) *and*
masked by the level at which each run sits.

Two secondary findings:

- **The in-sample / prequential gap closes to zero** on all four modes (0.25-0.69 points ->
  0.00). This independently confirms the mechanism identified earlier: the gap comes from
  boundaries inflating to absorb non-stationarity, so removing the non-stationarity removes
  the gap.
- **`width_ratio_global` rises to 0.65-0.85** from 0.40-0.52 — the interval moves closer to
  the empirical support instead of covering only half of it, which is what drove the
  per-dimension FAR above the Bonferroni target on raw TEP.

Against this, AUC of the per-feature violation statistic falls on Modes 1 and 4
(0.849 -> 0.791, 0.863 -> 0.780): differencing discards level information, so a fault that
holds a shifted level becomes a single spike at the transition. The sum-over-dimensions
statistic is far less affected (0.863 -> 0.844, 0.870 -> 0.853). Both raw and differenced
results are reported rather than replacing one with the other.

## One-step-ahead (prequential) coverage — reported alongside the in-sample figure

`compute_spc_metrics()` scores coverage **in-sample**: it applies the FINAL limits
retrospectively to the whole stream. Since RBULT limits only widen, the final interval is the
widest the chart ever held, and early observations are judged by limits fitted to data that
had not yet arrived. The per-chunk violation counts are in-sample too — a chunk widens the
limits first, then is measured against the widened ones.

A deployed control chart cannot work that way: limits must exist before the data they judge.
`compute_prequential_metrics()` now scores every chunk against the limits carried in from
chunks 1..m-1, before that chunk is allowed to update them (each dimension's first chunk is
excluded from the denominator). Tier 1 has always reported both *In-Sample Adaptation* and
*One-Step-Ahead Pre-Sequential*; Tier 2 now gives the same pair.

| Dataset | `width_ratio_local` | Coverage in-sample | **one-step-ahead** | gap | Joint in-sample | **joint one-step-ahead** | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Industrial Pump | **1.00** | 99.417% | 99.411% | **0.006** | 97.10% | 97.07% | **0.03** |
| TEP Mode 3 | 1.77 | 93.705% | 93.525% | 0.180 | 25.17% | 24.42% | 0.76 |
| TEP Mode 1 | 2.07 | 96.736% | 96.540% | 0.196 | 61.40% | 60.71% | 0.69 |
| TEP Mode 4 | 2.67 | 96.670% | 95.918% | 0.753 | 65.27% | 63.36% | 1.91 |
| MetroPT-3 | 1.55 | 98.895% | 98.109% | 0.787 | 94.89% | 91.30% | 3.58 |
| TEP Mode 5 | 8.19 | 97.789% | 96.865% | 0.924 | 70.45% | 65.80% | 4.66 |
| Water Pump | **8.51** | **99.955%** | 97.808% | 2.146 | **99.55%** | **87.68%** | **11.87** |
| AI4I 2020 | 4.55 | 97.792% | 95.495% | 2.297 | 91.34% | 82.47% | 8.87 |

**The gap is a direct function of how much the interval had to inflate.** Correlation between
`width_ratio_local` and the marginal gap is **0.686**, and with the joint gap **0.782**.
Industrial Pump, whose rows are i.i.d. and whose interval already matches local variation
(ratio 1.00), loses **0.006 points** — the two protocols are indistinguishable there. Water
Pump, whose interval is 8.5x wider than local variation, loses **11.87 points of joint
coverage** (99.55% → 87.68%). The same mechanism drives both quantities: an interval that
widens to absorb non-stationarity is one that was fitted using data the forecaster had not
yet seen.

Validated on synthetic data before deployment: a stationary stream gives 99.22% in-sample
against 99.21% prequential, while a drifting one gives 99.93% against **71.23%** — the gap
appears only where there is drift to exploit, as theory requires.

Both figures are now reported. The in-sample values remain valid for what they measure
(how well the final interval covers the observed history); the prequential values are the
ones that describe deployment behaviour.

## Interval width — reported alongside coverage from this run onward

Coverage cannot be interpreted alone: a wide enough interval attains 100% coverage while
carrying no information, and RBULT boundaries expand monotonically, so they inflate to absorb
non-stationarity. Tier 1 always reported `Mean_Interval_Width` and `Sigma_L`/`Sigma_R`; Tier 2
now does too, accumulated with Welford so the chart still holds only O(D) state (9 scalars
per feature, independent of stream length).

| Dataset | Lag-1 AC | Coverage | Joint | **width_ratio_local** | width_ratio_global |
|---|---:|---:|---:|---:|---:|
| Water Pump | 0.998 | **99.95%** | 99.55% | **8.51** (max 19.4) | 1.11 |
| TEP Mode 5 | 0.948 | 97.79% | 70.49% | **8.19** (max **157.6**) | 0.79 |
| AI4I 2020 | — | 97.79% | 91.34% | 4.55 | 0.94 |
| TEP Mode 4 | 0.948 | 96.66% | 65.26% | 2.67 | 0.74 |
| TEP Mode 1 | 0.948 | 96.74% | 61.39% | 2.08 | 0.69 |
| TEP Mode 3 | 0.948 | 93.70% | 25.02% | 1.77 | 0.65 |
| MetroPT-3 | 0.970 | 98.90% | 94.89% | 1.55 | 1.40 |
| Industrial Pump | 0.001 | 99.40% | 97.04% | **1.00** | 1.01 |

`width_ratio_local` is the final width divided by the mean within-chunk data range;
`width_ratio_global` divides it by the stream's 0.5–99.5 percentile span.

Two consequences:

- **The best coverage carries the widest interval.** Water Pump's 99.955% comes with an
  interval 8.5x wider than the data's own within-chunk variation (19x on one channel).
  Industrial Pump, whose rows are i.i.d., sits at exactly 1.00. The ordering of
  `width_ratio_local` follows autocorrelation, not estimator quality — and it is the same
  reason detection collapses on those datasets (Water Pump AUC 0.402, median violations 0 in
  both classes).
- **On TEP the interval is narrower than the empirical support** (`width_ratio_global`
  0.65–0.79), which is the direct cause of the per-dimension FAR running 15–43x above the
  Bonferroni target and of the joint-coverage collapse to 25–70%.

RBULT-SPC genuinely does not require stationary preprocessing, where Shewhart and EWMA
mean-level models break (coverage 25–77%). But the coverage that follows is obtained partly
by widening, so it must be reported with `width_ratio_local`. With `width_ratio_global` ~ 1
off TEP, the defensible claim is an interval *equivalent* to a full-history percentile
baseline at O(D) memory instead of O(N*D) — not a better interval.

## Recommended framing

The defensible claim is: **RBULT attains coverage comparable to a full-history
bootstrap while using O(D) memory independent of N.** The false-alarm-superiority
claim should be withdrawn.

## Open issues not addressed

- **`exp_tep_sensitivity.py` still uses total-across-features semantics** for all four
  methods, while the main benchmarks now use per-feature. Internally consistent, but
  not comparable to the main tables — the two disagree at C=25 on TEP Mode 1
  (34.21% vs 0.00%). Pick one convention for the paper.
- **`_compute_arl1` returns 1.0 when nothing is ever detected**, indistinguishable
  from "detected immediately". Always read ARL1 alongside a detection count.
- **ARL0 is censored when no false alarm fires** — it returns the in-control chunk
  count (38.00, 41.00, 44.00 on TEP), a lower bound, not an estimate.
- **Chunk FAR on TEP rests on 38–44 in-control chunks.** An observed 0/38 supports
  only "below ~9% at 95% confidence" (Clopper-Pearson), not "= 0%".
- **Label imbalance**: TEP is 98.9% OOC chunks, AI4I 94%, Industrial Pump 100%. These
  are not SPC monitoring scenarios, where in-control data should dominate.
- **The ~156,000x TEP memory claim** in `paper1_experiment_summary.md` rests on a
  504,425.95 KB full-history baseline that `exp_tep_benchmark.py` no longer
  implements — it now uses a sliding window (582.87 KB), giving ~180x.
- **RNG is unseeded** (`np.random.choice` in `bootstrap_online.py:239`), so RBULT
  varies run to run. Measured spread is +/-0.01 pp, harmless, but it prevents exact
  reproduction.

---
Raw numbers: `tier2_final_all_methods.csv`, plus the per-dataset
`spc_*_benchmark_results.csv` and `spc_*_benchmark_comparison.md`.
Earlier investigations: `spc_cthresh_sweep_report.md`, `spc_threshold_rules_report.md`,
`spc_pct_threshold_report.md` — note these were produced before the baseline fix, so
their baseline numbers use the old total-across-features rule.
