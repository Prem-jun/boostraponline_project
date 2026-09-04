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
