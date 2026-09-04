# Chunk-Alarm Threshold Rules: Phase I / Phase II Evaluation

Tested on the only two Tier-2 datasets with enough in-control chunks to measure
chunk-level false alarm rate meaningfully (MetroPT-3: 1,482; Water Pump: 405).
Industrial Pump has **zero** in-control chunks and AI4I has only 6, so neither can
support this comparison.

## Protocol

| Stage | Chunks | Purpose |
|---|---|---|
| Warm-up | first 10% | **discarded** — RBULT bounds are still expanding, FAR is transient |
| Phase I | 10%–30% | calibrate thresholds on **in-control chunks only** |
| Phase II | 30%–100% | all reported metrics |

Target: `alpha_chunk = 0.05` system-wide false-alarm probability per chunk.
A chunk alarms when **any** feature *d* has `V_d >= C_d`. The last three rules are
per-dimension (each feature gets its own `C_d`).

Warm-up matters: MetroPT-3's online FAR runs 2.55% and 5.72% in the first two
deciles before settling near 1.2%; Water Pump's is 9.12% during warm-up vs 1.86%
in Phase I. Calibrating without discarding it inflates `p_hat` roughly 3x.

## Rules compared

| Rule | Definition |
|---|---|
| `fixed C=3` | current default in every experiment script |
| `fixed C=10` | best fixed value from the C_thresh sweep |
| `binomial (nominal)` | `Binom.ppf(1 - alpha_chunk/D, k, alpha_dim) + 1` — theory-driven |
| `binomial (empirical)` | same, with `p_hat_d` estimated from Phase I |
| `binomial (effective n)` | as above, `k` deflated by observed overdispersion `phi = Var_obs / Var_binom` |
| `empirical quantile` | `quantile_{1-alpha_chunk/D}(V_d over Phase I) + 1` — distribution-free |

## Phase II results

### MetroPT-3 (D=7, k=1000; 1,027 in-control / 35 OOC chunks)

| Rule | C (min/med/max) | Chunk FAR % | Detect % | ARL0 | ARL1 |
|---|---|---:|---:|---:|---:|
| fixed C=3 | 3 / 3 / 3 | 94.55 | 22.86 | 0.06 | 3.12 |
| fixed C=10 | 10 / 10 / 10 | 94.06 | 22.86 | 0.06 | 3.12 |
| binomial (nominal) | 15 / 15 / 15 | 93.67 | 22.86 | 0.07 | 3.12 |
| binomial (empirical) | 7 / 34 / 123 | 22.40 | 14.29 | 3.45 | 4.20 |
| binomial (effective n) | 4 / 8 / 27 | 93.28 | 20.00 | 0.07 | 3.14 |
| empirical quantile | 13 / 126 / 698 | **2.63** | 8.57 | 35.71 | 7.00 |

### Water Pump (D=10, k=500; 283 in-control / 26 OOC chunks)

| Rule | C (min/med/max) | Chunk FAR % | Detect % | ARL0 | ARL1 |
|---|---|---:|---:|---:|---:|
| fixed C=3 | 3 / 3 / 3 | 25.44 | 23.08 | 2.89 | 2.33 |
| fixed C=10 | 10 / 10 / 10 | 11.31 | 19.23 | 7.61 | 2.80 |
| binomial (nominal) | 8 / 8 / 8 | 14.49 | 23.08 | 5.76 | 2.33 |
| **binomial (empirical)** | 1 / 2 / 115 | **9.54** | **23.08** | 9.14 | 2.50 |
| binomial (effective n) | 1 / 2 / 14 | 12.01 | 23.08 | 7.11 | 2.50 |
| empirical quantile | 1 / 2 / 226 | 5.30 | 11.54 | 16.75 | 5.00 |

On Water Pump `binomial (empirical)` dominates every fixed rule: it cuts Chunk FAR
from 25.44% to 9.54% (2.7x) while holding detection at 23.08%, the same as the most
permissive fixed threshold. That is the one clear win in this study.

On MetroPT-3 no rule reaches the 5% target while keeping detection: every reduction
in Chunk FAR costs proportionally more detection.

## Why MetroPT-3 cannot be fixed by any threshold

Ranking chunks by violation count against the true fault label (Phase II):

| Statistic | MetroPT-3 AUC | Water Pump AUC |
|---|---:|---:|
| max over features | **0.170** | 0.513 |
| sum over features | **0.166** | 0.512 |
| number of features violating | **0.127** | 0.495 |

Median max-violations per chunk on MetroPT-3: **29 when in-control, 0 during faults.**

AUC 0.17 is not weak signal — it is *inverted* signal. RBULT's chunk alarm fires
during normal operation and goes quiet during the labelled failures. Water Pump's
AUC of 0.51 is indistinguishable from random: its bounds are wide enough
(99.955% coverage) that neither class produces violations.

RBULT bounds only ever expand, so violation counts fall over time by construction —
which could manufacture this anti-correlation if faults sit late in the stream. A
time-matched test rules that out. Each OOC chunk was compared only against
in-control chunks within +/-30 chunks:

| Dataset | OOC violation count higher than local normal | Null |
|---|---:|---:|
| MetroPT-3 | **14.3%** | 50% |
| Water Pump | 47.2% | 50% |

The inversion survives time-matching, so it is a property of the detector, not a
trend artefact. (The time trend is real and large — Water Pump's in-control median
violations go from 41 in the first half to 0 in the second — but it is not the cause.)

## Conclusion

The threshold question is not answerable on these two datasets, because the
statistic being thresholded carries no usable signal:

- **Water Pump** — no discriminative power (AUC 0.51). The 23% "detection" is chance.
  A better threshold still gives a better-calibrated *false alarm rate*, which is why
  `binomial (empirical)` is a genuine improvement there, but it cannot buy detection.
- **MetroPT-3** — inverted discriminative power (AUC 0.17). No threshold on this
  statistic can work; the detector would have to be reversed to be useful.

This does not touch the paper's primary claims, which are about **coverage** and
**O(D) memory** — those reproduce exactly. It affects the secondary chunk-level
detection metrics (Chunk FAR, ARL0, ARL1). The published ARL1 values (3.12 for
MetroPT-3, 2.40 for Water Pump) read as fast detection but are produced by an alarm
firing on ~95% and ~25% of in-control chunks respectively; with an alarm that
frequent, any fault chunk is "detected" quickly by coincidence.

## Metric caveat found while running this

`_compute_arl1` returns `1.0` when nothing is ever detected, which is identical to
the value for "detected immediately." In the `empirical quantile` row for Water Pump
in an earlier run, ARL1 read 1.00 alongside 0.00% detection. Any ARL1 near 1.0 must
be read together with a detection count. This affects the published TEP tables,
where ARL1 = 1.00 appears throughout.

---
Generated from `exp_threshold_rules.py`; raw numbers in `spc_threshold_rules.csv`.
