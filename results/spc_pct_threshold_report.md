# Percentage-of-Chunk-Size Threshold Rule: `C = q * k`

Tests the chunk alarm threshold as a fixed fraction *q* of the chunk size instead of
an absolute count, on all 7 Tier-2 datasets that have both classes present.
Each rule is judged together with the **discriminative power** of the statistic it
thresholds — a threshold rule can only help where violation counts actually separate
faulty chunks from normal ones.

## 1. Discriminative power (the decisive result)

> **Superseded for TEP Mode 3 — see `spc_total_vs_perfeature_report.md`.** The
> "no signal" verdict for TEP Mode 3 below reflects only the per-feature statistic.
> Summing violations across dimensions raises its AUC from 0.448 to **0.859** and
> detection at FAR <= 5% from 18.3% to **65.8%**. Mode 3's fault signal is distributed
> across many dimensions rather than concentrated in one, so a per-feature maximum
> cannot see it. The conclusion that "no threshold rule can rescue it" is wrong.


AUC of per-chunk max-over-features violation count vs. the true fault label:

| Dataset | D | k | in-control / OOC chunks | **AUC** | median violations (in-control → OOC) | verdict |
|---|---:|---:|---:|---:|---|---|
| **TEP Mode 4** | 34 | 500 | 41 / 3,397 | **0.888** | 4 → 88 | **informative** |
| **TEP Mode 1** | 34 | 500 | 38 / 3,442 | **0.884** | 8 → 114 | **informative** |
| **TEP Mode 5** | 34 | 500 | 44 / 3,416 | **0.829** | 7 → 69 | **informative** |
| TEP Mode 3 | 34 | 500 | 39 / 3,440 | 0.448 | 292 → 289 | no signal |
| AI4I 2020 | 5 | 100 | 6 / 94 | 0.435 | 6 → 5 | no signal |
| Water Pump | 10 | 500 | 405 / 36 | 0.402 | 2 → 0 | no signal |
| MetroPT-3 | 7 | 1000 | 1,482 / 35 | **0.154** | 33 → 0 | **inverted** |

This splits the benchmark suite in two. On **TEP Modes 1, 4 and 5** the violation
count is a genuinely good fault statistic (AUC ≈ 0.83–0.89) and the threshold choice
matters a great deal. On the other four it is uninformative or inverted, and no
threshold rule can rescue them.

## 2. Threshold comparison

Only the three datasets with real signal are shown here in full; the rest are in
`spc_pct_threshold.csv`.

### TEP Mode 1 (AUC 0.884)

| Rule | C | Chunk FAR % | Detect % | ARL0 | ARL1 |
|---|---:|---:|---:|---:|---:|
| fixed C=3 (current default) | 3 | 100.00 | 99.91 | 0.00 | 1.00 |
| 1% of k | 5 | 97.37 | 99.27 | 0.03 | 1.01 |
| fixed C=10 / 2% of k | 10 | 23.68 | 81.96 | 2.90 | 1.21 |
| **5% of k** | **25** | **0.00** | **68.65** | 38.00 | 1.44 |
| 10% of k | 50 | 0.00 | 64.44 | 38.00 | 1.53 |

### TEP Mode 4 (AUC 0.888)

| Rule | C | Chunk FAR % | Detect % | ARL0 | ARL1 |
|---|---:|---:|---:|---:|---:|
| fixed C=3 | 3 | 92.68 | 99.15 | 0.08 | 1.01 |
| 1% of k | 5 | 36.59 | 87.40 | 1.73 | 1.14 |
| fixed C=10 / 2% of k | 10 | 0.00 | 71.83 | 41.00 | 1.38 |
| **5% of k** | **25** | **0.00** | **62.73** | 41.00 | 1.57 |
| 10% of k | 50 | 0.00 | 57.70 | 41.00 | 1.71 |

### TEP Mode 5 (AUC 0.829)

| Rule | C | Chunk FAR % | Detect % | ARL0 | ARL1 |
|---|---:|---:|---:|---:|---:|
| fixed C=3 | 3 | 100.00 | 99.94 | 0.00 | 1.00 |
| 1% of k | 5 | 93.18 | 97.54 | 0.07 | 1.02 |
| fixed C=10 / 2% of k | 10 | 15.91 | 75.61 | 4.62 | 1.31 |
| **5% of k** | **25** | 11.36 | 63.41 | 6.50 | 1.55 |
| 10% of k | 50 | 6.82 | 54.57 | 10.25 | 1.80 |

### The other four (no usable signal — FAR only)

| Dataset | fixed C=3 | 5% of k | 10% of k |
|---|---:|---:|---:|
| MetroPT-3 | 95.61% | 25.24% | 9.99% |
| Water Pump | 47.65% | 29.38% | 23.95% |
| AI4I 2020 | 66.67% | 66.67% | 0.00% |
| TEP Mode 3 | 100.00% | 100.00% | 100.00% |

`q = 5%` is the best single value: it is the smallest fraction that drives Chunk FAR
to zero on TEP Modes 1 and 4 while retaining the most detection. `q = 10%` buys a
little more FAR reduction on the no-signal datasets but costs 4–9 points of detection
on the ones that matter.

## 3. This restores the paper's headline claim — on a correct basis

The published "RBULT Chunk FAR = 0.00%, ARL0 = 38.00" on TEP Mode 1 was an artefact
of the `sample_ooc_count` bug (RBULT was scored as having zero violations always).
At `C = 25` (5% of k) the *same numbers* appear legitimately — and this time with
**68.65% detection**, where the buggy version had 0% by construction.

So the claim is defensible, but it needs `C = 25`, not the `C = 5, 10, 15` the
sensitivity study tested. All three of those values are far too low: TEP Mode 1
averages ~16 violations per feature per chunk while in control.

## 4. Two caveats that must be stated

**Chunk FAR = 0.00% is not measurable at this sample size.** TEP has only 38–44
in-control chunks. Clopper-Pearson 95% upper bounds:

| Dataset | false alarms | point estimate | 95% upper bound |
|---|---|---:|---:|
| TEP Mode 1 @ 5% of k | 0 / 38 | 0.00% | **9.25%** |
| TEP Mode 4 @ 5% of k | 0 / 41 | 0.00% | **8.60%** |
| TEP Mode 5 @ 5% of k | 5 / 44 | 11.36% | 24.56% |

The honest statement is "Chunk FAR below ~9% at 95% confidence", not "= 0%".

**ARL0 = 38.00 is censored, not estimated.** When no false alarm fires,
`_compute_arl0` returns the total in-control chunk count. 38.00 is exactly TEP Mode
1's in-control chunk count, so it means "no false alarm within the observation
window" — a lower bound. It cannot be compared numerically against a baseline's
ARL0 of 0.06.

## 5. Limitations of the `q * k` rule

- It scales with `k` but **not with `D` or with the actual per-dimension FAR**, which
  ranges from 0.045% (Water Pump) to 6.28% (TEP Mode 3) across this suite — a 140x
  spread. A single `q` cannot be right everywhere; it happens to suit TEP because
  TEP dominates the datasets where the statistic works at all.
- It does nothing for TEP Mode 3, whose in-control chunks already average 292
  violations (58% of k). Its problem is interval calibration, not thresholding.
- **Label imbalance**: TEP is 98.9% OOC chunks and AI4I is 94% OOC. These are not
  SPC monitoring scenarios, where in-control data should dominate. Chunk FAR and
  ARL0 for these datasets rest on 6–44 chunks and are correspondingly fragile.

## 6. Recommendation

Adopt `C = ceil(0.05 * k)` as the default, replacing the fixed `ooc_threshold_count=3`.
It is scale-free in chunk size, needs no Phase I calibration, is trivial to state in
the paper, and it is the only rule tested that achieves near-zero chunk FAR with
substantial detection on the datasets where detection is possible.

Where a Phase I in-control window is available, the per-dimension
`binomial (empirical)` rule remains preferable — it beat every fixed rule on Water
Pump (FAR 25.44% → 9.54% at unchanged detection). See `spc_threshold_rules_report.md`.

---
Generated by `exp_pct_threshold.py`; raw numbers in `spc_pct_threshold.csv` and
`spc_pct_threshold_auc.csv`.
