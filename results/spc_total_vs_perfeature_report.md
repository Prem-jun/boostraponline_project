# PER-FEATURE vs TOTAL Chunk Alarm Semantics

Two ways to turn per-feature violation counts `V_d` into a chunk-level alarm, both
evaluated at the same `C = ceil(0.05 * k)` and applied to all four methods:

- **PER-FEATURE** (what `RBULTControlChart` does): alarm if **any single** feature has `V_d >= C`
- **TOTAL**: alarm if `sum_d V_d >= C`

## 1. A detector that never fires looks perfect on Chunk FAR

The single most important thing this table shows is not about RBULT. Under
PER-FEATURE at C=25, the **sliding-window bootstrap baseline never raises a single
alarm on any TEP mode**:

| TEP mode | Bootstrap(slide) Chunk FAR | Bootstrap(slide) **detection** |
|---|---:|---:|
| Mode 1 | 0.00% | **0.00%** |
| Mode 3 | 0.00% | **0.00%** |
| Mode 4 | 0.00% | **0.00%** |
| Mode 5 | 0.00% | **0.00%** |

Its "Chunk FAR = 0.00%" and "ARL0 = 38/39/41/44" mean *it never alarms*, not *it
alarms accurately*. Its ARL1 of 1.00 in the benchmark tables is the
`_compute_arl1` fallback for "nothing was ever detected". **Chunk FAR, ARL0 and ARL1
must never be read without a detection count beside them.**

This corrects a claim made earlier in `TIER2_RERUN_SUMMARY.md`: the sliding-window
bootstrap does not "beat" RBULT on TEP. On Mode 1, RBULT achieves 0.00% Chunk FAR
*with 68.65% detection*; Shewhart achieves 0.00% *with 79.92%*; the bootstrap
achieves 0.00% with nothing.

## 2. Chunk FAR / detection at C = ceil(0.05k)

| Dataset | k | D | C | Method | PER-FEATURE FAR / det | TOTAL FAR / det |
|---|---:|---:|---:|---|---|---|
| AI4I 2020 | 100 | 5 | 5 | RBULT | 66.67 / 52.13 | 66.67 / 90.43 |
| | | | | Bootstrap(full) | 0.00 / 21.28 | 0.00 / 35.11 |
| Water Pump | 500 | 10 | 25 | RBULT | 29.38 / 16.67 | 29.88 / 16.67 |
| | | | | Bootstrap(full) | 32.59 / 72.22 | 43.70 / 72.22 |
| MetroPT-3 | 1000 | 7 | 50 | RBULT | 25.24 / 11.43 | 86.50 / 14.29 |
| | | | | Bootstrap(full) | 8.23 / 57.14 | 43.45 / 62.86 |
| TEP Mode 1 | 500 | 34 | 25 | RBULT | 0.00 / 68.65 | 28.95 / 84.28 |
| | | | | Shewhart | 0.00 / 79.92 | 55.26 / 91.57 |
| TEP Mode 3 | 500 | 34 | 25 | RBULT | 100.00 / 98.98 | 100.00 / 100.00 |
| TEP Mode 4 | 500 | 34 | 25 | RBULT | 0.00 / 62.73 | 4.88 / 73.56 |
| TEP Mode 5 | 500 | 34 | 25 | RBULT | 11.36 / 63.41 | 27.27 / 81.82 |

At a fixed C, TOTAL is simply a more sensitive operating point — both false alarms
and detection rise together. Comparing the two at the same C therefore says nothing
about which is better; the comparison has to be made at matched false alarm rate.

## 3. Matched-FAR comparison (RBULT only)

| Dataset | AUC (max over features) | AUC (sum over features) | det @ FAR<=5% pf → tot | det @ FAR<=25% pf → tot |
|---|---:|---:|---|---|
| AI4I 2020 | 0.435 | 0.512 | 10.6% → 12.8% | 16.0% → 21.3% |
| Water Pump | 0.402 | 0.404 | 5.6% → 2.8% | 11.1% = 11.1% |
| MetroPT-3 | 0.154 | 0.153 | 8.6% = 8.6% | 11.4% = 11.4% |
| TEP Mode 1 | 0.884 | 0.888 | 74.5% → 76.1% | 82.0% → 82.6% |
| **TEP Mode 3** | **0.448** | **0.859** | **18.3% → 65.8%** | **31.7% → 83.1%** |
| TEP Mode 4 | 0.888 | 0.869 | 74.7% → 74.4% | 81.5% → 77.7% |
| TEP Mode 5 | 0.829 | 0.838 | 51.0% → 52.2% | 78.8% → 78.4% |

On six of seven datasets the two semantics are equivalent — AUC differs by at most
0.02 and matched-FAR detection by a few points. TOTAL is marginally better on AI4I
and TEP Modes 1 and 5; PER-FEATURE is marginally better on Water Pump and TEP Mode 4.

## 4. TEP Mode 3 is the exception, and it overturns an earlier conclusion

AUC rises from **0.448 (indistinguishable from random) to 0.859**, and detection at
FAR <= 5% from 18.3% to **65.8%**.

`spc_pct_threshold_report.md` concluded that TEP Mode 3 "has no signal" and that no
threshold rule could help it. That conclusion was an artefact of measuring only the
per-feature statistic. Mode 3 does carry a strong fault signal — it just lives in
**many dimensions drifting slightly at once** rather than one dimension breaking
badly, so a per-feature maximum cannot see it while a sum across dimensions can.

This is physically plausible for Mode 3 (50/50 mass ratio), where a disturbance
would be expected to propagate across the whole process rather than localise in a
single sensor.

## 5. Recommendation

Do not choose one; **run both statistics together** — alarm when
`(any V_d >= C_perfeat)` **or** `(sum_d V_d >= C_total)`. They detect different fault
geometries: per-feature catches localised faults, total catches distributed ones.
TEP Mode 3 vs Mode 4 is direct evidence that a single benchmark suite contains both
kinds, and that either statistic alone misses one of them.

Both thresholds would need their own calibration, and the combined rule's false alarm
rate is bounded by the sum of the two — so each should be set at `alpha_chunk / 2`.

---
Raw numbers: `spc_total_vs_perfeature.csv`. Per-chunk per-feature violation counts for
every method and dataset are cached as `counts_*.npz` in the session scratchpad, so
further threshold rules can be evaluated without re-streaming.
