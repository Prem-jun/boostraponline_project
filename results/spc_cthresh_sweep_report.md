# Paper 1 — Tier 2 Re-run with Updated `spc_rbult.py` (C_thresh Sweep)

All Tier-2 industrial benchmarks were re-streamed with the updated `online_bootstrap/spc_rbult.py`. Coverage, Sample FAR, Peak RAM and Latency are independent of the chunk alarm threshold and are reported once per method. Chunk FAR, ARL0 and ARL1 are reported at C_thresh in {3, 5, 7, 10}.


## 1. Threshold-independent metrics (Coverage / Sample FAR / RAM / Latency)


### AI4I 2020 — N = 10,000, D = 5, chunk = 100, chunks = 100

| Method | Coverage (%) | Sample FAR (%) | Peak RAM (KB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 69.45 | 30.55 | 0.23 | 0.0739 |
| Baseline EWMA Chart | 62.57 | 37.43 | 0.45 | 0.1755 |
| Baseline Full-History Bootstrap | 98.81 | 1.19 | 413.78 | 0.9360 |
| **Proposed RBULT-SPC** | 98.38 | 1.62 | 0.52 | 40.0005 |

### MetroPT-3 — N = 1,516,948, D = 7, chunk = 1000, chunks = 1,517

| Method | Coverage (%) | Sample FAR (%) | Peak RAM (KB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 77.68 | 22.32 | 0.35 | 0.1253 |
| Baseline EWMA Chart | 51.01 | 48.99 | 0.70 | 1.0606 |
| Baseline Full-History Bootstrap | 98.76 | 1.24 | 90,932.70 | 169.4487 |
| **Proposed RBULT-SPC** | 98.90 | 1.10 | 0.70 | 5.2291 |

### Industrial Pump — N = 20,000, D = 5, chunk = 200, chunks = 100

| Method | Coverage (%) | Sample FAR (%) | Peak RAM (KB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 100.00 | 0.00 | 0.23 | 0.0782 |
| Baseline EWMA Chart | 99.91 | 0.09 | 0.45 | 0.1950 |
| Baseline Full-History Bootstrap | 98.95 | 1.05 | 826.91 | 2.0480 |
| **Proposed RBULT-SPC** | 99.41 | 0.59 | 0.52 | 15.8913 |

### Water Pump — N = 220,320, D = 10, chunk = 500, chunks = 441

| Method | Coverage (%) | Sample FAR (%) | Peak RAM (KB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 51.06 | 48.94 | 0.35 | 0.1779 |
| Baseline EWMA Chart | 25.65 | 74.35 | 0.70 | 0.9226 |
| Baseline Full-History Bootstrap | 98.63 | 1.37 | 17,667.15 | 31.2378 |
| **Proposed RBULT-SPC** | 99.95 | 0.05 | 0.98 | 26.9880 |

### TEP Mode 1 — N = 1,740,000, D = 34, chunk = 500, chunks = 3,480

| Method | Coverage (%) | Sample FAR (%) | Peak RAM (KB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 91.19 | 8.81 | 1.15 | 0.6877 |
| Baseline EWMA Chart | 81.79 | 18.21 | 2.30 | 3.3099 |
| Baseline Sliding-Window Bootstrap (W=2000) | 99.02 | 0.98 | 582.87 | 5.0295 |
| **Proposed RBULT-SPC** | 96.74 | 3.26 | 3.23 | 9.6481 |

### TEP Mode 3 — N = 1,739,400, D = 34, chunk = 500, chunks = 3,479

| Method | Coverage (%) | Sample FAR (%) | Peak RAM (KB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 80.80 | 19.20 | 1.15 | 0.6848 |
| Baseline EWMA Chart | 60.99 | 39.01 | 2.30 | 3.2556 |
| Baseline Sliding-Window Bootstrap (W=2000) | 99.01 | 0.99 | 582.87 | 5.0543 |
| **Proposed RBULT-SPC** | 93.72 | 6.28 | 3.23 | 9.8198 |

### TEP Mode 4 — N = 1,719,000, D = 34, chunk = 500, chunks = 3,438

| Method | Coverage (%) | Sample FAR (%) | Peak RAM (KB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 84.72 | 15.28 | 1.15 | 0.6181 |
| Baseline EWMA Chart | 69.33 | 30.67 | 2.30 | 2.9683 |
| Baseline Sliding-Window Bootstrap (W=2000) | 99.01 | 0.99 | 582.87 | 4.5283 |
| **Proposed RBULT-SPC** | 96.66 | 3.34 | 3.23 | 12.6644 |

### TEP Mode 5 — N = 1,729,800, D = 34, chunk = 500, chunks = 3,460

| Method | Coverage (%) | Sample FAR (%) | Peak RAM (KB) | Latency (ms) |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 85.15 | 14.85 | 1.15 | 0.6273 |
| Baseline EWMA Chart | 71.38 | 28.62 | 2.30 | 2.9804 |
| Baseline Sliding-Window Bootstrap (W=2000) | 99.05 | 0.95 | 582.87 | 4.5824 |
| **Proposed RBULT-SPC** | 97.79 | 2.21 | 3.23 | 10.4137 |


## 2. NEW — Marginal vs. Joint Coverage (added in this `spc_rbult.py` update)

`overall_coverage_pct` is the marginal coverage averaged over dimensions; `joint_coverage_pct` requires ALL D dimensions to be simultaneously in-bounds. Bonferroni guarantees joint coverage >= 1 - alpha_sys = 95% *only if* every marginal interval attains 1 - alpha_dim.

| Dataset | D | alpha_dim | Marginal Coverage (%) | Joint Coverage (%) | Joint FAR (%) | Joint >= 95%? |
|---|---:|---:|---:|---:|---:|:--:|
| AI4I 2020 | 5 | 0.01000 | 98.38 | **94.14** | 5.86 | **NO** |
| MetroPT-3 | 7 | 0.00714 | 98.90 | **94.89** | 5.11 | **NO** |
| Industrial Pump | 5 | 0.01000 | 99.41 | **97.07** | 2.93 | yes |
| Water Pump | 10 | 0.00500 | 99.95 | **99.55** | 0.45 | yes |
| TEP Mode 1 | 34 | 0.00147 | 96.74 | **61.37** | 38.63 | **NO** |
| TEP Mode 3 | 34 | 0.00147 | 93.72 | **25.07** | 74.93 | **NO** |
| TEP Mode 4 | 34 | 0.00147 | 96.66 | **65.26** | 34.74 | **NO** |
| TEP Mode 5 | 34 | 0.00147 | 97.79 | **70.46** | 29.54 | **NO** |


## 3. Chunk-level metrics across C_thresh


### AI4I 2020

> The table currently stored in `results/` for this dataset matches **C_thresh = 10**.

| Method | C_thresh | Chunk FAR (%) | ARL0 | ARL1 |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 3 | 100.00 | 0.00 | 1.02 |
| Baseline Shewhart Chart | 5 | 100.00 | 0.00 | 1.02 |
| Baseline Shewhart Chart | 7 | 100.00 | 0.00 | 1.02 |
| Baseline Shewhart Chart | 10 | 100.00 | 0.00 | 1.02 |
| Baseline EWMA Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline Full-History Bootstrap | 3 | 0.00 | 6.00 | 2.29 |
| Baseline Full-History Bootstrap | 5 | 0.00 | 6.00 | 2.60 |
| Baseline Full-History Bootstrap | 7 | 0.00 | 6.00 | 3.47 |
| Baseline Full-History Bootstrap | 10 | 0.00 | 6.00 | 4.73 |
| **Proposed RBULT-SPC** | 3 | 66.67 | 0.50 | 1.12 |
| **Proposed RBULT-SPC** | 5 | 66.67 | 0.50 | 1.73 |
| **Proposed RBULT-SPC** | 7 | 33.33 | 2.00 | 3.94 |
| **Proposed RBULT-SPC** | 10 | 0.00 | 6.00 | 6.50 |

### MetroPT-3

> The table currently stored in `results/` for this dataset matches **C_thresh = 7**.

| Method | C_thresh | Chunk FAR (%) | ARL0 | ARL1 |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 3 | 99.80 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 5 | 99.80 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 7 | 99.80 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 10 | 99.80 | 0.00 | 1.00 |
| Baseline EWMA Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline Full-History Bootstrap | 3 | 96.83 | 0.03 | 1.40 |
| Baseline Full-History Bootstrap | 5 | 96.76 | 0.03 | 1.40 |
| Baseline Full-History Bootstrap | 7 | 96.69 | 0.03 | 1.40 |
| Baseline Full-History Bootstrap | 10 | 96.42 | 0.04 | 1.40 |
| **Proposed RBULT-SPC** | 3 | 95.61 | 0.05 | 3.12 |
| **Proposed RBULT-SPC** | 5 | 95.55 | 0.05 | 3.12 |
| **Proposed RBULT-SPC** | 7 | 95.41 | 0.05 | 3.12 |
| **Proposed RBULT-SPC** | 10 | 95.28 | 0.05 | 3.12 |

### Industrial Pump

> The table currently stored in `results/` for this dataset matches **C_thresh = 3**.

| Method | C_thresh | Chunk FAR (%) | ARL0 | ARL1 |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 3 | 0.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 5 | 0.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 7 | 0.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 10 | 0.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 3 | 0.00 | 0.00 | 12.25 |
| Baseline EWMA Chart | 5 | 0.00 | 0.00 | 32.67 |
| Baseline EWMA Chart | 7 | 0.00 | 0.00 | 46.00 |
| Baseline EWMA Chart | 10 | 0.00 | 0.00 | 1.00 |
| Baseline Full-History Bootstrap | 3 | 0.00 | 0.00 | 1.00 |
| Baseline Full-History Bootstrap | 5 | 0.00 | 0.00 | 1.02 |
| Baseline Full-History Bootstrap | 7 | 0.00 | 0.00 | 1.10 |
| Baseline Full-History Bootstrap | 10 | 0.00 | 0.00 | 1.67 |
| **Proposed RBULT-SPC** | 3 | 0.00 | 0.00 | 1.45 |
| **Proposed RBULT-SPC** | 5 | 0.00 | 0.00 | 8.18 |
| **Proposed RBULT-SPC** | 7 | 0.00 | 0.00 | 79.00 |
| **Proposed RBULT-SPC** | 10 | 0.00 | 0.00 | 1.00 |

### Water Pump

> The table currently stored in `results/` for this dataset matches **C_thresh = 3**.

| Method | C_thresh | Chunk FAR (%) | ARL0 | ARL1 |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline Full-History Bootstrap | 3 | 81.23 | 0.23 | 1.00 |
| Baseline Full-History Bootstrap | 5 | 76.30 | 0.31 | 1.17 |
| Baseline Full-History Bootstrap | 7 | 71.60 | 0.40 | 1.17 |
| Baseline Full-History Bootstrap | 10 | 63.46 | 0.58 | 1.17 |
| **Proposed RBULT-SPC** | 3 | 47.65 | 1.09 | 2.40 |
| **Proposed RBULT-SPC** | 5 | 42.72 | 1.33 | 2.40 |
| **Proposed RBULT-SPC** | 7 | 39.26 | 1.54 | 2.40 |
| **Proposed RBULT-SPC** | 10 | 36.79 | 1.71 | 2.67 |

### TEP Mode 1

| Method | C_thresh | Chunk FAR (%) | ARL0 | ARL1 |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 3 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 5 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 7 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 10 | 100.00 | 0.00 | 1.00 |
| **Proposed RBULT-SPC** | 3 | 100.00 | 0.00 | 1.00 |
| **Proposed RBULT-SPC** | 5 | 97.37 | 0.03 | 1.01 |
| **Proposed RBULT-SPC** | 7 | 68.42 | 0.46 | 1.06 |
| **Proposed RBULT-SPC** | 10 | 23.68 | 2.90 | 1.21 |

### TEP Mode 3

| Method | C_thresh | Chunk FAR (%) | ARL0 | ARL1 |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 3 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 5 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 7 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 10 | 97.44 | 0.03 | 1.00 |
| **Proposed RBULT-SPC** | 3 | 100.00 | 0.00 | 1.00 |
| **Proposed RBULT-SPC** | 5 | 100.00 | 0.00 | 1.00 |
| **Proposed RBULT-SPC** | 7 | 100.00 | 0.00 | 1.00 |
| **Proposed RBULT-SPC** | 10 | 100.00 | 0.00 | 1.01 |

### TEP Mode 4

| Method | C_thresh | Chunk FAR (%) | ARL0 | ARL1 |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 10 | 97.56 | 0.03 | 1.00 |
| Baseline EWMA Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 3 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 5 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 7 | 97.56 | 0.03 | 1.01 |
| Baseline Sliding-Window Bootstrap (W=2000) | 10 | 95.12 | 0.05 | 1.02 |
| **Proposed RBULT-SPC** | 3 | 92.68 | 0.08 | 1.01 |
| **Proposed RBULT-SPC** | 5 | 36.59 | 1.73 | 1.14 |
| **Proposed RBULT-SPC** | 7 | 12.20 | 6.00 | 1.29 |
| **Proposed RBULT-SPC** | 10 | 0.00 | 41.00 | 1.38 |

### TEP Mode 5

| Method | C_thresh | Chunk FAR (%) | ARL0 | ARL1 |
|---|---:|---:|---:|---:|
| Baseline Shewhart Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline Shewhart Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 3 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 5 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 7 | 100.00 | 0.00 | 1.00 |
| Baseline EWMA Chart | 10 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 3 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 5 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 7 | 100.00 | 0.00 | 1.00 |
| Baseline Sliding-Window Bootstrap (W=2000) | 10 | 100.00 | 0.00 | 1.00 |
| **Proposed RBULT-SPC** | 3 | 100.00 | 0.00 | 1.00 |
| **Proposed RBULT-SPC** | 5 | 93.18 | 0.07 | 1.02 |
| **Proposed RBULT-SPC** | 7 | 59.09 | 0.67 | 1.14 |
| **Proposed RBULT-SPC** | 10 | 15.91 | 4.62 | 1.31 |


## 4. RBULT-SPC only — Chunk FAR / ARL0 sensitivity to C_thresh


**Chunk FAR (%)**

| Dataset | C=3 | C=5 | C=7 | C=10 |
|---|---:|---:|---:|---:|
| AI4I 2020 | 66.67 | 66.67 | 33.33 | 0.00 |
| MetroPT-3 | 95.61 | 95.55 | 95.41 | 95.28 |
| Industrial Pump | 0.00 | 0.00 | 0.00 | 0.00 |
| Water Pump | 47.65 | 42.72 | 39.26 | 36.79 |
| TEP Mode 1 | 100.00 | 97.37 | 68.42 | 23.68 |
| TEP Mode 3 | 100.00 | 100.00 | 100.00 | 100.00 |
| TEP Mode 4 | 92.68 | 36.59 | 12.20 | 0.00 |
| TEP Mode 5 | 100.00 | 93.18 | 59.09 | 15.91 |

**ARL0**

| Dataset | C=3 | C=5 | C=7 | C=10 |
|---|---:|---:|---:|---:|
| AI4I 2020 | 0.50 | 0.50 | 2.00 | 6.00 |
| MetroPT-3 | 0.05 | 0.05 | 0.05 | 0.05 |
| Industrial Pump | 0.00 | 0.00 | 0.00 | 0.00 |
| Water Pump | 1.09 | 1.33 | 1.54 | 1.71 |
| TEP Mode 1 | 0.00 | 0.03 | 0.46 | 2.90 |
| TEP Mode 3 | 0.00 | 0.00 | 0.00 | 0.00 |
| TEP Mode 4 | 0.08 | 1.73 | 6.00 | 41.00 |
| TEP Mode 5 | 0.00 | 0.07 | 0.67 | 4.62 |

**ARL1**

| Dataset | C=3 | C=5 | C=7 | C=10 |
|---|---:|---:|---:|---:|
| AI4I 2020 | 1.12 | 1.73 | 3.94 | 6.50 |
| MetroPT-3 | 3.12 | 3.12 | 3.12 | 3.12 |
| Industrial Pump | 1.45 | 8.18 | 79.00 | 1.00 |
| Water Pump | 2.40 | 2.40 | 2.40 | 2.67 |
| TEP Mode 1 | 1.00 | 1.01 | 1.06 | 1.21 |
| TEP Mode 3 | 1.00 | 1.00 | 1.00 | 1.01 |
| TEP Mode 4 | 1.01 | 1.14 | 1.29 | 1.38 |
| TEP Mode 5 | 1.00 | 1.02 | 1.14 | 1.31 |


## 5. Corrected TEP Mode 1 threshold-sensitivity study

`exp_tep_sensitivity.py:56` reads `summary['sample_ooc_count']`, a key that `update_chunk` does not return, so RBULT was scored as having zero violations in every chunk. That, not a measurement, is the source of the published 'Chunk FAR = 0.00% at every threshold, ARL0 = 38.00' — and 38 is exactly TEP Mode 1's in-control chunk count, which `_compute_arl0` returns when no alarm ever fires. Below, RBULT's per-feature counts are summed across features, matching how that script aggregates violations for the three baselines.

| C_thresh | Method | Chunk FAR (%) | ARL0 | ARL1 | Published (buggy) RBULT |
|---:|---|---:|---:|---:|---|
| 5 | Baseline Shewhart Chart | 100.00 | 0.00 | 1.00 |  |
| 5 | Baseline EWMA Chart | 100.00 | 0.00 | 1.00 |  |
| 5 | Baseline Sliding-Window Bootstrap (W=2000) | 100.00 | 0.00 | 1.00 |  |
| 5 | **Proposed RBULT-SPC** | 100.00 | 0.00 | 1.00 | 0.00% / ARL0 38.00 |
| 10 | Baseline Shewhart Chart | 100.00 | 0.00 | 1.00 |  |
| 10 | Baseline EWMA Chart | 100.00 | 0.00 | 1.00 |  |
| 10 | Baseline Sliding-Window Bootstrap (W=2000) | 100.00 | 0.00 | 1.00 |  |
| 10 | **Proposed RBULT-SPC** | 100.00 | 0.00 | 1.00 | 0.00% / ARL0 38.00 |
| 15 | Baseline Shewhart Chart | 92.11 | 0.09 | 1.02 |  |
| 15 | Baseline EWMA Chart | 100.00 | 0.00 | 1.00 |  |
| 15 | Baseline Sliding-Window Bootstrap (W=2000) | 94.74 | 0.06 | 1.02 |  |
| 15 | **Proposed RBULT-SPC** | 94.74 | 0.06 | 1.01 | 0.00% / ARL0 38.00 |

The three baseline rows reproduce the published study exactly (Shewhart 92.11% and Sliding-Window 94.74% at C=15; all 100% at C=5 and C=10), which confirms the recomputation matches that script's semantics. Corrected, RBULT-SPC is **not** free of chunk-level false alarms on TEP Mode 1.

