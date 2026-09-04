# TEP OOC Threshold Count Sensitivity Study (Thresholds = [5, 15, 30, 60, 120])

| Threshold (`ooc_threshold_count`) | Method | Overall Coverage (%) | Sample FAR (%) | Chunk FAR (%) ⭐ | ARL0 | ARL1 (Delay) ⭐ | Peak RAM (KB) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **5** | Baseline Shewhart Chart | 91.23% | 8.77% | **74.00%** | 0.35 | **1.08** | 1.15 KB | 0.58 ms |
| **5** | Baseline EWMA Chart | 81.73% | 18.27% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 4.07 ms |
| **5** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **100.00%** | 0.00 | **1.00** | 582.87 KB | 4.37 ms |
| **5** | Proposed RBULT-SPC | 93.70% | 6.30% | **100.00%** | 0.00 | **1.00** | 3.23 KB | 11.59 ms |
| **15** | Baseline Shewhart Chart | 91.23% | 8.77% | **11.00%** | 7.42 | **1.35** | 1.15 KB | 0.58 ms |
| **15** | Baseline EWMA Chart | 81.73% | 18.27% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 4.07 ms |
| **15** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **3.00%** | 24.25 | **1.80** | 582.87 KB | 4.37 ms |
| **15** | Proposed RBULT-SPC | 93.70% | 6.30% | **5.00%** | 15.83 | **1.37** | 3.23 KB | 11.59 ms |
| **30** | Baseline Shewhart Chart | 91.23% | 8.77% | **1.00%** | 49.50 | **1.46** | 1.15 KB | 0.58 ms |
| **30** | Baseline EWMA Chart | 81.73% | 18.27% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 4.07 ms |
| **30** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **0.00%** | 100.00 | **nan** | 582.87 KB | 4.37 ms |
| **30** | Proposed RBULT-SPC | 93.70% | 6.30% | **0.00%** | 100.00 | **1.53** | 3.23 KB | 11.59 ms |
| **60** | Baseline Shewhart Chart | 91.23% | 8.77% | **0.00%** | 100.00 | **1.50** | 1.15 KB | 0.58 ms |
| **60** | Baseline EWMA Chart | 81.73% | 18.27% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 4.07 ms |
| **60** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **0.00%** | 100.00 | **nan** | 582.87 KB | 4.37 ms |
| **60** | Proposed RBULT-SPC | 93.70% | 6.30% | **0.00%** | 100.00 | **1.64** | 3.23 KB | 11.59 ms |
| **120** | Baseline Shewhart Chart | 91.23% | 8.77% | **0.00%** | 100.00 | **1.51** | 1.15 KB | 0.58 ms |
| **120** | Baseline EWMA Chart | 81.73% | 18.27% | **96.00%** | 0.04 | **1.01** | 2.30 KB | 4.07 ms |
| **120** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **0.00%** | 100.00 | **nan** | 582.87 KB | 4.37 ms |
| **120** | Proposed RBULT-SPC | 93.70% | 6.30% | **0.00%** | 100.00 | **1.66** | 3.23 KB | 11.59 ms |
