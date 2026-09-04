# TEP OOC Threshold Count Sensitivity Study (Thresholds = [5, 10, 15, 25, 50])

| Threshold (`ooc_threshold_count`) | Method | Overall Coverage (%) | Sample FAR (%) | Chunk FAR (%) ⭐ | ARL0 | ARL1 (Delay) ⭐ | Peak RAM (KB) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **5** | Baseline Shewhart Chart | 91.19% | 8.81% | **100.00%** | 0.00 | **1.00** | 1.15 KB | 0.58 ms |
| **5** | Baseline EWMA Chart | 81.79% | 18.21% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 3.45 ms |
| **5** | Baseline Sliding-Window Bootstrap (W=2000) | 99.02% | 0.98% | **100.00%** | 0.00 | **1.00** | 582.87 KB | 4.02 ms |
| **5** | Proposed RBULT-SPC | 96.73% | 3.27% | **100.00%** | 0.00 | **1.00** | 3.23 KB | 8.92 ms |
| **10** | Baseline Shewhart Chart | 91.19% | 8.81% | **100.00%** | 0.00 | **1.00** | 1.15 KB | 0.58 ms |
| **10** | Baseline EWMA Chart | 81.79% | 18.21% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 3.45 ms |
| **10** | Baseline Sliding-Window Bootstrap (W=2000) | 99.02% | 0.98% | **100.00%** | 0.00 | **1.00** | 582.87 KB | 4.02 ms |
| **10** | Proposed RBULT-SPC | 96.73% | 3.27% | **100.00%** | 0.00 | **1.00** | 3.23 KB | 8.92 ms |
| **15** | Baseline Shewhart Chart | 91.19% | 8.81% | **92.11%** | 0.09 | **1.02** | 1.15 KB | 0.58 ms |
| **15** | Baseline EWMA Chart | 81.79% | 18.21% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 3.45 ms |
| **15** | Baseline Sliding-Window Bootstrap (W=2000) | 99.02% | 0.98% | **94.74%** | 0.06 | **1.02** | 582.87 KB | 4.02 ms |
| **15** | Proposed RBULT-SPC | 96.73% | 3.27% | **94.74%** | 0.06 | **1.01** | 3.23 KB | 8.92 ms |
| **25** | Baseline Shewhart Chart | 91.19% | 8.81% | **55.26%** | 0.81 | **1.09** | 1.15 KB | 0.58 ms |
| **25** | Baseline EWMA Chart | 81.79% | 18.21% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 3.45 ms |
| **25** | Baseline Sliding-Window Bootstrap (W=2000) | 99.02% | 0.98% | **76.32%** | 0.30 | **1.06** | 582.87 KB | 4.02 ms |
| **25** | Proposed RBULT-SPC | 96.73% | 3.27% | **34.21%** | 1.79 | **1.18** | 3.23 KB | 8.92 ms |
| **50** | Baseline Shewhart Chart | 91.19% | 8.81% | **10.53%** | 6.80 | **1.21** | 1.15 KB | 0.58 ms |
| **50** | Baseline EWMA Chart | 81.79% | 18.21% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 3.45 ms |
| **50** | Baseline Sliding-Window Bootstrap (W=2000) | 99.02% | 0.98% | **73.68%** | 0.34 | **1.16** | 582.87 KB | 4.02 ms |
| **50** | Proposed RBULT-SPC | 96.73% | 3.27% | **0.00%** | 38.00 | **1.39** | 3.23 KB | 8.92 ms |
