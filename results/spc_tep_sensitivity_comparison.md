# TEP OOC Threshold Count Sensitivity Study (Thresholds = [5, 10, 15])

| Threshold (`ooc_threshold_count`) | Method | Overall Coverage (%) | Sample FAR (%) | Chunk FAR (%) ⭐ | ARL0 | ARL1 (Delay) ⭐ | Peak RAM (KB) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **5** | Baseline Shewhart Chart | 91.19% | 8.81% | **100.00%** | 0.00 | **1.00** | 1.15 KB | 2.14 ms |
| **5** | Baseline EWMA Chart | 81.79% | 18.21% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 13.68 ms |
| **5** | Baseline Sliding-Window Bootstrap (W=2000) | 99.02% | 0.98% | **100.00%** | 0.00 | **1.00** | 582.87 KB | 14.15 ms |
| **5** | Proposed RBULT-SPC | 96.74% | 3.26% | **0.00%** | 38.00 | **1.00** | 3.23 KB | 35.10 ms |
| **10** | Baseline Shewhart Chart | 91.19% | 8.81% | **100.00%** | 0.00 | **1.00** | 1.15 KB | 2.14 ms |
| **10** | Baseline EWMA Chart | 81.79% | 18.21% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 13.68 ms |
| **10** | Baseline Sliding-Window Bootstrap (W=2000) | 99.02% | 0.98% | **100.00%** | 0.00 | **1.00** | 582.87 KB | 14.15 ms |
| **10** | Proposed RBULT-SPC | 96.74% | 3.26% | **0.00%** | 38.00 | **1.00** | 3.23 KB | 35.10 ms |
| **15** | Baseline Shewhart Chart | 91.19% | 8.81% | **92.11%** | 0.09 | **1.02** | 1.15 KB | 2.14 ms |
| **15** | Baseline EWMA Chart | 81.79% | 18.21% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 13.68 ms |
| **15** | Baseline Sliding-Window Bootstrap (W=2000) | 99.02% | 0.98% | **94.74%** | 0.06 | **1.02** | 582.87 KB | 14.15 ms |
| **15** | Proposed RBULT-SPC | 96.74% | 3.26% | **0.00%** | 38.00 | **1.00** | 3.23 KB | 35.10 ms |
