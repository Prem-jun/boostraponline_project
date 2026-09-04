# TEP OOC Threshold Count Sensitivity Study (Thresholds = [5, 10, 15, 25, 50])

| Threshold (`ooc_threshold_count`) | Method | Overall Coverage (%) | Sample FAR (%) | Chunk FAR (%) ⭐ | ARL0 | ARL1 (Delay) ⭐ | Peak RAM (KB) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **5** | Baseline Shewhart Chart | 91.23% | 8.77% | **100.00%** | 0.00 | **1.00** | 1.15 KB | 0.60 ms |
| **5** | Baseline EWMA Chart | 81.73% | 18.27% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 4.40 ms |
| **5** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **100.00%** | 0.00 | **1.00** | 582.87 KB | 4.28 ms |
| **5** | Proposed RBULT-SPC | 93.70% | 6.30% | **100.00%** | 0.00 | **1.00** | 3.23 KB | 11.75 ms |
| **10** | Baseline Shewhart Chart | 91.23% | 8.77% | **100.00%** | 0.00 | **1.00** | 1.15 KB | 0.60 ms |
| **10** | Baseline EWMA Chart | 81.73% | 18.27% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 4.40 ms |
| **10** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **100.00%** | 0.00 | **1.00** | 582.87 KB | 4.28 ms |
| **10** | Proposed RBULT-SPC | 93.70% | 6.30% | **100.00%** | 0.00 | **1.00** | 3.23 KB | 11.75 ms |
| **15** | Baseline Shewhart Chart | 91.23% | 8.77% | **98.00%** | 0.02 | **1.01** | 1.15 KB | 0.60 ms |
| **15** | Baseline EWMA Chart | 81.73% | 18.27% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 4.40 ms |
| **15** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **98.00%** | 0.02 | **1.02** | 582.87 KB | 4.28 ms |
| **15** | Proposed RBULT-SPC | 93.70% | 6.30% | **100.00%** | 0.00 | **1.00** | 3.23 KB | 11.75 ms |
| **25** | Baseline Shewhart Chart | 91.23% | 8.77% | **68.00%** | 0.47 | **1.09** | 1.15 KB | 0.60 ms |
| **25** | Baseline EWMA Chart | 81.73% | 18.27% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 4.40 ms |
| **25** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **87.00%** | 0.15 | **1.06** | 582.87 KB | 4.28 ms |
| **25** | Proposed RBULT-SPC | 93.70% | 6.30% | **83.00%** | 0.20 | **1.05** | 3.23 KB | 11.75 ms |
| **50** | Baseline Shewhart Chart | 91.23% | 8.77% | **14.00%** | 5.73 | **1.33** | 1.15 KB | 0.60 ms |
| **50** | Baseline EWMA Chart | 81.73% | 18.27% | **100.00%** | 0.00 | **1.00** | 2.30 KB | 4.40 ms |
| **50** | Baseline Sliding-Window Bootstrap (W=2000) | 99.14% | 0.86% | **75.00%** | 0.33 | **1.15** | 582.87 KB | 4.28 ms |
| **50** | Proposed RBULT-SPC | 93.70% | 6.30% | **5.00%** | 15.83 | **1.41** | 3.23 KB | 11.75 ms |
