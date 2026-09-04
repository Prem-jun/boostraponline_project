# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 5 (10/90 Mass Ratio, Max Rate) [within-run differenced])

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 97.73% | 96.31% | 99.14% | **97.37%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 2.27% | 3.69% | 0.86% | **2.63%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 0.00% | 0.00% | 0.00% | **0.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 100.00 | 100.00 | 100.00 | **100.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.90 | 1.46 | 1.00 | **1.84** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.6735 ms | 4.4836 ms | 5.2903 ms | **12.5315 ms** | Real-time Stream Execution |
