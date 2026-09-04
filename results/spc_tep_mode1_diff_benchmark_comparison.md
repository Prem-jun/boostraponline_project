# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 1 [within-run differenced])

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 98.49% | 97.46% | 99.11% | **98.13%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 1.51% | 2.54% | 0.89% | **1.87%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 0.00% | 1.00% | 0.00% | **0.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 100.00 | 49.50 | 100.00 | **100.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 2.01 | 1.77 | 1.00 | **1.97** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.5793 ms | 4.1867 ms | 4.9101 ms | **11.5497 ms** | Real-time Stream Execution |
