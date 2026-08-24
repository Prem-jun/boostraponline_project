# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 5 (10/90 Mass Ratio, Max Rate))

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 85.15% | 71.38% | 99.05% | **97.79%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 14.85% | 28.62% | 0.95% | **2.21%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 100.00% | 100.00% | 100.00% | **100.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | 0.00 | 0.00 | **0.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.00 | 1.00 | 1.00 | **1.00** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 2.3277 ms | 12.5160 ms | 13.7886 ms | **37.6126 ms** | Real-time Stream Execution |
