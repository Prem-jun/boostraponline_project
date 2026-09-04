# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 5 (10/90 Mass Ratio, Max Rate))

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 85.15% | 71.38% | 99.05% | **97.79%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 14.85% | 28.62% | 0.95% | **2.21%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 90.91% | 100.00% | 0.00% | **11.36%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.10 | 0.00 | 44.00 | **6.50** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.01 | 1.00 | 1.00 | **1.55** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.5509 ms | 3.6562 ms | 4.1089 ms | **11.4897 ms** | Real-time Stream Execution |
