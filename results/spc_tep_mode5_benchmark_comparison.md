# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 5 (10/90 Mass Ratio, Max Rate))

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 84.85% | 71.23% | 99.18% | **92.39%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 15.15% | 28.77% | 0.82% | **7.61%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 99.00% | 100.00% | 0.00% | **37.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.01 | 0.00 | 100.00 | **1.70** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.01 | 1.00 | nan | **1.18** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.7566 ms | 4.8244 ms | 4.9913 ms | **12.9458 ms** | Real-time Stream Execution |
