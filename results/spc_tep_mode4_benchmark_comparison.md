# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 4)

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 84.83% | 72.29% | 99.18% | **90.64%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 15.17% | 27.71% | 0.82% | **9.36%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 1.00% | 100.00% | 0.00% | **1.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 49.50 | 0.00 | 100.00 | **49.50** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.34 | 1.00 | nan | **1.54** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.5922 ms | 4.7265 ms | 5.0081 ms | **13.3418 ms** | Real-time Stream Execution |
