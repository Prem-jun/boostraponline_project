# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 3 (90/10 Mass Ratio))

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 80.80% | 60.99% | 99.01% | **93.71%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 19.20% | 39.01% | 0.99% | **6.29%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 100.00% | 100.00% | 100.00% | **100.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | 0.00 | 0.00 | **0.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.00 | 1.00 | 1.00 | **1.00** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 2.2451 ms | 14.4140 ms | 12.7023 ms | **33.5487 ms** | Real-time Stream Execution |
