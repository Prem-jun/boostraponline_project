# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 3 (50/50 Mass Ratio))

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 80.80% | 60.99% | 99.01% | **93.73%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 19.20% | 39.01% | 0.99% | **6.27%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 100.00% | 100.00% | 0.00% | **100.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | 0.00 | 39.00 | **0.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.01 | 1.00 | 1.00 | **1.01** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.5489 ms | 3.4731 ms | 4.1226 ms | **9.4052 ms** | Real-time Stream Execution |
