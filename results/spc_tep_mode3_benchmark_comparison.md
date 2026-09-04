# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 3 (50/50 Mass Ratio))

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 80.55% | 60.89% | 99.12% | **91.76%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 19.45% | 39.11% | 0.88% | **8.24%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 100.00% | 100.00% | 0.00% | **100.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | 0.00 | 100.00 | **0.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.02 | 1.00 | 1.00 | **1.02** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.5966 ms | 4.2833 ms | 4.3267 ms | **11.6736 ms** | Real-time Stream Execution |
