# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 1)

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 91.19% | 81.79% | 99.02% | **96.74%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 8.81% | 18.21% | 0.98% | **3.26%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 0.00% | 100.00% | 0.00% | **0.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 38.00 | 0.00 | 38.00 | **38.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.24 | 1.00 | 1.00 | **1.44** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.5761 ms | 3.5050 ms | 4.1118 ms | **9.1600 ms** | Real-time Stream Execution |
