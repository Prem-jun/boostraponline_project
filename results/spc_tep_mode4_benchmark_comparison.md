# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 4)

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 84.72% | 72.12% | 99.01% | **96.67%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 15.28% | 27.88% | 0.99% | **3.33%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 4.88% | 100.00% | 0.00% | **0.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 13.00 | 0.00 | 41.00 | **41.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.16 | 1.00 | 1.00 | **1.57** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.5755 ms | 3.5409 ms | 4.0217 ms | **13.6640 ms** | Real-time Stream Execution |
