# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 1)

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 91.23% | 81.73% | 99.14% | **93.70%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 8.77% | 18.27% | 0.86% | **6.30%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 1.00% | 100.00% | 0.00% | **0.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 49.50 | 0.00 | 100.00 | **100.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.46 | 1.00 | nan | **1.53** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.8241 ms | 4.4458 ms | 4.8603 ms | **13.1286 ms** | Real-time Stream Execution |
