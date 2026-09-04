# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 4 [within-run differenced])

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 96.56% | 94.08% | 99.13% | **96.09%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 3.44% | 5.92% | 0.87% | **3.91%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 0.00% | 3.00% | 0.00% | **0.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 100.00 | 24.25 | 100.00 | **100.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.75 | 1.58 | 1.00 | **1.73** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.7881 ms | 4.4808 ms | 5.3148 ms | **12.4608 ms** | Real-time Stream Execution |
