# Full SPC Benchmark Results: Tennessee Eastman Process (Mode 3 (50/50 Mass Ratio) [within-run differenced])

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W=2000) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 97.65% | 95.59% | 99.09% | **97.20%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | 2.35% | 4.41% | 0.91% | **2.80%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 0.00% | 100.00% | 0.00% | **0.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 100.00 | 0.00 | 100.00 | **100.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.68 | 1.00 | 1.00 | **1.55** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 1.15 KB | 2.30 KB | 582.87 KB | **3.23 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.6051 ms | 4.1827 ms | 4.8840 ms | **11.4221 ms** | Real-time Stream Execution |
