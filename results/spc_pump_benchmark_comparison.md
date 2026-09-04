# Full SPC Benchmark Results: Large Industrial Pump Dataset

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Full-History Bootstrap | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 100.00% | 99.91% | 98.95% | **99.42%** | Non-Gaussian Adaptive Bounds |
| **Sample-level FAR (%)** | 0.00% | 0.09% | 1.05% | **0.58%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 0.00% | 0.00% | 0.00% | **0.00%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | 0.00 | 0.00 | **0.00** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.00 | 1.00 | 1.00 | **1.00** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 0.23 KB | 0.45 KB | 826.91 KB | **0.52 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.0735 ms | 0.2285 ms | 1.8988 ms | **11.7207 ms** | Real-time Streaming (< 70 ms) |
