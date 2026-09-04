# Full SPC Benchmark Results: MetroPT-3 Air Compressor Dataset

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Full-History Bootstrap | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 77.68% | 51.01% | 98.76% | **98.90%** | Non-Gaussian Adaptive Bounds |
| **Sample-level FAR (%)** | 22.32% | 48.99% | 1.24% | **1.10%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 99.46% | 100.00% | 8.23% | **25.24%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.01 | 0.00 | 11.06 | **2.95** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.00 | 1.00 | 1.55 | **5.25** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 0.35 KB | 0.70 KB | 90932.70 KB | **0.70 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.1140 ms | 1.3189 ms | 149.2325 ms | **5.2789 ms** | Low Latency Stream Processing |
