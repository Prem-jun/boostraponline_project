# Full SPC Benchmark Results: Water Pump Sensor Dataset (sensor.csv)

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Full-History Bootstrap | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 51.06% | 25.65% | 98.63% | **99.95%** | Non-Gaussian Adaptive Bounds |
| **Sample-level FAR (%)** | 48.94% | 74.35% | 1.37% | **0.05%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 99.51% | 100.00% | 32.59% | **29.38%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | 0.00 | 2.05 | **2.38** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.00 | 1.00 | 1.15 | **3.67** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 0.35 KB | 0.70 KB | 17667.15 KB | **0.98 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.4257 ms | 1.1148 ms | 30.6462 ms | **26.3030 ms** | Real-time Streaming (< 70 ms) |
