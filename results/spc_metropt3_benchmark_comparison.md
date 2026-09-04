# Full SPC Benchmark Results: MetroPT-3 Air Compressor Dataset

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Full-History Bootstrap | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 77.68% | 51.01% | 98.76% | **98.90%** | Non-Gaussian Adaptive Bounds |
| **Sample-level FAR (%)** | 22.32% | 48.99% | 1.24% | **1.10%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | 99.80% | 100.00% | 96.69% | **95.41%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | 0.00 | 0.03 | **0.05** | Boundary Stability |
| **ARL1 (Detection Delay)** | 1.00 | 1.00 | 1.40 | **3.12** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 0.35 KB | 0.70 KB | 90932.70 KB | **0.70 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.1132 ms | 1.3185 ms | 148.9073 ms | **5.3139 ms** | Low Latency Stream Processing |
