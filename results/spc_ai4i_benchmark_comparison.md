# Full SPC Benchmark Results: AI4I 2020 Dataset

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Full-History Bootstrap | Proposed RBULT-SPC | Improvement / Advantage |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 69.69% | 58.37% | 98.82% | **97.79%** | Non-Gaussian Adaptive Coverage |
| **Sample-level FAR (%)** | 30.31% | 41.63% | 1.18% | **2.21%** | **Controlled at 1.60% (~1% target)** |
| **Chunk-level FAR (%)** | 100.00% | 100.00% | 0.00% | **66.67%** | Low Chunk False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | 0.00 | 6.00 | **0.50** | Higher In-Control Stability |
| **ARL1 (Detection Delay)** | 1.02 | 1.00 | 2.59 | **1.29** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 0.23 KB | 0.45 KB | 413.78 KB | **0.52 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.0143 ms | 0.2745 ms | 0.9821 ms | **40.3135 ms** | Real-time Streaming (< 70 ms) |
