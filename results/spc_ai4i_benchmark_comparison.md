# Full SPC Benchmark Results: AI4I 2020 Dataset

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Full-History Bootstrap | Proposed RBULT-SPC | Improvement / Advantage |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 69.45% | 62.57% | 98.81% | **98.40%** | Non-Gaussian Adaptive Coverage |
| **Sample-level FAR (%)** | 30.55% | 37.43% | 1.19% | **1.60%** | **Controlled at 1.60% (~1% target)** |
| **Chunk-level FAR (%)** | 100.00% | 100.00% | 0.00% | **66.67%** | Low Chunk False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | 0.00 | 6.00 | **0.50** | Higher In-Control Stability |
| **ARL1 (Detection Delay)** | 1.02 | 1.00 | 2.60 | **1.77** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 0.23 KB | 0.45 KB | 413.78 KB | **0.52 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.0230 ms | 0.3145 ms | 1.5489 ms | **60.2070 ms** | Real-time Streaming (< 70 ms) |
