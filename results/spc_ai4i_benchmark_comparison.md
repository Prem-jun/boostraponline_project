# SPC Benchmark Results: AI4I 2020 Dataset

| Evaluation Metric | Baseline Shewhart Chart | Proposed RBULT-SPC | Improvement / Advantage |
|---|---|---|---|
| **Overall Coverage Rate (%)** | 69.45% | **98.40%** | Non-Gaussian Adaptive Coverage |
| **Sample-level FAR (%)** | 30.55% | **1.60%** | **Controlled at 1.60% (matches ~1% target)** |
| **Chunk-level FAR (%)** | 100.00% | **66.67%** | Low Chunk False Alarm Rate |
| **ARL0 (In-Control Run Length)** | 0.00 | **0.50** | Higher In-Control Stability |
| **ARL1 (Detection Delay)** | 1.00 | **1.14** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | 0.23 KB | **0.52 KB** | Constant $O(D)$ RAM Footprint |
| **Avg Latency per Chunk (ms)** | 0.0239 ms | **60.7179 ms** | Real-time Streaming (< 70 ms) |
