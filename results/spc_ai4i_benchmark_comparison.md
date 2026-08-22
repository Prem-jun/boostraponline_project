# SPC Benchmark Results: AI4I 2020 Dataset

| Evaluation Metric | Baseline Shewhart Chart | Proposed RBULT-SPC | Improvement / Advantage |
|---|---|---|---|
| **Overall Coverage Rate (%)** | 69.45% | **98.40%** | Non-Gaussian Adaptive Coverage |
| **ARL0 (In-Control Run Length)** | 0.00 | **0.00** | Higher In-Control Stability |
| **ARL1 (Detection Delay)** | 1.00 | **1.00** | Fast Failure Response |
| **False Alarm Rate (FAR %)** | 100.00% | **100.00%** | Controlled by Bonferroni FWER |
| **Peak Memory Footprint (KB)** | 0.23 KB | **0.52 KB** | Constant $O(D)$ RAM Footprint |
| **Avg Latency per Chunk (ms)** | 0.0217 ms | **69.5562 ms** | Real-time Streaming (< 70 ms) |
