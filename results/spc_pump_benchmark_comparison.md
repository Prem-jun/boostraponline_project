# Full SPC Benchmark Results: Large Industrial Pump Dataset

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Full-History Bootstrap | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | 100.00% | 99.91% | 98.95% | **99.44%** | Non-Gaussian Adaptive Bounds |
| **Sample-level FAR (%)** | 0.00% | 0.09% | 1.05% | **0.56%** | Controlled near Bonferroni $\alpha_{dim}$ |
| **Chunk-level FAR (%)** | — | — | — | **—** | **UNDEFINED — no in-control chunks** (see label quality note) |
| **ARL0 (In-Control Run Length)** | — | — | — | **—** | **UNDEFINED — no in-control chunks** (see label quality note) |
| **ARL1 (Detection Delay)** | — | — | — | **—** | **UNDEFINED — no in-control chunks** (see label quality note) |
| **Peak Memory Footprint (KB)** | 0.23 KB | 0.45 KB | 826.91 KB | **0.52 KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | 0.0724 ms | 0.2242 ms | 1.8748 ms | **11.0319 ms** | Real-time Streaming (< 70 ms) |
