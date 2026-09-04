# Strategic Research Publication Plan & Manuscript Roadmap

## Executive Summary
This document outlines the strategic publication roadmap splitting the research contributions into two distinct, high-impact Q1 journal manuscripts: **Paper 1 (Current Scope - Ready to Publish)** and **Paper 2 (Future Extension - Concept Drift Adaptation)**.

---

## 📘 Paper 1: Resource-Bounded Non-Parametric SPC Framework for High-Dimensional Non-Gaussian IoT Streams

- **Target Journals:** 
  - *Information Sciences* (Elsevier, Q1, IF ~8.1) ⭐ **(Highly Recommended Candidate)**
  - *IEEE Transactions on Industrial Informatics (TII)* (IEEE, Q1, IF ~11.7)
  - *Expert Systems with Applications (ESWA)* (Elsevier, Q1, IF ~7.5)
  - *IEEE Transactions on Instrumentation and Measurement (TIM)* (IEEE, Q1, IF ~5.6)
  - *IEEE Transactions on Knowledge and Data Engineering (TKDE)* (IEEE, Q1, IF ~8.9)

- **Status:** **100% Complete & Ready for Manuscript Drafting**

### Core Scientific Contributions
1. **Strict $O(D)$ Bounded RAM Storage (3.23 KB):** Eliminates memory overflow bottlenecks on edge IoT streaming devices, achieving $>99.99\%$ RAM reduction over conventional sliding-window bootstrap ($180\times$ RAM savings).
2. **Z-Score Spike Filtering & Bonferroni FWER Control:** Outlier suppression via Algorithm 4 + FWER tail scaling for high-dimensional sensor streams ($D=34$).
3. **Multi-Mode Cross-Regime Robustness:** Comprehensive evaluation across 5 real datasets (AI4I 2020, MetroPT-3, Industrial Pump, Water Pump, TEP Modes 1, 3, 4, 5).
4. **Scale-Free Batch Alarm Threshold ($C = \lceil 0.05\,k \rceil$):** Comparable across chunk sizes and dimensionalities, replacing an absolute count. Under this matched protocol RBULT-SPC reaches zero batch false alarms on TEP Modes 1 and 4 with $63\%-69\%$ fault-batch detection, but does **not** dominate the classical charts at the batch level — the decisive advantage is memory, not false-alarm control.
5. **Marginal *and* Joint Coverage Reporting:** Joint (all-dimension) coverage falls to $25\%-71\%$ on $D=34$ streams where marginal coverage exceeds $93\%$ — a gap that marginal coverage alone conceals.

### Key Empirical Result Matrix (TEP Multi-Mode Benchmark Summary)

| Operating Regime | Shewhart Sample FAR | EWMA Sample FAR | Proposed RBULT Sample FAR | Proposed RBULT Coverage | Peak RAM (KB) | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mode 1 (Nominal)** | 8.81% | 18.21% | **3.26%** | **96.74%** | **3.23 KB** | 31.95 ms |
| **Mode 3 (Feed Skew 90/10)** | 19.20% | 39.01% | **6.29%** | **93.71%** | **3.23 KB** | 33.55 ms |
| **Mode 4 (Max Production Rate)** | 15.28% | 27.88% | **3.33%** | **96.67%** | **3.23 KB** | 44.44 ms |
| **Mode 5 (Combined Extreme Stress)** | 14.85% | 28.62% | **2.21%** | **97.79%** | **3.23 KB** | 37.61 ms |

---

## 📙 Paper 2: Adaptive Online Bootstrap SPC under Continuous Concept Drift and Process Evolution

- **Target Journals:** 
  - *Information Sciences* (Q1, IF ~8.1)
  - *IEEE Transactions on Cybernetics* (Q1, IF ~11.8)
  - *Knowledge-Based Systems* (Q1, IF ~8.8)
- **Status:** **Future Scope (Second Project Roadmap)**

### Core Novelty & Proposed Research Extensions
1. **Concept Drift Integration:** Coupling RBULT-SPC with streaming drift detectors (ADWIN, DDM, Page-Hinkley, EWMA-Drift).
2. **Dynamic Window Forgetting / Decay Factor:** Auto-reinitialization and decaying weighting of tail summary statistics when long-term operational regimes continuously drift over months/years.
3. **Active Model Refitting:** Triggering adaptive distribution refitting upon confirmed structural drift detection events.

---

## 🗓️ Execution Schedule & Next Milestones

```
Phase 1: Paper 1 Manuscript Drafting (LaTeX Template: IEEE / Elsevier) [NOW]
   ├── Section 1: Introduction & Problem Statement (Edge IoT RAM & Non-Gaussianity)
   ├── Section 2: Related Work & SPC Control Chart Baselines
   ├── Section 3: Proposed RBULT-SPC Architecture & Bonferroni FWER Formulation
   ├── Section 4: Empirical Benchmark Results (5 Datasets & TEP Multi-Mode Matrix)
   └── Section 5: Conclusion & Q1 Journal Submission

Phase 2: Paper 2 Concept Drift Framework & Dynamic Refitting [NEXT PROJECT]
   ├── Integrate ADWIN / DDM Drift Engine into online_bootstrap/
   └── Evaluate on Non-Stationary Streaming Benchmarks
```
