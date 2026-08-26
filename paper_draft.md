# Unified Manuscript Blueprint: Integrated Second Draft for *Information Sciences*

> **Target Journal:** *Information Sciences* (Elsevier, SJR Q1, Impact Factor ~8.1)  
> **Article Type:** Full Length Research Paper  
> **Manuscript Title:** *Resource-Bounded Non-Parametric Bootstrap Control Chart for High-Dimensional Non-Gaussian Data Streams via RBULT*  
> **Authors:** P. Junsawamg, S. Phimoltares, C. Lursinsap  

---

## 🎯 The Unified Master Storyline & Narrative Flow

To achieve seamless coherence between **Draft 1 (`boostrap_expert_wo_authors.pdf`)** and **`research_plan.md`**, the manuscript follows a natural **Conceptual Evolution**:

```
[1. Mathematical & 1D Foundations]       ──►  [2. High-Dimensional SPC Extension]    ──►  [3. Unified Two-Tier Evaluation]
- 1D Range Estimation Problem                 - Algorithm 4 Outlier Spike Filter              - Tier 1: 1D Simulated & Economic Datasets
- Theorems 1-2 & Corollary 1 Proofs           - 11-Candidate MLE Tail Fitting                 - Tier 2: 5 Real Industrial Telemetry Streams
- Core RBULT Tail Resampling Concept          - Bonferroni FWER Tail Scaling ($D=34$)         - Tier 2 Stress: TEP Multi-Mode (Modes 1,3,4,5)
```

---

## 📝 Detailed Section-by-Section Integrated Manuscript Blueprint

### Highlights
- We propose **RBULT-SPC**, a novel resource-bounded non-parametric control chart framework for high-dimensional non-Gaussian data streams.
- Grounded in dual-protocol streaming evaluation: **In-Sample Adaptation Mode** and **One-Step-Ahead Pre-Sequential Predictive Mode** (IEEE TKDE gold standard).
- Rigorously stress-tested across a **7-Scenario Gold Standard Suite** (Clean, GAWN $0.1\sigma - 0.3\sigma$, and Impulse Spikes $1\% - 10\%$), achieving 100% transient glitch absorption at 1% spikes ($\text{Coverage} = 95.29\%$, $\sigma_L = 0.05$).
- Achieves strict $O(D)$ constant RAM memory boundedness (**0.52 KB – 3.23 KB**), delivering a **$180\times$ memory reduction** over conventional sliding-window bootstrap ($582.87\text{ KB}$).
- Integrates Algorithm 4 Z-score spike filtering with dynamic 11-candidate distribution MLE fitting to prevent control limit pollution.
- Eliminates batch false-alarm spam (**Chunk FAR = 0.00%**) on high-dimensional streams ($D=34$) while retaining an immediate anomaly detection response delay ($\text{ARL}_1 = 1.00$).
- Validated across a comprehensive two-tier benchmark: 1,260 1D benchmark runs across 5 distributions and 7 scenarios, 4 economic datasets, 5 real industrial streams, and 4 TEP stress regimes (Modes 1, 3, 4, 5) up to 1.74M observations.

---

### Abstract
Bootstrapping is a powerful non-parametric technique for quantifying uncertainty without restrictive distributional assumptions. However, applying conventional bootstrap methods to real-time industrial data streams poses two fundamental challenges: prohibitive memory overhead ($O(N \cdot D)$ storage growth) and susceptibility to control limit pollution when anomalous samples contaminate historical memory buffers. This paper presents **RBULT-SPC**, a resource-bounded non-parametric control chart framework designed for high-dimensional, non-Gaussian IoT data streams. Grounded in mathematical proofs of chunk extreme probabilities, RBULT-SPC maintains only compact tail summary statistics in $O(D)$ constant space, preventing memory overflow and eliminating buffer pollution. To handle high dimensionality ($D=34$), the framework combines Z-score outlier filtering, lazy boundary expansion, 11-candidate non-parametric distribution MLE fitting, and Bonferroni Family-Wise Error Rate (FWER) tail adjustment. Extensive two-tier empirical evaluations evaluated under both **In-Sample Adaptation** and **One-Step-Ahead Pre-Sequential Predictive** protocols across a **7-Scenario Gold Standard Suite** (Clean, GAWN continuous noise $0.1\sigma - 0.3\sigma$, and Impulse Spikes $1\% - 10\%$), 4 economic datasets, 5 industrial streams, and 4 operating regimes of the Tennessee Eastman Process (TEP) demonstrate that RBULT-SPC achieves optimal coverage ($95.29\% - 99.95\%$) and controlled false alarm rates ($0.05\% - 3.27\%$), outperforming classical Shewhart and EWMA charts whose false alarm rates collapse to $18.21\% - 48.94\%$ under non-Gaussianity and operational throughput stress. Furthermore, hyperparameter sensitivity analysis demonstrates that dimension-aware thresholding completely eliminates batch false alarms ($\text{Chunk FAR} = 0.00\%$) while maintaining an immediate failure detection response delay ($\text{ARL}_1 = 1.00$) at an average latency under 45 ms.

---

### Section 1: Introduction (The Conceptual Evolution Narrative)
- **1.1 Industrial Context:** Smart factory IoT telemetry, Industry 4.0 sensor streams, real-time quality control.
- **1.2 Statistical & Computational Bottlenecks:**
  - Non-Gaussianity & Heavy Tails: Real sensor signals violate Gaussian normality, causing parametric charts (Shewhart, EWMA) to collapse.
  - Memory Overhead & Limit Pollution: Sliding-window bootstrap requires $O(W \cdot D)$ RAM ($582\text{ KB}$) and suffers limit distortion when fault samples enter the buffer.
  - High-Dimensional Alarm Spam: Monitoring $D=34$ channels simultaneously leads to severe batch false-alarm spam without FWER adjustment.
- **1.3 Conceptual Evolution from Draft 1 to Second Draft:** 
  - Explaining how the core RBULT concept (originally developed for 1D population range estimation in `boostrap_expert_wo_authors.pdf`) is theoretically extended into a multivariate Statistical Process Control (SPC) framework for high-dimensional streaming data.
- **1.4 Summary of Contributions:**
  - Mathematical proof of chunk excursion bounds (Theorems 1–2, Corollary 1) and multivariate FWER control (Theorem 3).
  - $O(D)$ Bounded RAM Architecture (**3.23 KB**, $180\times$ savings).
  - 4-Stage Statistical Pipeline (Spike Filter, Lazy Expansion, MLE Candidate Fitting, Bonferroni Adjustment).
  - Unified Two-Tier Benchmark (10 1D distributions, 4 economic datasets, 5 industrial streams, TEP Modes 1, 3, 4, 5, and Sensitivity Analysis).

---

### Section 2: Related Work & Literature Review
- **2.1 Classical & Multivariate SPC Control Charts:** Shewhart X-Bar, EWMA, CUSUM, Hotelling's $T^2$ — mathematical breakdown under non-normality and non-stationary noise.
- **2.2 Bootstrap Resampling in Streaming Environments:** Efron's percentile bootstrap, blockwise bootstrap, online bootstrap for SGD — high memory footprint and vulnerability to buffer contamination.
- **2.3 Uncertainty Quantification & Outlier Filtering in IoT Data Streams:** Review of range estimation, hyper-ellipsoid bounds, and z-score spike filtering.

---

### Section 3: Theoretical Foundations & 1D Range Estimation (Part I - From Draft 1 PDF)
- **3.1 Streaming Data Model & Discard-After-Learn Constraint:** Chunks $c_i \in \mathbb{R}^{m \times D}$ arriving over time $t=1, 2, \dots$
- **3.2 Univariate Population Range Estimation ($[L, R]$):** Formulation of left-end ($L$) and right-end ($R$) boundary expansion.
- **3.3 Mathematical Analysis of Extreme Value Excursion:**
  - **Theorem 1 (First Chunk Extreme Probability):** Proof of joint minimum/maximum occurrence probability in initial chunk:
    $$P(E_1) = \frac{l(l-1)}{N(N-1)}$$
  - **Theorem 2 (Streaming Chunk Distribution):** Proof of uniform extreme occurrence probability across streaming chunks for $i \ge 1$:
    $$P(E_i) = \frac{l(l-1)}{N(N-1)}$$
  - **Corollary 1 (Expected Chunk Index):** Derivation of expected chunk index $\mathbb{E}[X]$ for initial boundary expansion:
    $$\mathbb{E}[X] = \frac{1}{2} \left( \frac{l(l-1) L (L+1)}{N(N-1)} \right)$$

---

### Section 4: Proposed High-Dimensional RBULT-SPC Framework (Part II - From research_plan.md)
- **4.1 System Architecture:** 4-Stage streaming pipeline.
- **4.2 Algorithm 1: First Chunk Initial Fitting & Tail Bin Partitioning**
- **4.3 Algorithm 2: Streaming Chunk Update & Lazy Boundary Expansion**
- **4.4 Algorithm 3: Recurrent Tail Resampling & Quantile Interpolation**
- **4.5 Algorithm 4: Z-Score Outlier Spike Filtering:** Suppression of contaminated sensor noise to prevent control limit pollution.
- **4.6 Bonferroni / Šidák FWER Multi-Dimensional Scaling:** Tail boundary adjustment $\alpha_{\text{dim}} = \frac{\alpha_{\text{sys}}}{D}$ for $D$-dimensional feature spaces.
- **4.6.1 Mathematical Evolution of Control Limits (1D Range vs. Dimension-Aware LCL/UCL):**

#### 📐 Formula Comparison: Draft 1 (1D Range) vs. Draft 2 (Multivariate RBULT-SPC)

| Mathematical Dimension | Draft 1 (Univariate 1D Baseline in PDF) | Draft 2 (Multivariate RBULT-SPC in Second Draft) |
|---|---|---|
| **Target Data Dimension** | Single Scalar Feature ($D = 1$) | **Multivariate Sensor Stream ($D \ge 1$, e.g., $D=34$)** |
| **Tail Quantile Adjustment** | Univariate Tail Probability $\alpha$ ($D=1 \Rightarrow \frac{\alpha}{1} = \alpha$) | **Bonferroni FWER Scaled Quantile: $\alpha_{\text{dim}} = \frac{\alpha_{\text{sys}}}{D}$** |
| **Lower Control Limit ($\text{LCL}_d$)** | $L = \text{Percentile}_{\text{left}}\left(\frac{\alpha}{2}\right)$ | $$\mathbf{\text{LCL}_d = L_d = Q_{\text{tail}}\left(\frac{\alpha_{\text{sys}}}{2D}\right)}$$ |
| **Upper Control Limit ($\text{UCL}_d$)** | $R = \text{Percentile}_{\text{right}}\left(1 - \frac{\alpha}{2}\right)$ | $$\mathbf{\text{UCL}_d = R_d = Q_{\text{tail}}\left(1 - \frac{\alpha_{\text{sys}}}{2D}\right)}$$ |
| **Outlier & Noise Defense** | Basic z-score outlier elimination | **Algorithm 4 Z-Score Filter integrated prior to $Q_{\text{tail}}$ calculation** |

#### 💡 Key Scientific Rationale for the Mathematical Extension:
1. **Univariate ($D=1$) vs. Multivariate ($D=34$) Scaling:** In Draft 1, all 10 simulated distributions and 4 Kaggle datasets were 1-dimensional ($D=1$), so the tail quantile scaling factor was $\frac{\alpha}{1} = \alpha$. In Draft 2, expanding to $D=34$ sensor channels requires dividing by $D$ ($\alpha_{\text{dim}} = \frac{\alpha_{\text{sys}}}{D}$) to constrain system-wide false alarms to $\le \alpha_{\text{sys}} = 5\%$, preventing an explosion of false alarm spam ($1 - (1-0.05)^{34} \approx 82.5\%$).
2. **Eliminating Control Limit Pollution:** Integrating Algorithm 4 Z-score filtering before computing $Q_{\text{tail}}$ ensures that transient sensor spikes or long-duration faults do not distort $L_d$ and $R_d$.

- **4.7 Theorem 3 (Family-Wise Error Rate Bound):** Formal proof of FWER upper bound control under Bonferroni tail adjustment in $D$-dimensional non-Gaussian space.
- **4.8 Complexity Analysis:**
  - Space Complexity: Strictly $O(D)$ bounded RAM ($3.23\text{ KB}$).
  - Time Complexity: Amortized $O(1)$ per sample ($< 45\text{ ms}$ per chunk).

---

### Section 5: Two-Tier Empirical Benchmark Experiments
- **5.1 Experimental Setup & Evaluation Metrics:** Range Error, Coverage Rate (%), Sample FAR (%), Chunk FAR (%), ARL0, ARL1 (Detection Delay), Peak RAM (KB), Latency per Chunk (ms).
- **5.2 Tier 1: Univariate Range Approximation & Dual-Protocol Noise Sensitivity Benchmarks:**
  - Evaluates 1,260 simulation runs across 5 synthetic distributions ($F(5,10)$, Uniform, Wald, Gamma, Normal), 3 chunk sizes (50, 100, 500), 2 target alphas ($\alpha=0.05, 0.01$), 3 methods, and 2 protocols.
  - **Dual Evaluation Protocol:** 
    1. *In-Sample Adaptation Protocol:* Measures post-update boundary enclosure precision.
    2. *One-Step-Ahead Pre-Sequential Protocol (IEEE TKDE Gold Standard):* Measures predictive violation rates using bounds from chunk $m-1$ prior to updating chunk $m$.
  - **7-Scenario Gold Standard Suite:**
    * *Scenario A (Clean Stream):* Baseline pure stream ($\text{Coverage} = 95.41\%$, $\bar{W} = 23.65$, $\text{Latency} = 4.45\text{ ms}$).
    * *Scenarios B1–B3 (GAWN $0.1\sigma, 0.2\sigma, 0.3\sigma$):* Scale-invariant continuous noise adaptation ($\text{Coverage} = 95.34\% - 95.74\%$, $\text{NSR} = 1.05 - 1.21$).
    * *Scenario C1 (1% Spikes):* Perfect transient glitch absorption ($\text{Coverage} = 95.29\%$, $\bar{W} = 24.22$, $\sigma_L = 0.05$).
    * *Scenarios C2–C3 (5% & 10% Spikes):* Extreme stress resilience preventing memory explosion and boundary collapse ($\text{Coverage} = 97.70\% - 99.44\%$, $\sigma_L = 0.34$).
  - 4 Real-World Economic Datasets (Laptop Prices, Electronic Sales, E-Commerce Sales, World Tourism Economy).
- **5.3 Tier 2: Real-World Industrial Streaming Benchmarks (From research_plan.md):**
  - AI4I 2020 Predictive Maintenance ($N=10,000, D=5$)
  - MetroPT-3 Air Compressor ($N=1,516,948, D=7$)
  - Large Industrial Pump Maintenance ($N=20,000, D=5$)
  - Water Pump Sensor Dataset ($N=220,320, D=10$)
- **5.4 Tier 2 Stress Benchmark: Tennessee Eastman Process (TEP Multi-Mode Matrix):**
  - Mode 1: Nominal Operating Conditions (50/50 Mass Ratio, Nominal Throughput)
  - Mode 3: Chemical Feed Skewness (90/10 Mass Ratio)
  - Mode 4: Operational Throughput Stress (50/50 Mass Ratio, Max Production Rate)
  - Mode 5: Combined Extreme Stress (10/90 Mass Ratio, Max Production Rate)
- **5.5 Hyperparameter Sensitivity Study (`ooc_threshold_count` $\in \{5, 10, 15\}$):** Demonstration of zero false alarm spam ($\text{Chunk FAR} = 0.00\%$) and immediate detection ($\text{ARL}_1 = 1.00$).
- **5.6 Resource & Latency Trade-off Analysis:** $180\times$ RAM savings vs. $<45\text{ ms}$ real-time execution.

---

### Section 6: Discussion & Data Taxonomy Insights
- **6.1 Data Taxonomy Synthesis:** Performance across Skewed/Heavy-Tailed, Uniform Density, High Throughput Stress, and High Dimensionality ($D=34$).
- **6.2 Methodological Synthesis:** Comparing Tier 1 range error accuracy with Tier 2 multivariate SPC control chart coverage.
- **6.3 Practical Engineering Guidelines for Smart Factory IoT Implementations.**

---

### Section 7: Conclusion & Future Scope
- **7.1 Summary of Scientific Breakthroughs.**
- **7.2 Roadmap for Paper 2:** Integrating Concept Drift Detection (ADWIN/DDM) and Dynamic Model Refitting for long-term non-stationary streams.

---

## 📊 Summary Table of Data Coverage in Unified Manuscript

| Tier / Level | Dataset Name | Sample Size ($N$) | Dimensions ($D$) | Data Type & Distribution | Source |
|---|---|:---:|:---:|---|---|
| **Tier 1** | 1D Benchmark (7-Scenario Suite) | 1,260 Runs ($N=10,000$) | 1 | $F$, Uniform, Wald, Gamma, Normal | Section 10 & PDF |
| **Tier 1** | 4 Economic Kaggle Datasets | 1,000 - 10,000 | 1 | Laptop, Electronic, E-Com, Tourism | Draft 1 PDF |
| **Tier 2** | AI4I 2020 Predictive Maint. | 10,000 | 5 | Mechanical Tool Wear Sensor | `research_plan.md` |
| **Tier 2** | MetroPT-3 Air Compressor | 1,516,948 | 7 | Heavy-Tailed Pneumatic Stream | `research_plan.md` |
| **Tier 2** | Large Industrial Pump | 20,000 | 5 | Uniform Density Maintenance | `research_plan.md` |
| **Tier 2** | Water Pump (`sensor.csv`) | 220,320 | 10 | Multi-Sensor Industrial Stream | `research_plan.md` |
| **Tier 2** | TEP Mode 1 (Nominal) | 1,740,000 | 34 | Chemical Plant Nominal Stream | `research_plan.md` |
| **Tier 2** | TEP Mode 3 (Feed Skew 90/10) | 1,739,400 | 34 | Chemical Feed Skewness Stream | `research_plan.md` |
| **Tier 2** | TEP Mode 4 (Max Production Rate) | 1,719,000 | 34 | Operational Throughput Stress | `research_plan.md` |
| **Tier 2** | TEP Mode 5 (Combined Stress) | 1,729,800 | 34 | Combined Extreme Stress Stream | `research_plan.md` |

---

## 💡 Additional Strategic Comments & Reviewer Framing Notes

### 🎯 1. Rationale for Two-Tier Experimental Hierarchy (Tier 1 vs. Tier 2)
Structuring the experimental evaluation into **Tier 1** (Univariate Controlled Benchmarks) and **Tier 2** (Multivariate Streaming SPC Benchmarks) is a standard high-impact strategy for top Q1 journals (*Information Sciences*, *IEEE TKDE*):

- **Satisfying Theoretical Reviewers (Tier 1):** Validates the fundamental RBULT tail-fitting engine ($[L, R]$ range estimation accuracy) under 10 controlled 1D synthetic distributions and 4 economic datasets where mathematical ground-truth bounds are known.
- **Satisfying Applied/Systems Reviewers (Tier 2):** Validates the operational streaming process control performance (Coverage %, Sample FAR %, Chunk FAR %, ARL1, Memory KB, and Latency ms) on 5 real industrial telemetry streams and 4 TEP stress regimes ($D=34$, up to $1.74\text{M}$ observations).
- **Preventing Common Reviewer Objections:** Eliminates common rejection risks such as *"lack of ground-truth controlled synthetic validation"* or *"lack of high-dimensional real-time streaming applicability"*.

---

### ✍️ 2. Recommended Framing Passage for Section 5.1 (Experimental Setup)
Insert this bridging text at the start of Section 5.1 to guide reviewers seamlessly from Tier 1 to Tier 2:

> *"To rigorously evaluate the proposed RBULT-SPC framework from both fundamental estimation accuracy and operational streaming process control perspectives, our empirical study is structured into a progressive two-tier experimental hierarchy:*
> 
> * **Tier 1 (Base Estimation Accuracy Benchmark):** Evaluates the fundamental range estimation accuracy of the RBULT engine ($pop\_range - est\_range$) across 10 controlled 1D synthetic distributions and 4 economic datasets where theoretical ground-truth bounds are known.
> * **Tier 2 (Multivariate Streaming SPC Benchmark):** Evaluates operational process control metrics (Coverage Rate %, Sample FAR %, Chunk FAR %, ARL1, Peak RAM, and Latency) across 5 real-world industrial telemetry streams and 4 Tennessee Eastman Process stress regimes ($D=34$, up to 1.74M observations)."*

---

### 📐 3. Mathematical Parameter Evolution Note ($\alpha$ & Bonferroni FWER Scaling)
- **Draft 1 Baseline ($D=1$):** Evaluated single scalar features where the tail percentile parameter was unscaled ($\alpha_{\text{dim}} = \frac{\alpha}{1} = \alpha$).
- **Draft 2 Extension ($D \ge 1$):** Extends the theoretical formulation to multivariate streaming control limits by introducing Bonferroni FWER tail scaling ($\alpha_{\text{dim}} = \frac{\alpha_{\text{sys}}}{D}$). This constrains system-wide false alarms to $\le \alpha_{\text{sys}} = 5\%$, preventing an explosion of false alarm spam ($1 - (1-0.05)^{34} \approx 82.5\%$).

