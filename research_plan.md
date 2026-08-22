# Research Plan: Memory-Bounded Adaptive Control Chart for Non-Gaussian Industrial Data Streams via RBULT

## 📌 Executive Summary & Target Publication

This research plan formulates a novel application and theoretical extension of **RBULT (Recurring Bootstrap Upper/Lower Tail / Online Chunk Bootstrap)** to solve a fundamental problem in modern industrial engineering and data stream mining: **Dynamic Statistical Process Control (SPC) for IoT Data Streams in Smart Factories**.

### 🎯 Target Q1 Journals
1. **IEEE Transactions on Knowledge and Data Engineering (TKDE)** — *Impact Factor ~8.9, Subject: Data Stream Mining, Algorithmic Guarantees*
2. **Information Sciences (Elsevier)** — *Impact Factor ~8.1, Subject: Information Processing, Stream Analytics*
3. **Expert Systems with Applications (ESWA)** — *Impact Factor ~7.5, Subject: Intelligent Industrial Control, Smart Factory Applications*
4. **Computational Statistics & Data Analysis (CSDA)** — *Impact Factor ~1.8, Subject: Computational Statistics, Resampling Methods*

---

## 1. Problem Statement & Research Gap

### 1.1 Conventional Bootstrap in SPC
In industrial quality control (Smart Manufacturing / Industry 4.0), Statistical Process Control (SPC) charts (e.g., Shewhart $\bar{X}$-chart, $R$-chart, EWMA, CUSUM) are used to monitor sensor telemetry (temperature, vibration, pressure) and detect out-of-control states. When process data is non-Gaussian (e.g., Weibull tool wear, Gamma sensor noise), standard normality assumptions fail. Quality engineers rely on **Conventional Bootstrap** to estimate Lower Control Limits (LCL) and Upper Control Limits (UCL) non-parametrically.

### 1.2 The Three Critical Bottlenecks

```
                  CONVENTIONAL BOOTSTRAP IN IoT STREAMS
+-----------------------------------------------------------------------+
| 1. Memory Bottleneck (O(N * D))                                       |
|    • Requires storing all historical stream observations in RAM.      |
|    • Continuous IoT telemetry causes RAM overflow / Out-Of-Memory.    |
+-----------------------------------------------------------------------+
| 2. Computational Latency Bottleneck (O(B * N * D))                    |
|    • Full resampling across N samples for B iterations is too slow.   |
|    • Fails real-time low-latency response requirement (< 10 ms).      |
+-----------------------------------------------------------------------+
| 3. Outlier Contamination Sensitivity & High-Dimensional Loss          |
|    • Spikes/Sensor noise contaminate bootstrap resamples.             |
|    • Single composite scores (Hotelling T^2 / PCA) lose root-cause    |
|      interpretability (which specific sensor failed?).                |
+-----------------------------------------------------------------------+
```

---

## 2. Proposed Solution: RBULT-SPC Framework

We propose replacing Conventional Bootstrap with **RBULT (Recurring Bootstrap Upper/Lower Tail)** to construct a **Memory-Bounded Adaptive Control Chart** supporting both **Univariate** and **Multivariate Feature-Wise Hyper-Rectangle** monitoring.

### 2.1 Core Architectural Principles

```
  Multivariate IoT Data Stream X_m in R^(k x D) (Chunk m)
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│ Algorithm 4: Feature-wise Z-Score Outlier Filter          │  <-- Suppresses Spikes per sensor
│ Cleans each dimension d in {1, ..., D}                    │
└───────────────────────┬───────────────────────────────────┘
                        │ Clean Chunk X_m^clean
                        ▼
┌───────────────────────────────────────────────────────────┐
│ Parallel RBULT Online Bound Estimators (1 per dimension)  │  <-- O(D) Memory Complexity:
│ Computes [L_d, R_d] for d in {1, ..., D}                  │      Discards old data chunks
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│ D-Dimensional Adaptive Bounding Hyper-rectangle B_m       │  <-- Dynamic Control Limits:
│ B_m = [L_1, R_1] x [L_2, R_2] x ... x [L_D, R_D]          │      LCL_d = L_d, UCL_d = R_d
└───────────────────────────────────────────────────────────┘
```

1. **$O(D)$ Memory Storage Guarantee:** Total memory is strictly linear in the number of dimensions $D$ and constant $O(1)$ with respect to stream length $N$. Old data chunks are discarded immediately after updating boundary vectors $\mathbf{L}_m$ and $\mathbf{R}_m$.
2. **Feature-Level Root-Cause Explainability:** Unlike black-box composite projections, feature-wise bounds $[L_d, R_d]$ immediately pinpoint *which specific sensor* violated its upper/lower threshold.
3. **Outlier Contamination Suppression:** Integrated Z-Score Outlier Detection (Algorithm 4) filters transient sensor spikes per dimension prior to tail expansion, ensuring robust LCL/UCL bounds.

---

## 3. Mathematical Formulation of Multivariate RBULT Control Limits

For an incoming stream chunk of $D$-dimensional vectors $\mathbf{X}_m = \{\mathbf{x}_{m,1}, \mathbf{x}_{m,2}, \dots, \mathbf{x}_{m,k}\} \subset \mathbb{R}^D$ of size $k$:

### 3.1 Feature-wise Outlier Filtering (Algorithm 4)
For each dimension $d \in \{1, 2, \dots, D\}$:
$$\mathbf{X}_{m,d}^{\text{clean}} = \{ x_{d} \in \mathbf{X}_{m,d} \mid |x_{d} - \bar{\mu}_{m,d}| \le \theta \cdot \hat{\sigma}_{m,d} \}$$

### 3.2 Tail Bin Extraction & Adaptive Limits
For each dimension $d$:
1. **Extract Tail Bins:**
   $$\text{Bin}_{\text{left}, d} = \{ x \in \mathbf{X}_{m,d}^{\text{clean}} \mid \bar{\mu}_d - 4\hat{\sigma}_d \le x \le \bar{\mu}_d - 3\hat{\sigma}_d \}$$
   $$\text{Bin}_{\text{right}, d} = \{ x \in \mathbf{X}_{m,d}^{\text{clean}} \mid \bar{\mu}_d + 3\hat{\sigma}_d \le x \le \bar{\mu}_d + 4\hat{\sigma}_d \}$$

2. **Update Dimensional Bounds:**
   $$\text{LCL}_{m,d} = L_{m,d} = \text{Bootstrap}_{\text{online}}(\text{Bin}_{\text{left}, d}, \text{"left"})$$
   $$\text{UCL}_{m,d} = R_{m,d} = \text{Bootstrap}_{\text{online}}(\text{Bin}_{\text{right}, d}, \text{"right"})$$

### 3.3 Streaming Bounding Box Geometry $\mathcal{B}_m \subset \mathbb{R}^D$
The overall process control region is defined as a $D$-dimensional bounding hyper-rectangle:
$$\mathcal{B}_m = \prod_{d=1}^D [L_{m,d}, R_{m,d}] = [L_{m,1}, R_{m,1}] \times [L_{m,2}, R_{m,2}] \times \dots \times [L_{m,D}, R_{m,D}]$$

**Out-of-Control (OOC) Trigger Condition:**
$$\text{Status}(\mathbf{x}_t) = \begin{cases} \text{In-Control}, & \text{if } \mathbf{x}_t \in \mathcal{B}_m \quad (\forall d: L_{m,d} \le x_{t,d} \le R_{m,d}) \\ \text{Out-of-Control (Alarm on Dim } d\text{)}, & \text{if } \exists d \in \{1,\dots,D\} \text{ s.t. } x_{t,d} < L_{m,d} \text{ or } x_{t,d} > R_{m,d} \end{cases}$$

### 3.4 Family-Wise Error Rate (FWER) Adjustment
To maintain a target overall System False Alarm Rate $\alpha_{\text{sys}}$ across $D$ monitored channels, the per-dimension tail probability coverage $\alpha_{\text{dim}}$ is adjusted using **Bonferroni / Šidák Corrections**:
$$\alpha_{\text{dim}} = \frac{\alpha_{\text{sys}}}{D} \quad \text{(Bonferroni Correction)}$$
$$\alpha_{\text{dim}} = 1 - (1 - \alpha_{\text{sys}})^{1/D} \quad \text{(Šidák Correction)}$$

---

### 3.5 Algorithmic Specification & Flowchart for Left and Right Boundary Expansion

#### A. Algorithm Pseudocode (RBULT Left/Right Boundary Expansion Engine)

```text
Algorithm: RBULT Online Boundary Expansion (expand_bt_online)
Input: 
  - Data Chunk: X_m = [x_1, x_2, ..., x_k]
  - Current left/right bounds: L, R (initially +inf / -inf)
  - Hyperparameters: outlier_flag (bool), minmax_boost (bool), nboost (int), dist_list
Output:
  - Updated boundaries L, R and expansion indicator (bool)

1:  if outlier_flag is True then
2:     X_m <- OutlierFilter_ZScore(X_m, threshold=3.0)
3:  end if
4:  x_min <- min(X_m),  x_max <- max(X_m)
5:  expand_min <- False, expand_max <- False

    // Stage 1: Preliminary Boundary Check & Expansion
6:  if x_min < L then
7:     if minmax_boost is True and |min_list| >= nboost then
8:        L_adj <- BootstrapOnline(min_list, direction="left")
9:        if L >= L_adj then L <- L_adj end if
10:    else
11:       L <- x_min
12:    end if
13:    expand_min <- True
14: end if

15: if x_max > R then
16:    if minmax_boost is True and |max_list| >= nboost then
17:       R_adj <- BootstrapOnline(max_list, direction="right")
18:       if R <= R_adj then R <- R_adj end if
19:    else
20:       R <- x_max
21:    end if
22:    expand_max <- True
23: end if

    // Stage 2: Histogram Binning & Theoretical Density Fitting Loop
24: if expand_min is True or expand_max is True then
25:    mu <- (L + R) / 2,  sigma <- (R - L) / 8
26:    Bin_left  <- { x in X_m | mu - 4*sigma <= x <= mu - 3*sigma }
27:    Bin_right <- { x in X_m | mu + 3*sigma <= x <= mu + 4*sigma }
28:    H_obs[0]  <- |Bin_left|,  H_obs[-1] <- |Bin_right|
29:    H_theo    <- FitTheoreticalDistribution(Bin_left, Bin_right, dist_list, total_size)
30:    
31:    Delta_min <- H_obs[0] - H_theo[0]
32:    Delta_max <- H_obs[-1] - H_theo[-1]
33:    dif_expand <- (Delta_min > 0 or Delta_max > 0)

34:    while dif_expand is True do
35:       L_old <- L,  R_old <- R
36:       
          // Right-end Expansion Loop
37:       if Delta_max > 0 and H_obs[-1] >= nboost then
38:          R_tmp <- BootstrapOnline(Bin_right, direction="right")
39:          if R_tmp > R_old then R <- R_tmp end if
40:          if R <= max(Bin_right) then
41:             R_new <- BootstrapOnline(Bin_right, direction="right")
42:             R <- max(R_new, R_old)  // Ensure boundary doesn't shrink
43:          end if
44:       end if

          // Left-end Expansion Loop
45:       if Delta_min > 0 and H_obs[0] >= nboost then
46:          L_tmp <- BootstrapOnline(Bin_left, direction="left")
47:          if L_tmp < L_old then L <- L_tmp end if
48:          if L >= min(Bin_left) then
49:             L_new <- BootstrapOnline(Bin_left, direction="left")
50:             L <- min(L_new, L_old)  // Ensure boundary doesn't shrink
51:          end if
52:       end if

53:       // Recompute Histogram & Check Convergence
54:       mu <- (L + R) / 2,  sigma <- (R - L) / 8
55:       Bin_left  <- { x in X_m | mu - 4*sigma <= x <= mu - 3*sigma }
56:       Bin_right <- { x in X_m | mu + 3*sigma <= x <= mu + 4*sigma }
57:       H_obs[0]  <- |Bin_left|,  H_obs[-1] <- |Bin_right|
58:       
59:       if L_old == L and R_old == R then
60:          dif_expand <- False
61:       else
62:          Delta_min <- H_obs[0] - H_theo[0]
63:          Delta_max <- H_obs[-1] - H_theo[-1]
64:       end if
65:    end while
66: end if
67: return Updated Boundaries (L, R)
```

#### B. Mermaid Flowchart Diagram

```mermaid
flowchart TD
    A[New Data Chunk X_m] --> B{Outlier Detection Enabled?}
    B -- Yes --> C[Filter Outliers via Z-Score]
    B -- No --> D[Extract x_min and x_max]
    C --> D
    
    D --> E{x_min < L ?}
    E -- Yes --> F[Expand Left Boundary L]
    E -- No --> G{x_max > R ?}
    
    F --> G
    G -- Yes --> H[Expand Right Boundary R]
    G -- No --> I{Any Boundary Changed?}
    H --> I
    
    I -- No --> J[Keep Current L, R]
    I -- Yes --> K[Compute Mean mu & Std sigma]
    K --> L[Extract Tail Bins: Bin_left & Bin_right]
    L --> M[Compute Observed H_obs & Theoretical H_theo]
    M --> N[Calculate Tail Excess Delta_min & Delta_max]
    
    N --> O{Delta_min > 0 OR Delta_max > 0?}
    O -- No --> P[Finalize Boundaries L, R]
    O -- Yes --> Q[Save L_old = L, R_old = R]
    
    Q --> R{Delta_max > 0 & |Bin_right| >= nboost?}
    R -- Yes --> S[Bootstrap Right Tail -> Update R]
    R -- No --> T{Delta_min > 0 & |Bin_left| >= nboost?}
    S --> T
    
    T -- Yes --> U[Bootstrap Left Tail -> Update L]
    T -- No --> V[Recompute mu, sigma & Bin Counts]
    U --> V
    
    V --> W{L == L_old AND R == R_old ?<br>Convergence Check}
    W -- Converged (Yes) --> P
    W -- Not Converged (No) --> N
```

---

## 4. Key Contributions & Novelty for Q1 Reviewers

1. **First $O(D)$-Memory Multivariate Non-Parametric Control Chart:** Demonstrates that exact non-Gaussian control limits across $D$ dimensions can be updated continuously with constant memory complexity $O(D)$ regardless of stream length $N$.
2. **Exact Feature-Level Root-Cause Diagnosis:** Provides individual $[L_d, R_d]$ bounds per sensor, enabling immediate identification of malfunctioning physical components without black-box projection loss.
3. **Robustness Against Stream Spikes:** Combines online distribution fitting with local Z-score outlier filtering per channel, eliminating false alarm cascades caused by corrupted sensor telemetry.
4. **Theoretical & Empirical Validation:** Rigorous proofs and empirical benchmarks proving that RBULT achieves control limit convergence equivalent to full-history bootstrap while reducing memory usage by $>99\%$.

---

## 5. Public Benchmark Datasets (Kaggle & Industrial Gold Standards)

สำหรับโจทย์ **"Memory-Bounded Adaptive Control Chart for Non-Gaussian Industrial Data Streams"** มีชุดข้อมูลเปิดสาธารณะ (Public/Open Datasets) จากทั้ง Kaggle, UCI Machine Learning Repository และคลังข้อมูลมาตรฐานทางวิศวกรรมอุตสาหการ ที่นิยมใช้เป็น Benchmark ในวารสารระดับ **Q1** ดังนี้:

### 5.1 Kaggle Datasets (ดาวน์โหลดง่าย มี Benchmark ชุมชน)

* **AI4I 2020 Predictive Maintenance Dataset (Kaggle / UCI):**
  * **ลักษณะข้อมูล:** ข้อมูลการทำงานของเครื่องจักรสังเคราะห์ที่สะท้อนสภาวะจริงในอุตสาหกรรม ประกอบด้วยค่าเซนเซอร์ เช่น อุณหภูมิอากาศ, อุณหภูมิกระบวนการ, ความเร็วรอบ (Rotational speed), แรงบิด (Torque) และการสึกหรอของเครื่องมือ (Tool wear)
  * **ความเหมาะสมกับ RBULT:** ค่าแรงบิดและความเร็วรอบมักมีการแจกแจงที่ไม่เป็น Normal Distribution (Non-Gaussian) และมีรูปแบบความล้มเหลวหลายประเภท (Failure modes) เหมาะอย่างยิ่งสำหรับการจำลองข้อมูลไหลเข้าแบบ Streaming เพื่อทดสอบการสร้าง Control Limits

* **MetroPT-3 Dataset (Kaggle):**
  * **ลักษณะข้อมูล:** สัญญาณเซนเซอร์อนุกรมเวลา (Time Series) จากระบบอัดอากาศของรถไฟใต้ดิน (Metro Train Compressor) บันทึกทุก 1 วินาที เช่น แรงดัน (Pressure), อุณหภูมิ, กระแสไฟฟ้า และอัตราการไหล
  * **ความเหมาะสมกับ RBULT:** ปริมาณข้อมูลมีขนาดใหญ่มากและต่อเนื่อง เหมาะอย่างยิ่งสำหรับการจำลองสถานการณ์ Memory-Bounded ที่ต้อง Discard ข้อมูลเก่าทิ้งและรักษาเฉพาะช่วง $[L, R]$

* **Pump Sensor Data (Kaggle):**
  * **ลักษณะข้อมูล:** ข้อมูลจากเซนเซอร์ 52 ตัวที่ติดตั้งบนปั๊มน้ำขนาดใหญ่ในโรงงาน พร้อมป้ายกำกับสถานะ Normal, Broken, และ Recovering
  * **ความเหมาะสมกับ RBULT:** สามารถใช้ทดสอบ Algorithm 4 ในการกรอง Noise/Spikes ช่วงปั๊มทำงานปกติ และทดสอบว่าเมื่อระบบเริ่มเข้าสู่สถานะผิดปกติ ค่าเซนเซอร์จะทะลุขอบเขต Upper-Lower Bounds ที่ RBULT ประมาณค่าไว้หรือไม่

---

### 5.2 Gold Standard Industrial SPC Benchmark Datasets

* **Tennessee Eastman Process (TEP) Benchmark:**
  * **แหล่งดาวน์โหลด:** มีให้ดาวน์โหลดทั้งบน Kaggle (`Tennessee Eastman Process Simulation Data`) และ GitHub Repository
  * **ลักษณะข้อมูล:** เป็นชุดข้อมูลมาตรฐานอันดับหนึ่งของโลกในงานวิจัยด้าน Process Monitoring และ Statistical Process Control จำลองกระบวนการผลิตทางเคมีที่มี 52 ตัวแปร และมีสถานการณ์ความผิดปกติ (Disturbances) 20 รูปแบบ
  * **ความเหมาะสมกับ RBULT:** ตัวแปรทางเคมีส่วนใหญ่มีพฤติกรรมแบบ Non-Gaussian และ Non-linear การนำ RBULT ไปสร้าง Adaptive Control Chart บน TEP จะมีน้ำหนักและความน่าเชื่อถือสูงมากสำหรับ Reviewer

* **SECOM (Semiconductor Manufacturing) Dataset (UCI / Kaggle):**
  * **ลักษณะข้อมูล:** ข้อมูลเซนเซอร์ 590 สัญญาณจากกระบวนการผลิตเซมิคอนดักเตอร์
  * **ความเหมาะสมกับ RBULT:** ข้อมูลมีสัญญาณรบกวนสูงมาก มีค่าสูญหาย และมีสัดส่วนของเสียต่ำ (Imbalanced / Contaminated) เหมาะสำหรับทดสอบความทนทานต่อ Outliers และการประมาณค่าความแปรปรวนในมิติสูง

---

### 5.3 Streaming Setup & Workflow Architecture

| ขั้นตอน | การดำเนินการ |
| --- | --- |
| **1. Stream Simulation** | โหลดไฟล์ CSV แล้วจำลองการป้อนข้อมูลทีละ Chunk เช่น ขนาด Chunk ละ 50–200 ตัวอย่าง ตามลำดับเวลา (Timestamp) |
| **2. Control Limit Initialization** | ใช้ Chunk แรกในการฟิต Histogram และกำหนดขอบเขตเริ่มต้น ($\text{LCL}_{0,d} = L_d, \text{UCL}_{0,d} = R_d$) แยกตามมิติ $d \in \{1,\dots,D\}$ |
| **3. Stream Monitoring** | เมื่อ Chunk ถัดไปเข้ามา ตรวจสอบว่าสัญญาณหลุด $\mathcal{B}_m$ หรือไม่ หากเป็นสัญญาณการเปลี่ยนแปลงของกระบวนการ ให้ทำการ Recursive Tail-Bootstrapping เพื่อขยายขอบเขต |
| **4. Performance Metrics** | วัดค่า **ARL (Average Run Length)**, **False Alarm Rate (FAR)**, **Detection Delay**, และ **Memory Footprint (KB)** |

---

### 5.4 Quantitative Baseline Comparison Matrix

| Baseline Method | Memory Complexity | Non-Gaussian Support | Robust to Noise | Root-Cause Interpretability | Target Metric |
|---|---|---|---|---|---|
| Shewhart $\bar{X}$ / $R$ Chart | $O(D)$ | ❌ (Gaussian only) | ❌ Poor | ✅ High | Standard baseline |
| EWMA / CUSUM | $O(D)$ | ⚠️ Moderate | ❌ Poor | ✅ High | Dynamic baseline |
| Conventional Full-History Bootstrap | $O(N \cdot D)$ (OOM risk) | ✅ Excellent | ❌ Poor | ✅ High | Resampling benchmark |
| Online KDE / Hotelling $T^2$ | $O(N \cdot D)$ | ✅ Excellent | ⚠️ Moderate | ❌ Low (Composite loss) | Dimensional reduction baseline |
| **Proposed RBULT-SPC** | **$O(D)$** | **✅ Excellent** | **✅ High (Algorithm 4)** | **✅ High (Per-dimension $[L_d, R_d]$)** | **Target Method** |

---

## 6. Project Implementation Roadmap

```
Phase 1: SPC Engine Module Development (src/spc_rbult.py)
   ├── Implement RBULT_ControlChart class extending BootstrapOnline
   └── Integrate real-time LCL/UCL evaluation pipeline per dimension d in {1, ..., D}

Phase 2: Benchmark Experiment Suite (experiments/exp_spc_benchmark.py)
   ├── Run AI4I 2020 & TEP datasets across baselines
   └── Measure ARL0, ARL1, FAR, RAM usage, and Latency

Phase 3: Visualization & Result Generation (experiments/plot_spc_charts.py)
   ├── Generate Control Chart comparison plots (LCL/UCL bounds vs time per sensor)
   └── Produce LaTeX performance tables for manuscript

Phase 4: Manuscript Preparation (paper.tex)
   └── Draft manuscript following IEEE TKDE / ESWA formatting guidelines
```

---

## 7. Practical Execution Guide for New Datasets (ขั้นตอนการนำข้อมูลใหม่มาทดสอบ)

### รูปแบบที่ 1: Real-time Streaming Integration (Multivariate Python Call)
เหมาะสำหรับการประมวลผล Real-time บน IoT Telemetry หลายมิติ:

```python
from online_bootstrap import BootstrapOnline, ResBootstrap

# 1. กำหนดเซนเซอร์ 3 ตัว (เช่น Torque, Speed, Temperature)
sensors = ['Torque', 'Speed', 'Temperature']
engines = {s: BootstrapOnline() for s in sensors}

for s in sensors:
    engines[s].set_online(minmax_flag=False)

# 2. จำลอง Data Stream ไหลเข้าทีละ Chunk
multivariate_stream = [
    {'Torque': [10.2, 9.8, 10.5], 'Speed': [1500, 1520, 1490], 'Temperature': [45.1, 45.3, 45.0]},
    {'Torque': [10.8, 12.1, 9.9], 'Speed': [1510, 1480, 1505], 'Temperature': [45.2, 45.4, 45.1]},
]

for i, chunk in enumerate(multivariate_stream):
    print(f"--- Chunk {i+1} ---")
    for s in sensors:
        chunk_data = chunk[s]
        is_expanded = engines[s].expand_bt_online(chunk_data, outlier=True)
        print(f"  [{s}] LCL={engines[s].exp_l:.2f}, UCL={engines[s].exp_r:.2f}, Expanded={is_expanded}")
```

### รูปแบบที่ 2: Batch Benchmark Experiments (`main_boostrap_v2.py`)
1. **เตรียม JSON Data Chunk per Feature:** แยกไฟล์ JSON ตามแต่ละตัวแปร หรือรันวนลูปประมวลผลทีละมิติ
2. **รันคำสั่ง CLI ผ่าน Terminal:**
   ```bash
   conda activate ./.conda
   python main_boostrap_v2.py --dir config_sim_data --file my_config.yaml --outlier
   ```
