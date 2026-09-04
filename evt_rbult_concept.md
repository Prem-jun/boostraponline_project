# Extreme Value Theory (EVT) สำหรับการประมาณขอบเขตข้อมูล และความเชื่อมโยงกับ RBULT-SPC

> บันทึกแนวคิดจากการสนทนา — 3 กันยายน 2569

---

## 1. Extreme Value Theory (EVT) คืออะไร

EVT เป็นสาขาของสถิติที่ศึกษาพฤติกรรมของ **หางการแจกแจง (tail of distribution)** โดยเฉพาะ ไม่ใช่ค่าเฉลี่ยหรือกลางๆ แต่เป็น "เหตุการณ์รุนแรงและหายาก" เช่น

- น้ำท่วมครั้งใหญ่ที่สุดในรอบ 100 ปี
- ความผิดพลาดของ sensor ที่รุนแรงผิดปกติ
- ความสูญเสียทางการเงินในระดับ extreme

ทฤษฎีหลักคือ **Fisher-Tippett-Gnedenko Theorem** ซึ่งบอกว่าไม่ว่าข้อมูลต้นฉบับจะมาจากการแจกแจงแบบไหนก็ตาม ค่า maximum ของ sample ขนาดใหญ่จะ converge ไปหา **Generalized Extreme Value (GEV) Distribution** เสมอ ซึ่งเปรียบได้กับ "Central Limit Theorem ของหางการแจกแจง"

---

## 2. สองแนวทางหลักของ EVT

### แนวทางที่ 1: Block Maxima → GEV Distribution

แบ่งข้อมูลเป็น block ขนาด $n$ แล้วเก็บค่า max (และ min) ของแต่ละ block จากนั้น fit **Generalized Extreme Value (GEV) Distribution** กับค่า max เหล่านั้น

$$G(z) = \exp\left\{-\left[1 + \xi\left(\frac{z - \mu}{\sigma}\right)\right]^{-1/\xi}\right\}$$

โดย shape parameter $\xi$ กำหนดลักษณะของหาง:

| $\xi$ | ประเภท | ลักษณะ |
|:---:|---|---|
| $\xi > 0$ | Fréchet | Heavy tail |
| $\xi = 0$ | Gumbel | Light tail |
| $\xi < 0$ | Weibull type | Bounded tail |

ประมาณขอบเขตด้วย quantile เช่น $\text{UCL} = G^{-1}(0.99)$

**ข้อเสีย:** ต้องรอให้ครบ block ก่อนถึงจะ update ได้ และสูญเสียข้อมูลส่วนใหญ่ในแต่ละ block ไม่เหมาะกับ streaming

---

### แนวทางที่ 2: Peaks Over Threshold (POT) → GPD ✅ (แนะนำสำหรับงาน RBULT)

เลือก threshold $u$ แล้วเอาเฉพาะข้อมูลที่ **เกิน** $u$ มา fit **Generalized Pareto Distribution (GPD)**

$$H(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}, \quad y = x - u > 0$$

**ขั้นตอน:**
1. กำหนด threshold $u$ (เช่น $\mu + 3\sigma$ สำหรับ upper tail)
2. เก็บเฉพาะ exceedances $y_i = x_i - u$
3. Fit GPD ด้วย MLE เพื่อได้ $\hat{\xi}$ และ $\hat{\sigma}$
4. คำนวณ UCL ที่ coverage level $p$:

$$\text{UCL} = u + \frac{\hat{\sigma}}{\hat{\xi}}\left[(1-p)^{-\hat{\xi}} - 1\right]$$

ทำแบบเดียวกันกับ lower tail ด้วย $-x$ เพื่อหา LCL

---

## 3. ความเชื่อมโยงกับ RBULT-SPC

### 3.1 สิ่งที่ RBULT ทำอยู่ในปัจจุบัน (Bootstrap-based)

```
Tail Bin ← ข้อมูลในช่วง [μ+3σ, μ+4σ]
         ↓
    Bootstrap Resample จาก Tail Bin (B รอบ)
         ↓
    UCL = Quantile ของ resample
```

### 3.2 แนวทาง EVT/GPD (แทนที่ Bootstrap)

```
Exceedances ← ข้อมูลที่เกิน u = μ+3σ  (เหมือนกัน)
            ↓
    Fit GPD ด้วย MLE → ได้ ξ̂ และ σ̂
            ↓
    UCL = u + (σ̂/ξ̂)[(1-p)^(-ξ̂) - 1]   (closed-form)
```

**Core idea:** Tail bins ที่ RBULT ดึงออกมาอยู่แล้วนั้น **เทียบเท่ากับ exceedances ใน POT** — เพียงแต่แทนที่จะ bootstrap resample ก็ fit GPD กับข้อมูลพวกนั้นแทน

### 3.3 เปรียบเทียบข้อได้เปรียบ/ข้อเสีย

| ประเด็น | RBULT (Bootstrap) | EVT/GPD |
|---|---|---|
| **Theoretical guarantee** | Non-parametric, data-driven | Asymptotic guarantee (Fisher-Tippett) |
| **Extrapolation** | ❌ Bounded ใน observed data เท่านั้น | ✅ Extrapolate ออกนอกข้อมูลที่เคยเห็นได้ |
| **Small sample** | ✅ Robust เมื่อ exceedances น้อย | ❌ MLE ของ GPD unstable เมื่อ n < 30 |
| **Computation** | ช้ากว่า (resample B รอบ) | เร็วกว่า (closed-form formula) |
| **Threshold selection** | ตาม σ-rule อยู่แล้ว | ต้องเลือก $u$ อย่างระมัดระวัง |
| **Memory** | O(D) — discard chunk ได้ | O(D) — เก็บเฉพาะ exceedances |

---

## 4. แนวคิด Hybrid EVT-Bootstrap (สำหรับ Paper 2)

แทนที่จะเลือกอย่างใดอย่างหนึ่ง สามารถรวมทั้งสองแนวทางได้:

```
เมื่อ chunk ใหม่เข้ามา:
    ดึง exceedances ออกจาก tail
            ↓
    |exceedances| ≥ 30 ?
       ✅ ใช่ → Fit GPD → UCL จาก closed-form  (EVT)
       ❌ ไม่ใช่ → Bootstrap resample → UCL จาก quantile (RBULT)
```

**ข้อได้เปรียบของ Hybrid:**
- ได้ asymptotic guarantee จาก EVT ในสภาวะ data-rich
- ได้ robustness จาก bootstrap ในสภาวะ small-sample (ช่วงต้นของ stream)
- Memory ยังคง O(D) เหมือนเดิม
- Theoretical contribution ใหม่ที่ยังไม่มีใครทำสำหรับ streaming SPC

---

## 5. EVT กับ Concept Drift Detection

### 5.1 ปัญหาของ Drift Detector แบบดั้งเดิม

ADWIN, DDM, PHT ที่นิยมใช้กันนั้น monitor การเปลี่ยนแปลงของ **mean หรือ variance** ของ distribution ทั้งหมด ซึ่งหมายความว่าถ้า drift เกิดเฉพาะที่ **หางของการแจกแจง** โดยที่ค่าเฉลี่ยยังไม่เปลี่ยน detector พวกนี้จะตรวจไม่พบหรือตรวจพบช้ามาก

ในบริบท industrial IoT นี่สำคัญมาก เพราะ sensor ที่กำลังจะเสียมักแสดงอาการก่อนผ่านการเปลี่ยนแปลงของ extreme values ไม่ใช่ mean

### 5.2 แนวคิด: Monitor GPD Parameters ข้าม Chunk

เมื่อ fit GPD กับ tail data ใน chunk $m$ จะได้ parameter สองตัว คือ $\hat{\xi}_m$ (shape) และ $\hat{\sigma}_m$ (scale) ถ้า distribution เปลี่ยนไป parameter เหล่านี้จะเปลี่ยนตาม จึงสามารถ monitor การเปลี่ยนแปลงได้ดังนี้

$$\Delta\xi_m = \hat{\xi}_m - \hat{\xi}_{m-1}, \quad \Delta\sigma_m = \hat{\sigma}_m - \hat{\sigma}_{m-1}$$

ถ้า $|\Delta\xi_m|$ หรือ $|\Delta\sigma_m|$ เกิน threshold ที่กำหนด ก็แสดงว่า tail behavior เปลี่ยนแปลงไป ซึ่ง trigger ให้ refit model ใหม่

**ความหมายของแต่ละ parameter:**
- **$\xi$ เพิ่มขึ้น** → หางหนักขึ้น (heavy-tailed มากขึ้น) → sensor เริ่มสร้าง spike รุนแรงขึ้น
- **$\sigma$ เพิ่มขึ้น** → ความกว้างของหางเพิ่ม → extreme events มี magnitude ใหญ่ขึ้น

### 5.3 ประเภทของ Concept Drift ที่ EVT ตรวจได้

| ประเภท Drift | สัญญาณจาก EVT | ADWIN/DDM ตรวจได้มั้ย |
|---|---|:---:|
| **Sudden drift** | $\xi$ หรือ $\sigma$ เปลี่ยนแบบ abrupt ใน chunk เดียว | ✅ |
| **Gradual drift** | $\xi$ หรือ $\sigma$ ค่อยๆ trend ขึ้นหรือลงหลาย chunk | ✅ (ช้า) |
| **Recurring drift** | $\xi$ oscillate กลับมาค่าเดิมเป็นรอบๆ (seasonal) | ❌ |
| **Tail-only drift** | $\xi$ หรือ $\sigma$ เปลี่ยน แต่ mean ยังเท่าเดิม | ❌ |

**จุดเด่นที่สำคัญที่สุด:** Tail-only drift คือสิ่งที่ EVT ตรวจได้เพียงเจ้าเดียว เพราะ ADWIN/DDM monitor แค่ mean/variance ซึ่งยังไม่เปลี่ยนในกรณีนี้

### 5.4 Workflow การรวม EVT Drift Detector เข้ากับ RBULT

```
Chunk ใหม่ m เข้ามา
        ↓
  Fit GPD กับ tail → ได้ ξ̂_m, σ̂_m
        ↓
  คำนวณ Δξ_m และ Δσ_m เทียบกับ chunk ก่อนหน้า
        ↓
  |Δξ_m| > τ_ξ  หรือ  |Δσ_m| > τ_σ ?
    ✅ ใช่ → Trigger Concept Drift Alert
             → Reset tail buffer
             → Refit distribution ทั้งหมดด้วยข้อมูลใหม่
    ❌ ไม่ → Lazy boundary expansion ตามปกติ (RBULT)
        ↓
  อัปเดต ξ̂_(m-1) ← ξ̂_m,  σ̂_(m-1) ← σ̂_m
```

### 5.5 สัญญาณ Drift จาก RBULT ที่มีอยู่แล้ว

RBULT มีสัญญาณ drift โดยธรรมชาติอยู่แล้วคือเมื่อ **boundary expansion เกิดถี่ผิดปกติ** หมายความว่า incoming data เกินขอบเขตที่ประมาณไว้บ่อยขึ้น ซึ่งเป็น indirect signal ของ drift

การเพิ่ม EVT layer จะเปลี่ยน indirect signal นี้ให้เป็น **explicit, interpretable drift indicator** ผ่าน parameter $\hat{\xi}$ และ $\hat{\sigma}$ ซึ่งมี physical meaning ชัดเจน

---

## 6. วิธี Position งานเทียบกับ EVT ในบทความ

สามารถเขียนใน Related Work หรือ Discussion ได้ว่า:

> *"RBULT-SPC operates as a distribution-adaptive online tail estimator that employs non-parametric bootstrap resampling in place of GPD fitting, deliberately avoiding the threshold selection problem and parameter instability inherent to POT-based EVT approaches in small-sample streaming regimes. While EVT provides asymptotic guarantees for tail quantile estimation, its practical application requires sufficient exceedances (typically n ≥ 30–50) to achieve stable MLE convergence — a condition that cannot always be guaranteed in high-dimensional streaming environments where tail observations are inherently sparse."*

---

## 7. สรุป Roadmap

| ขั้นตอน | แนวทาง | เป้าหมาย |
|---|---|---|
| **Paper 1 (ปัจจุบัน)** | RBULT Bootstrap-based | Prove O(D) memory + streaming SPC capability |
| **Paper 2 (อนาคต)** | Hybrid EVT-Bootstrap + EVT Drift Detector | Add asymptotic EVT guarantee + tail-aware concept drift detection + adaptive refitting |

**Contribution หลักของ Paper 2 ที่น่าสนใจ:**
- EVT-based Concept Drift Detector ที่ monitor $\hat{\xi}$ และ $\hat{\sigma}$ แทน mean/variance
- ตรวจ tail-only drift ได้ ซึ่ง ADWIN/DDM ทำไม่ได้
- Hybrid fallback: GPD เมื่อ exceedances มาก, Bootstrap เมื่อน้อย
- Memory ยังคง O(D) ตลอด

---

*บันทึกโดย Claude (Cowork) จากการสนทนากับ Prem Junsawang — RBULT Project*
