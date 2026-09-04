# สรุปการทดลองทั้งหมดของ Paper 1: RBULT-SPC

> ปรับปรุง 4 กันยายน 2569 หลังการรันทดลองใหม่ทั้งหมด — ดู [Changelog](#changelog-สิ่งที่เปลี่ยนจากฉบับก่อน) ท้ายเอกสาร
> ตัวเลขทั้งหมดในเอกสารนี้มาจาก `results/tier2_final_all_methods.csv` และรายงานใน `results/`

---

## ภาพรวมโครงสร้างการทดลอง

| Tier | ชื่อ | วัตถุประสงค์ |
|---|---|---|
| **Tier 1** | 1D Synthetic Gold Standard Suite | ตรวจสอบ correctness ทางทฤษฎี ควบคุม ground-truth ได้ |
| **Tier 2** | Multivariate Industrial Benchmarks | พิสูจน์ว่าใช้งานได้จริงในข้อมูล real-world |

### กฎเกณฑ์การแจ้งเตือนระดับ chunk (เปลี่ยนใหม่)

เดิมใช้ค่าคงที่ `C_thresh = 3` ซึ่ง **ไม่ scale-free** — จำนวน violation ที่ chunk ปกติมีอยู่แล้วโตตามขนาด chunk $k$ ค่าคงที่จึงเข้มเกินไปสำหรับ chunk เล็ก และหลวมเกินไปสำหรับ chunk ใหญ่ (บน TEP ที่ $k=500, D=34$ chunk ปกติมี violation เฉลี่ย ~16 ครั้งต่อ feature ค่า $C=3$ จึงเตือนแทบทุก chunk)

ปัจจุบันใช้กฎอัตรา:

$$C = \lceil 0.05 \cdot k \rceil$$

| Dataset | $k$ | $C$ |
|---|---:|---:|
| AI4I 2020 | 100 | 5 |
| Industrial Pump | 200 | 10 |
| Water Pump / TEP | 500 | 25 |
| MetroPT-3 | 1000 | 50 |

**กฎนี้ใช้กับทั้ง 4 method เท่ากัน** เพื่อให้เปรียบเทียบได้อย่างยุติธรรม

---

## Tier 1: 1D Synthetic Benchmark — "7-Scenario Gold Standard Suite"

**5 การแจกแจง:** Gaussian, Laplace, Gamma, Beta, Mixture (โค้ดจริงใช้ F-Distribution, Uniform, Wald, Gamma, Normal)

**7 Scenarios:** A (Clean), B1–B3 (GAWN 0.1σ/0.2σ/0.3σ), C1–C3 (Impulse Spikes 1%/5%/10%)

**Grid:** chunk size 3 ระดับ × alpha 2 ระดับ × 3 methods × 2 protocols = 1,260 runs

### ผลลัพธ์หลัก Tier 1 (ไม่เปลี่ยนแปลง)

- RBULT รักษา Coverage ใกล้ target α ได้ในทุก distribution และทุก scenario
- ทำงานได้ดีทั้ง Clean (A) และ contaminated (B1–B3, C1–C3)
- ภายใต้ Pre-Sequential Predictive Protocol RBULT ยังใกล้เคียง Offline Bootstrap ทั้งที่ใช้ข้อมูลน้อยกว่ามาก
- C3 (Impulse 10%) ท้าทายที่สุด แต่ RBULT ยัง robust ผ่าน Z-score pre-filtering

> Tier 1 ไม่ได้ import `spc_rbult.py` การเปลี่ยนแปลงในรอบนี้จึงไม่กระทบ และการรันซ้ำได้รูปที่ byte-identical กับของเดิม ยืนยันว่า pipeline นี้ reproducible (มีการ seed RNG)

---

## Tier 2: Multivariate Industrial Benchmarks

### 2.1 รายละเอียดชุดข้อมูล

| Dataset | D | N | in-control chunks | ลักษณะข้อมูล |
|---|:---:|---:|---:|---|
| AI4I 2020 | 5 | 10,000 | **6 / 100** | Predictive maintenance |
| MetroPT-3 | 7 | 1,516,948 | 1,482 / 1,517 | Metro compressor |
| Industrial Pump | 5 | 20,000 | **0 / 100** | Pump sensor |
| Water Pump | 10 | 220,320 | 405 / 441 | SCADA water pump |
| TEP Mode 1 | 34 | 1,740,000 | **38 / 3,480** | Tennessee Eastman Process |
| TEP Mode 3 | 34 | 1,739,400 | **39 / 3,479** | TEP mode 3 |
| TEP Mode 4 | 34 | 1,719,000 | **41 / 3,438** | TEP mode 4 |
| TEP Mode 5 | 34 | 1,729,800 | **44 / 3,460** | TEP mode 5 |

> ⚠️ **ข้อจำกัดสำคัญที่ต้องระบุใน paper:** Industrial Pump **ไม่มี in-control chunk เลย** ค่า Chunk FAR และ ARL0 ของชุดนี้จึงไม่นิยาม (เลข 0.00 ที่เห็นมาจาก `max(1, 0)` ในโค้ด ไม่ใช่การวัด) ส่วน AI4I มีเพียง 6 chunk และ TEP มี 38–44 chunk จาก ~3,480 — ตัวเลข Chunk FAR ของชุดเหล่านี้เปราะมาก
>
> ทั้ง TEP (98.9% เป็น OOC chunk) และ AI4I (94%) มีสัดส่วนกลับด้านกับสถานการณ์ SPC ปกติที่ข้อมูล in-control ควรเป็นส่วนใหญ่

### 2.2 Coverage Rate

| Dataset | Shewhart | EWMA | Bootstrap | **RBULT** |
|---|---:|---:|---:|---:|
| AI4I 2020 | 69.69% | 58.37% | **98.82%** | 97.79% |
| MetroPT-3 | 77.68% | 51.01% | 98.76% | **98.90%** |
| Industrial Pump | 100.00% | 99.91% | 98.96% | **99.40%** |
| Water Pump | 51.06% | 25.65% | 98.63% | **99.95%** |
| TEP Mode 1 | 91.19% | 81.79% | **99.02%** | 96.74% |
| TEP Mode 3 | 80.80% | 60.99% | **99.01%** | 93.73% |
| TEP Mode 4 | 84.72% | 72.12% | **99.01%** | 96.67% |
| TEP Mode 5 | 85.15% | 71.38% | **99.05%** | 97.79% |

RBULT ชนะ Bootstrap บนข้อมูลมิติต่ำ–กลาง (MetroPT-3, Pump, Water Pump) แต่ **แพ้บน TEP ทุก mode** ซึ่ง Sliding-Window Bootstrap ได้ ~99.0% สม่ำเสมอ

### 2.3 Joint Coverage — เมตริกใหม่

Coverage ในตาราง 2.2 คือ **marginal coverage เฉลี่ยข้ามมิติ** ซึ่งเป็นข้อความที่อ่อนกว่าที่ดูเหมือน เมตริกใหม่ `joint_coverage_pct` นับเฉพาะกรณีที่ **ทุกมิติอยู่ในขอบเขตพร้อมกัน**

| Dataset | D | Marginal | **Joint** | ≥ 95%? |
|---|:---:|---:|---:|:---:|
| Water Pump | 10 | 99.95% | 99.55% | ✓ |
| Industrial Pump | 5 | 99.40% | 97.03% | ✓ |
| MetroPT-3 | 7 | 98.90% | **94.89%** | ✗ |
| AI4I 2020 | 5 | 97.79% | **91.34%** | ✗ |
| TEP Mode 5 | 34 | 97.79% | **70.50%** | ✗ |
| TEP Mode 4 | 34 | 96.67% | **65.26%** | ✗ |
| TEP Mode 1 | 34 | 96.74% | **61.39%** | ✗ |
| TEP Mode 3 | 34 | 93.73% | **25.36%** | ✗ |

**6 จาก 8 ชุดต่ำกว่าเป้า Bonferroni 95%** และบน TEP ตกลงไปถึง 25–70% — แปลว่าถ้าดูแบบ joint RBULT ตีธงว่าผิดปกติถึง 30–75% ของ observation

สาเหตุ: Bonferroni รับประกัน joint ≥ 1−α_sys **ก็ต่อเมื่อ** ทุกมิติทำ marginal ได้ถึง 1−α_dim แต่บน TEP per-dim FAR ที่วัดได้สูงกว่าเป้าถึง **15–43 เท่า** (Mode 3 สูงสุดที่ 43×) นี่คือรากของทั้งปัญหา joint coverage และพฤติกรรมระดับ chunk

### 2.4 Chunk-level False Alarm Rate

| Dataset | Shewhart | EWMA | Bootstrap | **RBULT** |
|---|---:|---:|---:|---:|
| AI4I 2020 | 100.00% | 100.00% | **0.00%** | 66.67% |
| MetroPT-3 | 99.46% | 100.00% | **8.23%** | 25.24% |
| Industrial Pump * | – | – | – | – |
| Water Pump | 99.51% | 100.00% | 32.59% | **29.38%** |
| TEP Mode 1 | 0.00% | 100.00% | 0.00% | **0.00%** |
| TEP Mode 3 | 100.00% | 100.00% | 0.00% † | 100.00% |
| TEP Mode 4 | 4.88% | 100.00% | 0.00% † | **0.00%** |
| TEP Mode 5 | 90.91% | 100.00% | 0.00% † | 11.36% |

\* ไม่นิยาม (ไม่มี in-control chunk) &nbsp;&nbsp; † **ไม่เคยส่งสัญญาณเลย — detection = 0.00%** ดูหมายเหตุด้านล่าง

> ⚠️ **ต้องอ่าน Chunk FAR คู่กับ detection เสมอ** ตัวตรวจจับที่ไม่เคยเตือนจะได้ Chunk FAR = 0.00% ที่ดูสมบูรณ์แบบ Sliding-Window Bootstrap บน TEP เป็นกรณีนี้พอดี — ได้ 0.00% ทุก mode เพราะไม่เคยเตือนเลยแม้แต่ครั้งเดียว ค่า ARL0 38/39/41/44 และ ARL1 = 1.00 ของมันล้วนเป็นค่า fallback

**เทียบที่จุดทำงานจริงบน TEP Mode 1** (Chunk FAR 0.00% เท่ากันทั้งสาม):

| Method | Chunk FAR | **Detection** |
|---|---:|---:|
| Shewhart | 0.00% | **79.92%** |
| **RBULT-SPC** | 0.00% | **68.65%** |
| Sliding-Window Bootstrap | 0.00% | 0.00% |

**สรุป: RBULT ไม่ได้เหนือกว่า baselines ในการคุม false alarm** บน TEP Mode 1/4/5 RBULT สูสีกับ Shewhart และแพ้บน Mode 3 ส่วนบน AI4I และ MetroPT-3 Bootstrap ชนะชัดเจน

### 2.5 เมตริกที่ไม่นิยาม รายงานเป็น NaN แทนเลขสวย

โค้ดเดิมมี fallback 3 จุดที่คืน**ตัวเลขที่ดูดี**ในกรณีที่ปริมาณนั้นไม่นิยาม แก้แล้วทั้งใน library และสคริปต์ทั้ง 6 ตัว

| สถานการณ์ | ค่าเดิม | ค่าใหม่ |
|---|---|---|
| Chunk FAR เมื่อไม่มี in-control chunk (0/0) | `0.00` จาก `max(1, in_control_chunks)` | **NaN** |
| ARL0 เมื่อไม่มี in-control chunk | `0.00` | **NaN** |
| ARL1 เมื่อไม่เคยตรวจเจออะไรเลย | `1.00` — เท่ากับ "ตรวจเจอทันที" ซึ่งเป็นคะแนนดีที่สุด | **NaN** + เพิ่มฟิลด์ `n_detected_episodes` |

ARL0 ยังคงคืนจำนวน in-control chunk เมื่อไม่มี false alarm เพราะเป็น lower bound แบบ censored ที่ถูกต้อง และเพิ่ม flag `arl_0_censored` กำกับ

**สิ่งที่การแก้นี้เปิดโปง — ตอนนี้ปรากฏในไฟล์ผลจริง ไม่ใช่แค่การวิเคราะห์แยก**

1. **Industrial Pump** — ทั้ง 4 method คืน NaN สำหรับ Chunk FAR, ARL0, ARL1 เพราะไม่มี in-control chunk เลย ค่า `0.00%` ทุกช่องที่เคยรายงานจึงเป็นผลของเลขคณิต ไม่ใช่การวัด และอ่านแล้วเหมือน RBULT คุม false alarm ได้สมบูรณ์แบบ

2. **Sliding-Window Bootstrap ไม่เคยตรวจเจออะไรเลยบน TEP** — `arl_1` เป็น NaN ทั้ง 8 การรัน (4 mode × raw/diff) ค่า `ARL1 = 1.00` ที่เคยตีพิมพ์เป็น fallback มาตลอด เมื่อรวมกับ Chunk FAR 0.00% และ ARL0 = 100 การอ่านที่ซื่อสัตย์คือ **baseline ตัวนี้ไม่เคยส่งสัญญาณเตือนเลยบน TEP** จึงไม่ใช่คู่แข่งที่แข็งแรง และการเทียบ chunk-level metric กับมันไม่มีความหมาย

#### Industrial Pump: label เป็นการโยนเหรียญ

`Maintenance_Flag` ไม่มีข้อมูลเลย ยืนยันด้วย 3 ทางที่สอดคล้องกัน:

| การตรวจ | ผล |
|---|---|
| AUC ของ 6 ตัวแปรในการทำนาย flag | 0.4925 – 0.5050, **p > 0.05 ทุกตัว** |
| flag rate ต่อปั๊ม (5 ปั๊ม) | 48.6% – 50.6% เท่ากันหมด |
| ความยาวช่วงต่อเนื่องเฉลี่ย | **1.99 แถว** ตรงกับที่ Bernoulli(0.4984) อิสระทำนายไว้ (1.99) พอดี |

**ไม่มีวิธีใดกู้ chunk-level metric จาก label ที่สุ่มได้** — ไม่ว่าจะนิยาม chunk label ใหม่ แยกตาม `Pump_ID` หรือปรับ threshold

`exp_pump_benchmark.py` จึงเพิ่ม label quality gate ที่ตรวจและเตือนตอนรัน พร้อมแสดง **—** และข้อความ "UNDEFINED — no in-control chunks" ในตาราง แทนที่จะปล่อย `nan%` วางคู่กับคำบรรยาย "Low Batch False Alarm Rate"

> เมตริกระดับ sample (coverage, sample FAR, interval width, RAM, latency) ไม่ขึ้นกับ label จึงยังใช้ได้ และชุดนี้ยังมีประโยชน์ **เพราะ**มันเป็น i.i.d. — เป็นจุดอ้างอิงที่ `width_ratio_local = 1.00` ซึ่งใช้วัดว่าช่วงบนข้อมูลที่มี autocorrelation พองไปเท่าไร

---

### 2.6 TEP: ปรับ chunk size ให้ตรง run และทำ within-run differencing

**chunk size 500 → 600** — TEP มีรูปทรง (runs × 600 steps × 34 vars) ต่อกันเป็นสตรีมเดียว ค่า k ที่ไม่หาร 600 ลงตัวทำให้ทุก chunk คร่อมรอยต่อ run ซึ่งเป็นความไม่ต่อเนื่องจากการจัดข้อมูล ไม่ใช่จากกระบวนการ

ผลข้างเคียงที่สำคัญกว่าคือ label: มีเพียง 100 จาก ~2,900 run ที่เป็น normal และแต่ละ run ยาว 1.2 chunk ของขนาด 500 **run ปกติจึงมักถูก label เป็น faulty เพราะแชร์ chunk กับ run ที่มี fault** พอ k=600 → **in-control chunks เพิ่มจาก 38–44 เป็น 100 ทุก mode** และ CI ของ Chunk FAR 0% แคบลงจาก < 9.25% เหลือ < 3.0% (C = ⌈0.05×600⌉ = 30)

> **ข้อจำกัดที่พบระหว่างทาง:** coverage *ลดลง* หลังแก้ chunking (Mode 1: 96.74 → 93.71) สาเหตุไม่ใช่การ align run เอง — ที่ N คงที่และข้อมูลเดียวกัน **ความกว้างของช่วงแปรตาม k แบบไม่เป็นเชิงเดียว** ทดสอบบน Mode 1 (300 runs): k = 100/150/300/600/1200 ให้ความกว้าง 62.98/31.22/34.32/36.10/45.37 และ coverage 98.17/96.64/97.53/93.59/95.14 ทำซ้ำ 4 ครั้งต่อค่าได้ spread 0.0 (width) และ 0.01 จุด (coverage) จึงเป็นสัญญาณจริงไม่ใช่ noise — **ตัวประมาณช่วงขึ้นกับขนาด chunk** ซึ่งเป็นพารามิเตอร์อิสระของการทดลอง ต้องระบุเป็นข้อจำกัด

**Within-run differencing** — diff ภายใน run เท่านั้น ไม่ข้ามรอยต่อ (ความผิดพลาดแบบเดียวกับ `Tool wear Rate` ของ AI4I) แต่ละ run เสีย sample แรก (600 → 599) ทำที่ระดับ dataframe เพื่อให้ทั้ง 4 method ได้ข้อมูลชุดเดียวกัน ส่วน `RBULTControlChart(difference=True)` ทำ transform เดียวกันแบบ streaming $O(D)$ (state 1 scalar ต่อ feature) ตรวจแล้วว่าให้ค่าตรงกับ `np.diff` ทุกประการ

| Mode | Coverage raw → diff | Joint raw → diff | Chunk FAR raw → diff | ARL0 raw → diff | width_ratio_global |
|---|---|---|---|---|---|
| 1 | 93.71 → **98.13** | 48.83 → **66.17** | 0.00 → 0.00 | 100.0 → 100.0 | 0.51 → **0.85** |
| 3 | 91.76 → **97.20** | 17.79 → **59.10** | **100.00 → 0.00** | **0.00 → 100.0** | 0.52 → **0.78** |
| 4 | 90.64 → **96.09** | 50.61 → **55.23** | 1.00 → 0.00 | 49.5 → **100.0** | 0.40 → **0.65** |
| 5 | 92.39 → **97.37** | 45.15 → **61.63** | **37.00 → 0.00** | 1.7 → **100.0** | 0.46 → **0.73** |

**ดีขึ้นทุก mode ทุกเมตริก** โดยเฉพาะ **Mode 3 ที่พลิกจากพังที่สุดเป็นใช้ได้** — เดิม Chunk FAR ติด 100% ทุก threshold, joint 17.8%, ARL0 = 0 ตอนนี้ Chunk FAR 0.00% และ joint 59.1%

ข้อสรุปเดิมที่ว่า Mode 3 "ไม่มีสัญญาณ" **ผิดสองชั้น**: สัญญาณมีอยู่จริง แต่ (ก) กระจายข้ามมิติ จึงต้องดูแบบ TOTAL และ (ข) ถูกกลบด้วยระดับที่แต่ละ run ตั้งอยู่ จึงต้อง diff

**ผลพลอยได้ 2 ข้อ**

1. **ช่องว่าง in-sample vs prequential หายเป็นศูนย์ทุก mode** (จาก 0.25–0.69 จุด → 0.00) — ยืนยันกลไกที่วัดไว้ก่อนหน้าอย่างอิสระ ว่าช่องว่างเกิดจากขอบเขตที่พองเพื่อดูดซับ non-stationarity พอตัด non-stationarity ทิ้ง ช่องว่างก็หายไป
2. **`width_ratio_global` ขึ้นเป็น 0.65–0.85** จาก 0.40–0.52 คือช่วงเข้าใกล้ support จริงแทนที่จะครอบได้แค่ครึ่งเดียว ซึ่งเป็นสาเหตุที่ per-dim FAR เกินเป้า Bonferroni บน raw TEP

**ข้อแลกเปลี่ยน:** AUC ของสถิติ violation รายมิติ **ลดลงบน Mode 1 และ 4** (0.849 → 0.791, 0.863 → 0.780) เพราะ differencing ทิ้งข้อมูลระดับ fault ที่ค้างระดับไว้จึงเหลือเป็น spike ครั้งเดียวตอนเปลี่ยน ส่วนสถิติแบบรวมข้ามมิติกระทบน้อยกว่ามาก (0.863 → 0.844, 0.870 → 0.853) — **รายงานทั้งสองแบบ ไม่ได้แทนที่กัน**

---

### 2.7 One-Step-Ahead Coverage — protocol ที่สะท้อนการใช้งานจริง

Coverage ในตาราง 2.2 เป็นแบบ **in-sample**: `compute_spc_metrics()` เอาขอบเขต **สุดท้าย** ไปวัดย้อนหลังทั้งสตรีม และเพราะขอบเขตขยายอย่างเดียวไม่เคยหด ข้อมูลช่วงต้นจึงถูกวัดด้วยขอบเขตที่เรียนจากข้อมูลที่ยังมาไม่ถึง ส่วนการนับ violation รายchunk ก็เอนเอียงเช่นกัน — chunk ขยายขอบเขตก่อน แล้วค่อยถูกวัดด้วยขอบเขตที่เพิ่งขยาย

ในการใช้งานจริงทำแบบนั้นไม่ได้ ขอบเขตต้องมีอยู่ก่อนข้อมูลที่จะถูกตัดสิน `compute_prequential_metrics()` จึงวัดแต่ละ chunk ด้วยขอบเขตจาก chunk $1 \dots m-1$ ก่อนที่ chunk นั้นจะขยายมัน (ตัด chunk แรกของแต่ละมิติออกจากตัวหาร) Tier 1 รายงานสอง protocol นี้มาตลอด ตอนนี้ Tier 2 มีครบเหมือนกัน

| Dataset | `width_ratio_local` | Coverage in-sample | **one-step-ahead** | ต่าง | Joint in-sample | **joint one-step-ahead** | ต่าง |
|---|---:|---:|---:|---:|---:|---:|---:|
| Industrial Pump | **1.00** | 99.417% | 99.411% | **0.006** | 97.10% | 97.07% | **0.03** |
| TEP Mode 3 | 1.77 | 93.705% | 93.525% | 0.180 | 25.17% | 24.42% | 0.76 |
| TEP Mode 1 | 2.07 | 96.736% | 96.540% | 0.196 | 61.40% | 60.71% | 0.69 |
| TEP Mode 4 | 2.67 | 96.670% | 95.918% | 0.753 | 65.27% | 63.36% | 1.91 |
| MetroPT-3 | 1.55 | 98.895% | 98.109% | 0.787 | 94.89% | 91.30% | 3.58 |
| TEP Mode 5 | 8.19 | 97.789% | 96.865% | 0.924 | 70.45% | 65.80% | 4.66 |
| Water Pump | **8.51** | **99.955%** | 97.808% | 2.146 | **99.55%** | **87.68%** | **11.87** |
| AI4I 2020 | 4.55 | 97.792% | 95.495% | 2.297 | 91.34% | 82.47% | 8.87 |

**ช่องว่างเป็นฟังก์ชันโดยตรงของการที่ช่วงต้องพองแค่ไหน** — สหสัมพันธ์กับ `width_ratio_local` เท่ากับ **0.686** (marginal) และ **0.782** (joint)

- **Industrial Pump** ข้อมูลเป็น i.i.d. ช่วงพอดีกับการแกว่งจริงอยู่แล้ว (ratio 1.00) เสียไปแค่ **0.006 จุด** — สอง protocol แทบแยกไม่ออก
- **Water Pump** ช่วงกว้างกว่าการแกว่งจริง 8.5 เท่า เสีย **joint coverage ไป 11.87 จุด** (99.55% → 87.68%)

กลไกเดียวกันอธิบายทั้งสองเรื่อง: ช่วงที่พองขึ้นเพื่อดูดซับ non-stationarity คือช่วงที่ถูก fit ด้วยข้อมูลที่ผู้พยากรณ์ยังไม่เห็น

ตรวจสอบกับข้อมูลสังเคราะห์ก่อนใช้จริงแล้ว — stationary ได้ 99.22% (in-sample) เทียบ 99.21% (prequential) ส่วน drift ได้ 99.93% เทียบ **71.23%** ช่องว่างเกิดเฉพาะเมื่อมี drift ให้เอาเปรียบ ตรงตามทฤษฎี

> **รายงานทั้งสองค่า** ค่า in-sample ยังใช้ได้กับสิ่งที่มันวัด (ขอบเขตสุดท้ายครอบข้อมูลในอดีตได้ดีแค่ไหน) ส่วนค่า prequential คือค่าที่อธิบายพฤติกรรมตอน deploy

---

### 2.8 Interval Width — ตัวคู่ที่ต้องอ่านพร้อม Coverage

Coverage อ่านลำพังไม่ได้ — ช่วงที่กว้างพอจะได้ coverage 100% เสมอโดยไม่มีข้อมูลอะไรเลย และเพราะขอบเขตของ RBULT ขยายอย่างเดียวไม่เคยหด มันจึงกว้างขึ้นเพื่อดูดซับความไม่นิ่งของข้อมูล Tier 1 มีเมตริกนี้อยู่แล้ว (`Mean_Interval_Width`, `Sigma_L/R`) ตอนนี้ Tier 2 มีครบเหมือนกัน (เก็บด้วย Welford → 9 scalar ต่อ feature ยังเป็น $O(D)$)

| Dataset | lag-1 AC | Coverage | Joint | **width_ratio_local** | width_ratio_global |
|---|---:|---:|---:|---:|---:|
| Water Pump | 0.998 | **99.95%** | 99.55% | **8.51** (สูงสุด 19.4) | 1.11 |
| TEP Mode 5 | 0.948 | 97.79% | 70.49% | **8.19** (สูงสุด **157.6**) | 0.79 |
| AI4I 2020 | — | 97.79% | 91.34% | 4.55 | 0.94 |
| TEP Mode 4 | 0.948 | 96.66% | 65.26% | 2.67 | 0.74 |
| TEP Mode 1 | 0.948 | 96.74% | 61.39% | 2.08 | 0.69 |
| TEP Mode 3 | 0.948 | 93.70% | 25.02% | 1.77 | 0.65 |
| MetroPT-3 | 0.970 | 98.90% | 94.89% | 1.55 | 1.40 |
| Industrial Pump | 0.001 | 99.40% | 97.04% | **1.00** | 1.01 |

- **`width_ratio_local`** = ความกว้างสุดท้าย ÷ การแกว่งเฉลี่ยภายใน chunk → ใกล้ 1 คือช่วงพอดีกับข้อมูล, มากกว่า 1 มาก คือช่วงพองเกิน coverage จึงได้มาง่าย
- **`width_ratio_global`** = ความกว้างสุดท้าย ÷ ช่วง percentile 0.5–99.5 ทั้งสตรีม → ใกล้ 1 คือลู่เข้าหา support จริง (เท่ากับที่ full-history bootstrap คำนวณ)

**ข้อสังเกต 2 ข้อ**

1. **Coverage สูงสุดมาพร้อมช่วงกว้างสุด** — Water Pump ได้ coverage ดีที่สุดในชุดทั้งหมด (99.955%) ด้วยช่วงที่กว้างกว่าการแกว่งจริง **8.5 เท่า** (มิติหนึ่งถึง 19 เท่า) ส่วน Industrial Pump ที่ข้อมูลเป็น i.i.d. ได้พอดี **1.00** ลำดับของ `width_ratio_local` เรียงตาม autocorrelation ไม่ได้เรียงตามคุณภาพของตัวประมาณ — และเป็นเหตุผลเดียวกันที่ detection พังบนชุดเหล่านั้นพอดี (Water Pump AUC 0.402, median violation = 0 ทั้งสองกลุ่ม)

2. **บน TEP ช่วงกลับ *แคบกว่า* support จริง** (`width_ratio_global` 0.65–0.79) ซึ่งเป็นสาเหตุโดยตรงที่ per-dim FAR สูงกว่าเป้า Bonferroni 15–43 เท่า และทำให้ joint coverage ตกไป 25–70%

> **ข้อสรุปที่ควรเขียนใน paper:** RBULT ไม่ต้องการ stationary preprocessing ต่างจาก Shewhart/EWMA ที่โมเดลค่ากลางพังทันที (coverage เหลือ 25–77%) — **นี่คือข้อได้เปรียบจริงที่วัดได้** แต่ coverage ที่สูงตามมานั้นได้จากการที่ช่วงกว้างขึ้นด้วย ไม่ใช่จากการประมาณที่แม่นขึ้นล้วน ๆ จึง **ต้องรายงาน coverage คู่กับ `width_ratio_local` เสมอ** และเพราะ `width_ratio_global` ≈ 1 บนชุดที่ไม่ใช่ TEP ข้ออ้างที่ซื่อสัตย์คือ RBULT ให้ช่วง **เทียบเท่า** full-history bootstrap โดยใช้หน่วยความจำ $O(D)$ แทน $O(N \cdot D)$ ไม่ใช่ให้ช่วงที่ดีกว่า

---

### 2.9 Memory Usage — จุดแข็งที่แท้จริง

| Dataset | D | **RBULT** | Baseline Bootstrap | ประหยัด |
|---|:---:|---:|---:|---:|
| AI4I 2020 | 5 | **0.52 KB** | 413.78 KB (full-history) | ~800× |
| Industrial Pump | 5 | **0.52 KB** | 826.91 KB (full-history) | ~1,600× |
| MetroPT-3 | 7 | **0.70 KB** | 90,932.70 KB (full-history) | **~130,000×** |
| Water Pump | 10 | **0.98 KB** | 17,667.15 KB (full-history) | ~18,000× |
| TEP Mode 1–5 | 34 | **3.23 KB** | 582.87 KB (sliding W=2000) | ~180× |

**นี่คือข้อสรุปเดียวที่ไม่เปลี่ยนเลยตลอดการตรวจสอบทั้งหมด** — วัดซ้ำกี่ครั้งก็ได้ค่าเดิมทุกหลัก และเป็น $O(D)$ จริงโดยไม่ขึ้นกับ $N$ แม้ที่ 1.7 ล้านสังเกต

> **แก้จากฉบับก่อน:** ตัวเลข "TEP ประหยัด ~156,000× เทียบ baseline 504,425.95 KB" **ใช้ไม่ได้แล้ว** เพราะ `exp_tep_benchmark.py` ปัจจุบันใช้ Sliding-Window (W=2000) ไม่ใช่ Full-History baseline ตัวที่ให้เลข 504,425.95 KB ไม่มีอยู่ในโค้ดอีกต่อไป ตัวเลขที่ reproduce ได้คือ **~180×**

### 2.10 Latency

| Dataset | D | RBULT |
|---|:---:|---:|
| MetroPT-3 | 7 | 5.24 ms |
| TEP Mode 1 | 34 | 9.16 ms |
| TEP Mode 3 | 34 | 9.41 ms |
| TEP Mode 5 | 34 | 10.15 ms |
| Industrial Pump | 5 | 10.99 ms |
| TEP Mode 4 | 34 | 12.30 ms |
| Water Pump | 10 | 25.59 ms |
| AI4I 2020 | 5 | 37.29 ms |

ทุกชุดต่ำกว่า 65 ms ต่อ chunk รองรับ real-time streaming ได้ (ค่าเหล่านี้แปรตามภาระเครื่องขณะรัน)

### 2.11 Baselines ที่เปรียบเทียบ

| Baseline | คำอธิบาย |
|---|---|
| Shewhart (3-sigma) | Classic control chart, assumes Gaussian |
| EWMA | λ = 0.2, L = 3 |
| Full-History Bootstrap | Percentile บนข้อมูลสะสมทั้งหมด, $O(N)$ memory — ใช้กับ AI4I, Pump, Water Pump, MetroPT-3 |
| Sliding-Window Bootstrap | Percentile บน window W = 2,000 ล่าสุด — ใช้กับ TEP |

---

## Tier 2 (Extra): TEP Threshold Sensitivity Study

### ผลลัพธ์ (แก้บั๊กแล้ว)

TEP Mode 1, กติกาแบบรวมทุกมิติ:

| C_thresh | Shewhart | EWMA | Sliding-Window | **RBULT** |
|:---:|---:|---:|---:|---:|
| 5 | 100.00% | 100.00% | 100.00% | 100.00% |
| 10 | 100.00% | 100.00% | 100.00% | 100.00% |
| 15 | 92.11% | 100.00% | 94.74% | 94.74% |
| 25 | 55.26% | 100.00% | 76.32% | **34.21%** |
| 50 | 10.53% | 100.00% | 73.68% | **0.00%** |

**RBULT ไวต่อการเลือก C_thresh อย่างมาก** — ต่างจากข้อสรุปเดิมที่ว่า "ไม่ sensitive" โดยสิ้นเชิง

> ⚠️ **ข้อสรุปเดิมเกิดจากบั๊ก** `exp_tep_sensitivity.py:56` อ่าน `summary['sample_ooc_count']` ซึ่งเป็น key ที่ `update_chunk` ไม่เคยคืนค่ามา `.get()` จึงตกไปที่ `0` ทุกครั้ง ทำให้ RBULT ถูกนับว่าไม่มี violation เลยในทุก chunk — นี่คือที่มาทั้งหมดของ "Chunk FAR = 0.00% ทุก threshold, ARL0 = 38.00" (และเลข 38.00 คือจำนวน in-control chunk ของ TEP Mode 1 พอดี ซึ่งเป็นค่าที่ `_compute_arl0` คืนเมื่อไม่มี alarm เกิดขึ้นเลย)

---

## PER-FEATURE vs TOTAL: การนับ violation ข้ามมิติ

มีสองวิธีรวม violation รายมิติเป็นสัญญาณระดับ chunk:

- **PER-FEATURE** (ที่ `RBULTControlChart` ใช้): เตือนเมื่อ**มิติใดมิติหนึ่ง**มี $V_d \geq C$
- **TOTAL**: เตือนเมื่อ $\sum_d V_d \geq C$

เทียบที่ FAR เท่ากัน (RBULT):

| Dataset | AUC (max) | AUC (sum) | detection @ FAR ≤ 5% |
|---|---:|---:|---|
| TEP Mode 4 | **0.888** | 0.869 | 74.7% → 74.4% |
| TEP Mode 1 | 0.884 | **0.888** | 74.5% → 76.1% |
| TEP Mode 5 | 0.829 | **0.838** | 51.0% → 52.2% |
| **TEP Mode 3** | 0.448 | **0.859** | **18.3% → 65.8%** |
| AI4I 2020 | 0.435 | 0.512 | 10.6% → 12.8% |
| Water Pump | 0.402 | 0.404 | 5.6% → 2.8% |
| MetroPT-3 | 0.154 | 0.153 | 8.6% = 8.6% |

**6 จาก 7 ชุดแทบไม่ต่าง** แต่ **TEP Mode 3 ต่างมหาศาล** — AUC กระโดดจาก 0.448 (สุ่ม) เป็น 0.859

Mode 3 มีสัญญาณ fault จริง แต่สัญญาณอยู่ในการที่ **หลายมิติเบี่ยงเบนเล็กน้อยพร้อมกัน** ไม่ใช่มิติเดียวเบี่ยงเบนหนัก การดูทีละมิติจึงมองไม่เห็น สมเหตุสมผลเชิงกระบวนการสำหรับ Mode 3 (50/50 mass ratio) ที่ความผิดปกติแพร่ทั่วระบบ

**ข้อเสนอ:** ใช้ทั้งสองสถิติควบคู่กัน — เตือนเมื่อ (มิติใดหลุดหนัก) **หรือ** (ผลรวมเกินเกณฑ์) โดยตั้งแต่ละตัวที่ $\alpha_{\text{chunk}}/2$ เพราะจับความผิดปกติคนละรูปแบบ: per-feature จับ fault ที่กระจุกตัว, total จับ fault ที่กระจายตัว

---

## สรุปภาพรวม: อะไรยืนได้ อะไรไม่ยืน

### ✅ ยืนได้

**1. Memory $O(D)$ — จุดแข็งที่แท้จริงและแข็งแรงที่สุด**
ลด memory 180× ถึง 130,000× คงที่ตาม $O(D)$ ไม่โตตาม $N$ แม้ที่ 1.7 ล้านสังเกต วัดซ้ำได้ค่าเดิมทุกครั้ง

**2. Coverage เทียบเท่า Full-History Bootstrap บนข้อมูลมิติต่ำ–กลาง**
ชนะบน MetroPT-3, Industrial Pump, Water Pump

**3. Latency ต่ำ** — ทุกชุด < 65 ms ต่อ chunk

**4. Robust ต่อ contamination (Tier 1)** — ทำงานได้ทั้ง clean และ noisy ทุก scenario

### ❌ ไม่ยืน

**1. "คุม false alarm ดีกว่า baselines"**
เมื่อเทียบด้วยกติกาเดียวกัน RBULT ไม่ได้เหนือกว่า — Bootstrap ชนะบน AI4I และ MetroPT-3, Shewhart สูสีหรือดีกว่าบน TEP Mode 1/4 ข้อความ *"Baselines มี Chunk FAR 92–100% ในขณะที่ RBULT = 0.00%"* **ต้องถอนออก**

**2. "Robust ต่อการเลือก C_thresh"**
RBULT ไวต่อ C_thresh มาก (100% → 0.00% เมื่อ C เปลี่ยนจาก 5 เป็น 50) ข้อสรุปเดิมเกิดจากบั๊ก

**3. Coverage บน TEP**
Sliding-Window Bootstrap ได้ ~99.0% ทุก mode ส่วน RBULT ได้ 93.7–97.8%

**4. ประหยัด memory 156,000× บน TEP**
โค้ดปัจจุบันให้ ~180× ตัวเลขเดิมมาจาก baseline ที่ไม่มีในโค้ดแล้ว

### 📐 กรอบการนำเสนอที่แนะนำ

> **RBULT-SPC ให้ coverage เทียบเท่า full-history bootstrap โดยใช้หน่วยความจำ $O(D)$ คงที่ที่ไม่ขึ้นกับ $N$**

ไม่ควรอ้างความเหนือกว่าด้าน false alarm

---

## ลักษณะข้อมูล Tier 2 — มีเพียง 2 จาก 5 ชุดที่เป็น time series แท้

วัดด้วย lag-1 autocorrelation (ถ้าเป็น time series ค่า ณ เวลา $t$ ต้องสัมพันธ์กับ $t-1$)

| ชุดข้อมูล       |   D | lag-1 autocorr | เป็น time series?             | in-control chunks |
| --------------- | --: | -------------: | ----------------------------- | ----------------: |
| Water Pump      |  10 |      **0.998** | ✅ ใช่                         |         405 / 441 |
| MetroPT-3       |   7 |      **0.970** | ✅ ใช่                         |     1,482 / 1,517 |
| TEP 1/3/4/5     |  34 |          0.948 | ⚠️ ต่อจาก 2,900 runs          |    38–44 / ~3,480 |
| AI4I 2020       |   5 |        0.931 * | ❌ ไม่ใช่ (แต่ละแถวคือชิ้นงาน) |           6 / 100 |
| Industrial Pump |   5 |      **0.001** | ❌ ไม่ใช่ (5 เครื่องต่อกัน)    |       **0 / 100** |

- **AI4I** \* — ค่า 0.931 ไม่ได้แปลว่าเป็น time series แต่เป็น artifact จากวิธีสร้างคอลัมน์: อุณหภูมิสองตัวถูกสร้างด้วย random walk และ `Tool wear [min]` เป็นตัวนับสะสม ส่วนสองมิติที่อธิบายกระบวนการจริงคือ `Rotational speed` (**0.008**) และ `Torque` (**0.005**) ซึ่งเป็น i.i.d. หลักฐานเชิงโครงสร้างชี้ชัดกว่า: `UDI` เป็น index ของผลิตภัณฑ์ที่เพิ่มทีละ 1 และ `Tool wear` รีเซ็ต 119 ครั้ง ความเสียหาย 339 sample กระจายเป็น **310 episode แยกกัน** จึงเหลือ in-control แค่ 6 chunk
- **Industrial Pump** — `Maintenance_Flag` เปลี่ยนค่า 9,979 ครั้งใน 20,000 แถว (ช่วง fault ยาวเฉลี่ย 2.0 แถว) ทำให้ทุก chunk มี flag → **ไม่มี in-control chunk เลย** ค่า Chunk FAR/ARL0 ของชุดนี้ไม่นิยาม
- **TEP** — รูปทรงจริงคือ (2900 runs × 600 steps × 34 vars) การ flatten สร้าง **รอยต่อเทียม 2,899 จุด** และที่ $k=500$ ทุก chunk คร่อมรอยต่อ อีกทั้ง **3 จาก 34 channels เป็นค่าคงที่**
- **Non-Gaussian ยืนยันได้** — ผ่าน Shapiro-Wilk เพียง **3 จาก 61 channels** และ excess kurtosis สูงถึง 117.91 บน TEP → ข้ออ้าง non-parametric มีน้ำหนักมาก

> รายละเอียดเต็มอยู่ใน `section_experimental_results.md` หัวข้อ *Public Benchmark Datasets*
> รันซ้ำได้ด้วย `python experiments/profile_tier2_datasets.py` ตัวเลขอยู่ใน `results/tier2_dataset_profile.csv`

---

## Metrics ที่ใช้ประเมิน

สัญลักษณ์: $N$ = จำนวนสังเกต, $M$ = จำนวน chunk, $M_0$ = จำนวน in-control chunk, $D$ = จำนวนมิติ, $[L_d, R_d]$ = ขอบเขตมิติที่ $d$, $C$ = เกณฑ์แจ้งเตือน

### 1. Coverage Rate (marginal)

$$\text{CR} = \frac{\sum_{d=1}^{D}\sum_{t=1}^{N} \mathbf{1}[L_d \leq x_{t,d} \leq R_d]}{D \cdot N}$$

### 2. Joint Coverage (ใหม่)

$$\text{JCR} = \frac{\sum_{t=1}^{N} \mathbf{1}[\forall d: L_d \leq x_{t,d} \leq R_d]}{N}$$

### 3. Sample FAR

$$\text{FAR}_{\text{sample}} = 1 - \text{CR}$$

### 4. Chunk FAR

$$V_m^{(d)} = \sum_{t \in \mathcal{C}_m} \mathbf{1}[x_{t,d} \notin [L_d, R_d]], \qquad A_m = \mathbf{1}[\exists\, d : V_m^{(d)} \geq C]$$

$$\text{FAR}_{\text{chunk}} = \frac{\sum_{m} A_m \cdot \mathbf{1}[\text{label}_m = 0]}{M_0}$$

### 5. ARL0 / ARL1

$$\widehat{\text{ARL}}_0 = \text{mean}(r_1,\ldots,r_K), \qquad \widehat{\text{ARL}}_1 = \text{mean}(\delta_1,\ldots,\delta_J)$$

### 6. Peak RAM / 7. Latency

$$\text{Peak RAM} = \frac{1}{1024}\Big(\texttt{sizeof}(\text{chart}) + \sum_{d}[\texttt{sizeof}(\text{engine}_d) + \texttt{sizeof}(\text{result}_d)]\Big), \qquad \text{Latency} = \frac{1}{M}\sum_m \Delta t_m$$

### ⚠️ กับดักของเมตริกที่ต้องระวัง

| ปัญหา | รายละเอียด |
|---|---|
| **ARL1 = 1.00 กำกวม** | `_compute_arl1` คืนค่า 1.0 เมื่อ**ไม่เคยตรวจเจออะไรเลย** ซึ่งอ่านแล้วเหมือน "ตรวจเจอทันที" ต้องอ่านคู่กับจำนวนที่ตรวจเจอเสมอ |
| **ARL0 เป็นค่า censored** | เมื่อไม่มี false alarm `_compute_arl0` คืนจำนวน in-control chunk ทั้งหมด (38.00/41.00/44.00 บน TEP) เป็น **lower bound** ไม่ใช่ค่าประมาณ เทียบเป็นตัวเลขกับ ARL0 = 0.06 ของ baseline ไม่ได้ |
| **Chunk FAR = 0.00% วัดไม่ได้** | TEP มี in-control chunk แค่ 38–44 chunk ค่า 0/38 รองรับได้แค่ "ต่ำกว่า 9.25% ที่ความเชื่อมั่น 95%" (Clopper-Pearson) ไม่ใช่ "= 0%" |
| **Chunk FAR ต้องมาคู่กับ detection** | ตัวตรวจจับที่ไม่เคยเตือนได้ 0.00% เสมอ |
| **RNG ไม่ได้ seed** | `np.random.choice` ใน `bootstrap_online.py:239` ไม่ได้ตั้ง seed และสคริปต์ Tier 2 ก็ไม่ได้ตั้ง ผลจึงต่างกันเล็กน้อยทุกครั้ง (วัดได้ ±0.01 pp ไม่กระทบข้อสรุป แต่ reproduce เป๊ะไม่ได้) |

---

## Changelog: สิ่งที่เปลี่ยนจากฉบับก่อน

| # | เปลี่ยนอะไร | เพราะอะไร |
|---|---|---|
| 1 | `C_thresh = 3` → `C = ceil(0.05·k)` | ค่าคงที่ไม่ scale-free กับขนาด chunk |
| 2 | Baselines เปลี่ยนจากนับรวมทุกมิติ → นับทีละมิติ | เดิม baselines รวม violation ทั้ง $D$ มิติแล้วเทียบกับ $C$ ส่วน RBULT ดูทีละมิติ ที่ $C$ เท่ากันจึงเป็นเงื่อนไขที่ต่างกันมาก **เอื้อ RBULT** และยิ่ง $D$ สูงยิ่งเอื้อมาก |
| 3 | ถอนข้ออ้าง "Chunk FAR 0.00% vs baselines 92–100%" | เกิดจาก (2) รวมกับบั๊ก `sample_ooc_count` |
| 4 | ถอนข้ออ้าง "robust ต่อ C_thresh" | เกิดจากบั๊ก `exp_tep_sensitivity.py:56` |
| 5 | แก้ memory savings TEP จาก ~156,000× → ~180× | baseline ที่ให้เลข 504,425.95 KB ไม่มีในโค้ดแล้ว |
| 6 | เพิ่ม Joint Coverage | เมตริกใหม่ เผยว่า marginal coverage เป็นข้อความที่อ่อนกว่าที่ดูเหมือนมาก |
| 7 | เพิ่มการวิเคราะห์ PER-FEATURE vs TOTAL | พบว่า TEP Mode 3 มีสัญญาณจริง (AUC 0.448 → 0.859) |
| 8 | เพิ่มหมายเหตุกับดักเมตริก 5 ข้อ | ARL1 fallback, ARL0 censored, CI ของ FAR, detection, RNG |
| 9 | เพิ่มข้อจำกัดของชุดข้อมูล | Pump ไม่มี in-control chunk, AI4I มี 6, TEP มี 38–44 |
| 10 | AI4I: เปลี่ยนจาก `Tool wear Rate [min diff]` → `Tool wear [min]` ค่าดิบ | ฟีเจอร์ diff สร้าง artifact — 98.8% ของความกว้างมาจากจุดเปลี่ยนเครื่องมือ 119 จุด และ **100% ของ violation บนมิตินั้นคือการเปลี่ยนเครื่องมือ ไม่ใช่ความผิดปกติของกระบวนการ** ค่า coverage 98.81% ของมันคือ 100% − 1.19% เฉย ๆ ผลคือ coverage รวมของ AI4I ลดจาก 98.40% → 97.79%, sample FAR 1.60% → 2.21%, joint 94.23% → 91.34% |
| 11 | เพิ่มเมตริก Interval Width เข้า Tier 2 | `mean_interval_width`, `final_interval_width`, `sigma_L/R`, `width_ratio_local`, `width_ratio_global` — Tier 1 มีอยู่แล้วแต่ Tier 2 ไม่มี ทำให้ coverage ถูกอ่านลำพังโดยไม่มีตัวคู่ เก็บด้วย Welford จึงยังเป็น $O(D)$ |
| 12 | เพิ่ม One-Step-Ahead (prequential) coverage เข้า Tier 2 | เดิม Tier 2 รายงานเฉพาะ in-sample ที่ใช้ขอบเขตสุดท้ายวัดย้อนหลัง ขณะที่ Tier 1 รายงานสอง protocol มาตลอด ช่องว่างสัมพันธ์กับ `width_ratio_local` (r = 0.686 marginal, 0.782 joint) ตั้งแต่ 0.006 จุด (Industrial Pump) ถึง 11.87 จุด joint (Water Pump) |
| 13 | TEP: chunk size 500 → 600 และเพิ่ม within-run differencing | k=600 ทำให้ 1 chunk = 1 run in-control chunks เพิ่มจาก 38–44 เป็น 100 ทุก mode; differencing ทำให้ **Mode 3 พลิกจาก Chunk FAR 100% เป็น 0.00%** และ joint 17.8% → 59.1% ทุก mode ดีขึ้นทุกเมตริก แต่ AUC รายมิติลดบน Mode 1/4 พบข้อจำกัดใหม่: ความกว้างของช่วงขึ้นกับ k แบบไม่เป็นเชิงเดียว |
| 14 | เมตริกที่ไม่นิยามคืน NaN แทน 0.00 / 1.00 | แก้ fallback 3 จุด: Chunk FAR และ ARL0 เมื่อ in-control chunk = 0, และ ARL1 เมื่อไม่เคยตรวจเจอ เปิดโปงว่า **Industrial Pump ทั้ง 4 method ไม่นิยาม** และ **Sliding-Window Bootstrap ไม่เคยเตือนเลยบน TEP ทั้ง 8 การรัน** (ARL1 = 1.00 เดิม เป็น fallback) เพิ่ม label quality gate ใน `exp_pump_benchmark.py` |

### ไฟล์อ้างอิงใน `results/`

| ไฟล์ | เนื้อหา |
|---|---|
| `TIER2_RERUN_SUMMARY.md` | รายงานหลักของการรันใหม่ |
| `tier2_final_all_methods.csv` | ตัวเลขดิบ 4 method × 8 ชุดข้อมูล |
| `spc_cthresh_sweep_report.md` | C_thresh ที่ตารางเดิมใช้ + joint coverage |
| `spc_threshold_rules_report.md` | กฎ binomial / empirical quantile บน Phase I |
| `spc_pct_threshold_report.md` | กฎ % ของ k + AUC |
| `spc_total_vs_perfeature_report.md` | PER-FEATURE vs TOTAL |

---

*สรุปโดย Claude (Cowork) จากไฟล์ผลการทดลองของ Prem Junsawang — RBULT Project*
