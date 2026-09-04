# boostraponline_project

Online (streaming) bootstrap for non-parametric interval estimation, and its application to
Statistical Process Control on high-dimensional IoT data streams.

- **[Paper 1 — RBULT-SPC](#paper-1--rbult-spc)** — the current experiment suite
- **[Legacy simulation pipeline](#legacy-simulation-pipeline)** — the earlier 1D population/chunk workflow

---

# Paper 1 — RBULT-SPC

**RBULT-SPC** (Resource-Bounded Upper/Lower Tail — Statistical Process Control) is a
memory-bounded adaptive control chart for non-Gaussian IoT streams. It keeps only
$O(D)$ tail summary statistics instead of the $O(N \cdot D)$ history a conventional
bootstrap needs.

Experiments are organised in two tiers:

| Tier | What it establishes |
|---|---|
| **Tier 1** | Correctness against ground truth on 1D synthetic data (5 distributions × 7 noise scenarios) |
| **Tier 2** | Practical behaviour on 8 real multivariate industrial streams ($D = 5 \dots 34$, $N$ up to 1.74 M) |

## Reading order — Markdown documents

Read top to bottom. Items 1–2 are enough to understand the current state; 3–7 are the
manuscript materials; 8–11 are the supporting analyses behind specific decisions.

| # | File | What it gives you |
|---|---|---|
| **1** | **`paper1_experiment_summary.md`** | **Start here.** Complete Thai-language summary of every experiment, all result tables, what holds up and what was withdrawn, plus a 9-item changelog and the metric pitfalls. |
| **2** | `results/TIER2_RERUN_SUMMARY.md` | The Tier 2 re-run in English: resolved thresholds, RBULT vs. bootstrap baseline head-to-head, and 7 open issues that still need attention. |
| **3** | `research_plan.md` | The method itself — algorithms, mathematical derivations, metric definitions, and per-dataset result tables with discussion. The longest and most detailed document. |
| **4** | `section_experimental_results.md` | Draft of the paper's experimental-results section (AI4I and MetroPT-3 in depth). |
| **5** | `paper_draft.md` | Full paper outline: abstract, highlights, section-by-section plan. |
| **6** | `paper_plan.md` | Contribution list and the key result matrix, i.e. the paper's selling points. |
| **7** | `papar_draft/if_draf.tex` | The LaTeX manuscript itself (compiles to `if_draf.pdf`). |
| **8** | `results/spc_cthresh_sweep_report.md` | Which `C_thresh` produced each previously published table, and the marginal-vs-joint coverage gap. |
| **9** | `results/spc_threshold_rules_report.md` | Phase I calibrated threshold rules (binomial, empirical quantile) evaluated on MetroPT-3 and Water Pump. |
| **10** | `results/spc_pct_threshold_report.md` | Why the threshold is a fraction of chunk size, plus the AUC of the violation statistic per dataset. |
| **11** | `results/spc_total_vs_perfeature_report.md` | PER-FEATURE vs TOTAL violation aggregation, and why TEP Mode 3 needs the latter. |

> ⚠️ Documents 3–7 were written before the September 2026 re-run and have been corrected
> in place. Each carries a *Revision Note* or *Changelog* explaining what changed. If a
> number in them disagrees with `results/*.csv`, the CSV is authoritative.

## Core library — `online_bootstrap/`

| Module | Class | Responsibility |
|---|---|---|
| `bootstrap_online.py` | `BootstrapOnline` | The streaming bootstrap engine. `expand_bt_online()` grows the tail boundaries $[v_{\min}, v_{\max}]$ chunk by chunk without retaining raw data; also provides the offline `expand_bt_trad()` for baseline comparison. |
| `res_bootstrap_v2.py` | `ResBootstrap` | Collects per-chunk boundary/error history and provides the Plotly plotting helpers. |
| `spc_rbult.py` | `RBULTControlChart` | **The Paper 1 contribution.** Wraps one `BootstrapOnline` per feature into a multivariate control chart: Bonferroni FWER correction across $D$ dimensions, per-chunk OOC decisions, and the SPC metrics (coverage, joint coverage, FAR, ARL0/ARL1, RAM, latency). |
| `stat_dist.py` | — | Per-distribution tail-area tables for the 11 candidate distributions used in MLE fitting. |
| `BatchOutlierDetection.py` | — | Z-score spike filtering (Algorithm 4) applied before boundary updates. |

### Chunk alarm threshold

A chunk is flagged Out-of-Control when **any single feature** accumulates at least $C$
out-of-bound samples. `RBULTControlChart` resolves $C$ in this order:

1. An explicit `ooc_threshold_count` passed to `update_chunk()`
2. Per-dimension $C_d$ from `calibrate_phase1()`, if Phase I calibration was run
3. The default scale-free rate rule $C = \lceil 0.05 \cdot k \rceil$

Rule 3 is the default because an absolute count is not comparable across chunk sizes —
the number of violations an in-control chunk carries grows with $k$.

```python
from online_bootstrap.spc_rbult import RBULTControlChart

chart = RBULTControlChart(features=['s1', 's2'], alpha_sys=0.05)

# Optional: per-dimension thresholds from a known in-control window
chart.start_phase1()
for chunk in warmup_chunks:
    chart.update_chunk(chunk)
chart.calibrate_phase1(warmup_chunks=10)   # discard the boundary-convergence transient

for chunk in stream:
    summary = chart.update_chunk(chunk)     # C = ceil(0.05 * len(chunk)) unless calibrated
    if summary['any_ooc']:
        ...
metrics = chart.compute_spc_metrics(true_labels=labels, sample_df=df)
```

## Tier 1 — 1D synthetic benchmark

| Script | What it does |
|---|---|
| `experiments/exp_1d_noise_benchmark.py` | Runs the full 7-scenario gold-standard suite: 5 distributions × 7 noise scenarios (Clean, GAWN $0.1\sigma$–$0.3\sigma$, Impulse Spikes 1–10%) × 3 chunk sizes × 2 α levels × 3 methods × 2 evaluation protocols = **1,260 runs**. Compares Traditional Offline, Cumulative Online and RBULT under both *In-Sample Adaptation* and *One-Step-Ahead Pre-Sequential* protocols. Writes the results CSV and the summary table. |
| `experiments/plot_1d_noise_benchmark.py` | Renders the two figures from that CSV (coverage comparison, noise sensitivity ratio). Run it **after** the benchmark — the benchmark does not produce figures itself. |

```bash
python experiments/exp_1d_noise_benchmark.py    # ~10 min, RNG is seeded
python experiments/plot_1d_noise_benchmark.py
```

Outputs land in `results_1d_noise_benchmark/`. This tier does **not** import `spc_rbult.py`,
so changes to the control chart cannot affect it.

## Tier 2 — Multivariate industrial benchmarks

Each script streams one dataset through four methods — Shewhart (3σ), EWMA (λ=0.2, L=3), a
bootstrap baseline, and RBULT-SPC — and writes a results CSV plus a Markdown comparison table.
The chunk alarm rule is applied identically to all four.

| Script | Dataset | $D$ | $N$ | chunk $k$ | $C$ | Bootstrap baseline |
|---|---|---:|---:|---:|---:|---|
| `exp_spc_benchmark.py` | AI4I 2020 | 5 | 10,000 | 100 | 5 | Full-history |
| `exp_pump_benchmark.py` | Industrial Pump | 5 | 20,000 | 200 | 10 | Full-history |
| `exp_waterpump_benchmark.py` | Water Pump (SCADA) | 10 | 220,320 | 500 | 25 | Full-history |
| `exp_metropt3_benchmark.py` | MetroPT-3 compressor | 7 | 1,516,948 | 1000 | 50 | Full-history |
| `exp_tep_benchmark.py` | Tennessee Eastman Process | 34 | ~1.74 M | 500 | 25 | Sliding window (W=2000) |
| `exp_tep_sensitivity.py` | TEP Mode 1 | 34 | 1.74 M | 500 | swept | Sliding window |

```bash
python experiments/exp_spc_benchmark.py         # ~10 s
python experiments/exp_pump_benchmark.py        # ~30 s
python experiments/exp_waterpump_benchmark.py   # ~1 min
python experiments/exp_metropt3_benchmark.py    # ~4 min (full-history percentiles dominate)
python experiments/exp_tep_benchmark.py         # ~1 min per mode
python experiments/exp_tep_sensitivity.py       # ~1 min
```

**`exp_tep_benchmark.py` runs only Mode 5 from its `__main__` block.** To reproduce all four
operating regimes, call the function directly:

```python
from experiments.exp_tep_benchmark import run_tep_spc_benchmark

for mode, suffix in [('1', ''), ('3', '_mode3'), ('4', '_mode4'), ('5', '_mode5')]:
    run_tep_spc_benchmark(
        pickle_path=f'TEPDataset_M1_M5/TEPDataset_Mode{mode}.pickle',
        mode_label=f'Mode {mode}',
        csv_output=f'results/spc_tep{suffix}_benchmark_results.csv',
        md_output=f'results/spc_tep{suffix}_benchmark_comparison.md',
        chunk_size=500)
```

`exp_tep_sensitivity.py` streams TEP Mode 1 once and re-thresholds post hoc, so it evaluates
a whole list of thresholds for the cost of a single pass. Note it aggregates violations
**across** dimensions for all four methods, unlike the benchmark scripts above which evaluate
per dimension — the two are internally consistent but not comparable to each other.

### Control chart figures

Run after the corresponding benchmark; each renders LCL/UCL bands against the stream.

| Script | Output |
|---|---|
| `experiments/plot_pump_spc.py` | `results/spc_pump_chart.png` |
| `experiments/plot_waterpump_spc.py` | `results/spc_waterpump_chart.png` |
| `experiments/plot_metropt3_spc.py` | `results/spc_metropt3_chart.png` |
| `experiments/plot_tep_spc.py` | `results/spc_tep_mode5_chart.png` (edit `__main__` for other modes) |

### Datasets

`ai4i2020_Predictive Maintenance Dataset.csv` and
`Large_Industrial_Pump_Maintenance_Dataset.csv` are in the repository. The rest are
**gitignored for size** and must be placed manually before running:

| Path | Dataset |
|---|---|
| `sensor.csv` | Water Pump SCADA (~124 MB) |
| `MetroPT3/MetroPT3_AirCompressor.csv` | MetroPT-3 |
| `TEPDataset_M1_M5/TEPDataset_Mode{1,3,4,5}.pickle` | Tennessee Eastman Process |

### Reproducibility caveats

- **The RNG is not seeded** in Tier 2. `np.random.choice` in `bootstrap_online.py` runs
  unseeded, so RBULT numbers move slightly between runs. Measured spread is ±0.01 pp —
  harmless for conclusions, but exact reproduction is not possible. Tier 1 *is* seeded.
- **Chunk FAR, ARL0 and ARL1 must be read together with a detection count.** A detector that
  never fires scores a perfect 0.00% Chunk FAR; `_compute_arl1` returns `1.0` both when a
  fault is detected immediately and when nothing is ever detected; `_compute_arl0` returns
  the in-control chunk count (a censored lower bound) when no false alarm occurs.
- **Industrial Pump has zero in-control chunks**, so its Chunk FAR and ARL0 are undefined
  rather than zero. AI4I has only 6, and each TEP mode 38–44 out of ~3,480.

---

# Legacy simulation pipeline

The earlier 1D population/chunk workflow, kept for reference.

1. `sim_data_pop.py` or `sim_data_pop_v2.py` => simulate the population based on the predefined values in the `.yaml` file. For examples: `config_wald.yaml` and `config_wiebull.yaml`. The simulated data was saved in `.pkl` file.

2. `sim_data_samp_chunk.py` => create the samples data chunks from the population file simulated from `sim_data_pop.py` and save the results into `.json` file.

3. `lib_boostrap.py` => library file relating to the online boostrap functions.

4. `main_boostrap.py` => the main program for executing boostrap online algorithms.

5. `main_result_analysis.py` => the main program for analysing the results from `main_boostrap.py`

## main_boostrap.py

    Step 1: Read population data file saved as list (json file).
    Step 2: For each data:
        Step 3:  

### expand_bt_online

    step 1: Compute the learned samples.
    step 2: Compute the min and max values (min_c and max_c) of the current chunk c.
    step 3: If min_c < v_min, then v_min = min_c.
    step 4: If max_c > v_max, then v_max = max_c.
    step 5: Compute average (avg) and standard deviation (sd) based on v_min and v_max.
    step 6: Construct theoritical distribution of 8 bins based on avg and std.
    step 7: Compute theoritical number of elements in the left bin (h_l) and the right bin (h_r).
    step 8: Find the number of elements falls into the left bin (n_l) and and the right bin (n_r).
    step 9: If n_l> h_l, perform bootstrap on the left elements falling in the leftmost bin to get
            the update v_min.
    step 10: If n_r > h_r, perform boostrap on the right elements falling in the rightmost bin to get
            the update v_max.
    step 11: If v_max or v_min changed, go to step 5.
    step 12: Else stop.
