"""
SPC Benchmark Experiment: Large Industrial Pump Maintenance Dataset
====================================================================

Evaluates RBULT-SPC (Memory-Bounded Adaptive Control Chart) against
three benchmark SPC baselines on time-series telemetry from industrial pumps:
  1. Baseline Shewhart X-Bar Control Chart
  2. Baseline EWMA Control Chart
  3. Baseline Conventional Full-History Bootstrap Chart
  4. Proposed RBULT-SPC Framework

Monitored Telemetry Channels (D = 5):
  - Temperature (°C)
  - Vibration (mm/s)
  - Pressure (kPa)
  - Flow_Rate (L/min)
  - RPM (rotations per minute)

Ground-Truth Target Label:
  - Maintenance_Flag (1 = Maintenance/Failure required, 0 = In-Control normal)
"""

import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from online_bootstrap.spc_rbult import RBULTControlChart

# Default chunk alarm threshold as a fraction of chunk size: C = ceil(q * k)
CHUNK_ALARM_RATE = 0.05


def load_and_preprocess_pump_data(csv_path: str = 'Large_Industrial_Pump_Maintenance_Dataset.csv') -> pd.DataFrame:
    """Load and sort Pump Maintenance dataset by Pump_ID and Operational_Hours."""
    print(f"Loading Industrial Pump dataset: {csv_path}...")
    df = pd.read_csv(csv_path)

    # Sort by Pump_ID and Operational_Hours to establish time-series stream order
    df = df.sort_values(by=['Pump_ID', 'Operational_Hours']).reset_index(drop=True)

    print(f"Total rows loaded: {len(df):,}")
    num_failures = df['Maintenance_Flag'].sum()
    print(f"Maintenance/Failure samples: {num_failures:,} ({num_failures / len(df) * 100:.2f}% of stream)")
    return df


def run_pump_spc_benchmark(csv_path: str = 'Large_Industrial_Pump_Maintenance_Dataset.csv',
                           chunk_size: int = 200,
                           outlier_filter: bool = True,
                           ooc_threshold_count: Optional[int] = None) -> dict:
    """Run streaming SPC benchmark comparing 4 methods on Industrial Pump dataset.

    Args:
        csv_path: Path to pump dataset CSV.
        chunk_size: Streaming chunk size (samples per batch, default 200).
        outlier_filter: Whether Z-score spike filtering (Algorithm 4) is enabled.
        ooc_threshold_count: Minimum sample violations per chunk to flag chunk OOC.

    Returns:
        Dict of results for all 4 evaluated methods.
    """
    df = load_and_preprocess_pump_data(csv_path)

    # Monitored telemetry features (D = 5)
    features = [
        'Temperature',
        'Vibration',
        'Pressure',
        'Flow_Rate',
        'RPM'
    ]

    label_col = 'Maintenance_Flag'
    num_chunks = int(np.ceil(len(df) / chunk_size))

    # Chunk alarm threshold: scale-free rate rule C = ceil(CHUNK_ALARM_RATE * k).
    # An absolute count is not scale-free -- the number of violations an in-control
    # chunk carries grows with k. Applied to every method for a fair comparison.
    if ooc_threshold_count is None:
        ooc_threshold_count = max(1, int(np.ceil(CHUNK_ALARM_RATE * chunk_size)))

    print(f"\nMonitored features ({len(features)}): {features}")
    print(f"Streaming chunk size: {chunk_size} samples per chunk")
    print(f"Total streaming chunks: {num_chunks:,}")
    print(f"Chunk Alarm Threshold: >= {ooc_threshold_count} sample violations per chunk")

    # Generate chunk-level true failure labels
    chunk_labels = []
    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        true_ooc = 1 if (label_col in chunk_df.columns and chunk_df[label_col].sum() > 0) else 0
        chunk_labels.append(true_ooc)

    in_control_chunks = sum(1 for label in chunk_labels if label == 0)

    # ------------------------------------------------------------------ #
    # Label quality gate                                                   #
    # ------------------------------------------------------------------ #
    # Maintenance_Flag on this dataset is statistically indistinguishable from an
    # independent coin flip: no monitored variable predicts it (AUC 0.49-0.51, all
    # p > 0.05), its rate is ~50% in every one of the five pumps, and its mean
    # contiguous run length of 1.99 rows matches the 1.99 implied by independent
    # Bernoulli(0.4984) draws. It therefore carries no ground truth, and every chunk
    # contains at least one flagged sample, leaving zero in-control chunks.
    #
    # Chunk FAR, ARL0 and ARL1 are undefined here and are reported as NaN rather than
    # the 0.00 the old max(1, in_control_chunks) guard produced -- which read as
    # flawless in-control behaviour. Sample-level metrics (coverage, sample FAR,
    # interval width, RAM, latency) do not depend on the label and remain valid.
    if in_control_chunks == 0:
        print("\n" + "!" * 78)
        print("WARNING: no in-control chunks -- chunk-level metrics are UNDEFINED")
        print(f"  label column      : {label_col}")
        print(f"  positive rate     : {df[label_col].mean() * 100:.2f}% of rows")
        print(f"  mean run length   : {df[label_col].sum() / max(1, int((np.diff(np.r_[0, df[label_col].values, 0]) == 1).sum())):.2f} rows")
        print("  Chunk FAR / ARL0 / ARL1 will be reported as NaN.")
        print("  Only sample-level metrics are interpretable on this dataset.")
        print("!" * 78)
    ooc_chunks = sum(1 for label in chunk_labels if label == 1)
    print(f"Chunk distribution -> In-Control Chunks: {in_control_chunks:,} | Maintenance/Failure Chunks: {ooc_chunks:,}")

    # ================================================================== #
    # 1. Proposed Method: RBULT-SPC Framework                            #
    # ================================================================== #
    print("\n--- [1/4] Running Proposed RBULT-SPC Framework ---")
    rbult_chart = RBULTControlChart(
        features=features,
        minmax_flag=False,
        outlier_filter=outlier_filter,
        alpha_sys=0.05,
        fwer_correction='bonferroni'
    )

    start_time = time.perf_counter()
    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        rbult_chart.update_chunk(chunk_df, ooc_threshold_count=ooc_threshold_count)

    rbult_total_time = time.perf_counter() - start_time
    rbult_metrics = rbult_chart.compute_spc_metrics(true_labels=chunk_labels, sample_df=df)
    rbult_metrics['total_time_sec'] = rbult_total_time
    rbult_metrics['method'] = 'Proposed RBULT-SPC'

    # ================================================================== #
    # 2. Baseline Method: Classical Shewhart X-Bar Chart                 #
    # ================================================================== #
    print("\n--- [2/4] Running Baseline 1: Classical Shewhart Chart ---")
    start_time = time.perf_counter()
    shewhart_bounds = {}
    phase1_df = df.iloc[0:chunk_size]
    for feat in features:
        mu = phase1_df[feat].mean()
        sd = phase1_df[feat].std()
        shewhart_bounds[feat] = (mu - 3 * sd, mu + 3 * sd)

    shewhart_history = []
    shewhart_covered = 0
    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        t_start = time.perf_counter()

        feat_ooc_counts = []
        for feat in features:
            vals = chunk_df[feat].values
            lcl, ucl = shewhart_bounds[feat]
            feat_ooc_counts.append(np.sum((vals < lcl) | (vals > ucl)))
            shewhart_covered += np.sum((vals >= lcl) & (vals <= ucl))

        # ANY single feature reaching the threshold flags the chunk, matching
        # RBULTControlChart.update_chunk and the Bonferroni per-dimension design.
        any_ooc = any(c >= ooc_threshold_count for c in feat_ooc_counts)
        t_latency = (time.perf_counter() - t_start) * 1000.0
        shewhart_history.append({'any_ooc': any_ooc, 'latency_ms': t_latency})

    shewhart_total_time = time.perf_counter() - start_time
    shewhart_fa = sum(1 for h, label in zip(shewhart_history, chunk_labels) if h['any_ooc'] and label == 0)
    shewhart_coverage = (shewhart_covered / (len(df) * len(features))) * 100.0

    shewhart_metrics = {
        'method': 'Baseline Shewhart Chart',
        'total_chunks': num_chunks,
        'total_samples': len(df),
        'avg_latency_ms': np.mean([h['latency_ms'] for h in shewhart_history]),
        'peak_memory_kb': sys.getsizeof(shewhart_bounds) / 1024.0,
        'total_ooc_chunks': sum(1 for h in shewhart_history if h['any_ooc']),
        'ooc_chunk_rate': sum(1 for h in shewhart_history if h['any_ooc']) / num_chunks,
        'overall_coverage_pct': shewhart_coverage,
        'sample_far_pct': 100.0 - shewhart_coverage,
        'false_alarm_rate': (shewhart_fa / in_control_chunks
                             if in_control_chunks > 0 else float('nan')),
        'arl_0': _compute_arl0(shewhart_history, chunk_labels, in_control_chunks),
        'arl_1': _compute_arl1(shewhart_history, chunk_labels),
        'total_time_sec': shewhart_total_time
    }

    # ================================================================== #
    # 3. Baseline Method: EWMA Control Chart (lambda=0.2, L=3)           #
    # ================================================================== #
    print("\n--- [3/4] Running Baseline 2: EWMA Control Chart ---")
    start_time = time.perf_counter()
    lam = 0.2
    L_ewma = 3.0
    ewma_stats = {feat: phase1_df[feat].mean() for feat in features}
    ewma_bounds = {}
    for feat in features:
        mu = phase1_df[feat].mean()
        sd = phase1_df[feat].std()
        margin = L_ewma * sd * np.sqrt(lam / (2.0 - lam))
        ewma_bounds[feat] = (mu - margin, mu + margin)

    ewma_history = []
    ewma_covered = 0

    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        t_start = time.perf_counter()
        feat_ooc_counts = []

        for feat in features:
            vals = chunk_df[feat].values
            lcl, ucl = ewma_bounds[feat]
            n_viol_feat = 0
            for v in vals:
                ewma_stats[feat] = lam * v + (1.0 - lam) * ewma_stats[feat]
                if ewma_stats[feat] < lcl or ewma_stats[feat] > ucl:
                    n_viol_feat += 1
                else:
                    ewma_covered += 1
            feat_ooc_counts.append(n_viol_feat)

        # ANY single feature reaching the threshold flags the chunk, matching
        # RBULTControlChart.update_chunk and the Bonferroni per-dimension design.
        any_ooc = any(c >= ooc_threshold_count for c in feat_ooc_counts)
        t_latency = (time.perf_counter() - t_start) * 1000.0
        ewma_history.append({'any_ooc': any_ooc, 'latency_ms': t_latency})

    ewma_total_time = time.perf_counter() - start_time
    ewma_fa = sum(1 for h, label in zip(ewma_history, chunk_labels) if h['any_ooc'] and label == 0)
    ewma_coverage = (ewma_covered / (len(df) * len(features))) * 100.0

    ewma_metrics = {
        'method': 'Baseline EWMA Chart',
        'total_chunks': num_chunks,
        'total_samples': len(df),
        'avg_latency_ms': np.mean([h['latency_ms'] for h in ewma_history]),
        'peak_memory_kb': (sys.getsizeof(ewma_bounds) + sys.getsizeof(ewma_stats)) / 1024.0,
        'total_ooc_chunks': sum(1 for h in ewma_history if h['any_ooc']),
        'ooc_chunk_rate': sum(1 for h in ewma_history if h['any_ooc']) / num_chunks,
        'overall_coverage_pct': ewma_coverage,
        'sample_far_pct': 100.0 - ewma_coverage,
        'false_alarm_rate': (ewma_fa / in_control_chunks
                             if in_control_chunks > 0 else float('nan')),
        'arl_0': _compute_arl0(ewma_history, chunk_labels, in_control_chunks),
        'arl_1': _compute_arl1(ewma_history, chunk_labels),
        'total_time_sec': ewma_total_time
    }

    # ================================================================== #
    # 4. Baseline Method: Conventional Full-History Bootstrap Chart      #
    # ================================================================== #
    print("\n--- [4/4] Running Baseline 3: Conventional Full-History Bootstrap ---")
    start_time = time.perf_counter()
    conv_history = []
    conv_covered = 0
    history_buffer = {feat: [] for feat in features}
    conv_peak_memory = 0.0

    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        t_start = time.perf_counter()
        feat_ooc_counts = []

        for feat in features:
            history_buffer[feat].extend(chunk_df[feat].tolist())

        current_mem = sys.getsizeof(history_buffer) + sum(sys.getsizeof(v) for v in history_buffer.values())
        conv_peak_memory = max(conv_peak_memory, current_mem / 1024.0)

        for feat in features:
            hist_vals = np.array(history_buffer[feat])
            lcl, ucl = np.percentile(hist_vals, [0.5, 99.5])
            vals = chunk_df[feat].values
            feat_ooc_counts.append(np.sum((vals < lcl) | (vals > ucl)))
            conv_covered += np.sum((vals >= lcl) & (vals <= ucl))

        # ANY single feature reaching the threshold flags the chunk, matching
        # RBULTControlChart.update_chunk and the Bonferroni per-dimension design.
        any_ooc = any(c >= ooc_threshold_count for c in feat_ooc_counts)
        t_latency = (time.perf_counter() - t_start) * 1000.0
        conv_history.append({'any_ooc': any_ooc, 'latency_ms': t_latency})

    conv_total_time = time.perf_counter() - start_time
    conv_fa = sum(1 for h, label in zip(conv_history, chunk_labels) if h['any_ooc'] and label == 0)
    conv_coverage = (conv_covered / (len(df) * len(features))) * 100.0

    conv_metrics = {
        'method': 'Baseline Full-History Bootstrap',
        'total_chunks': num_chunks,
        'total_samples': len(df),
        'avg_latency_ms': np.mean([h['latency_ms'] for h in conv_history]),
        'peak_memory_kb': conv_peak_memory,
        'total_ooc_chunks': sum(1 for h in conv_history if h['any_ooc']),
        'ooc_chunk_rate': sum(1 for h in conv_history if h['any_ooc']) / num_chunks,
        'overall_coverage_pct': conv_coverage,
        'sample_far_pct': 100.0 - conv_coverage,
        'false_alarm_rate': (conv_fa / in_control_chunks
                             if in_control_chunks > 0 else float('nan')),
        'arl_0': _compute_arl0(conv_history, chunk_labels, in_control_chunks),
        'arl_1': _compute_arl1(conv_history, chunk_labels),
        'total_time_sec': conv_total_time
    }

    # ================================================================== #
    # 5. Benchmark Summary Display & Export                              #
    # ================================================================== #
    all_metrics = [shewhart_metrics, ewma_metrics, conv_metrics, rbult_metrics]

    print("\n=====================================================================================================================")
    print("                                INDUSTRIAL PUMP SPC BENCHMARK COMPARISON MATRIX                                      ")
    print("=====================================================================================================================")
    print(f"{'Evaluation Metric':<32} | {'Shewhart Chart':<18} | {'EWMA Chart':<18} | {'Full-Hist Bootstrap':<20} | {'Proposed RBULT-SPC':<20}")
    print("-" * 118)
    print(f"{'Overall Coverage Rate (%)':<32} | {shewhart_metrics['overall_coverage_pct']:<18.2f} | {ewma_metrics['overall_coverage_pct']:<18.2f} | {conv_metrics['overall_coverage_pct']:<20.2f} | {rbult_metrics['overall_coverage_pct']:<20.2f}")
    print(f"{'Sample-level FAR (%)':<32} | {shewhart_metrics['sample_far_pct']:<18.2f} | {ewma_metrics['sample_far_pct']:<18.2f} | {conv_metrics['sample_far_pct']:<20.2f} | {rbult_metrics['sample_far_pct']:<20.2f}")
    print(f"{'Chunk-level FAR (%)':<32} | {shewhart_metrics['false_alarm_rate'] * 100:<18.2f} | {ewma_metrics['false_alarm_rate'] * 100:<18.2f} | {conv_metrics['false_alarm_rate'] * 100:<20.2f} | {rbult_metrics['false_alarm_rate'] * 100:<20.2f}")
    print(f"{'ARL0 (In-Control Run Length)':<32} | {shewhart_metrics['arl_0']:<18.2f} | {ewma_metrics['arl_0']:<18.2f} | {conv_metrics['arl_0']:<20.2f} | {rbult_metrics['arl_0']:<20.2f}")
    print(f"{'ARL1 (Detection Delay)':<32} | {shewhart_metrics['arl_1']:<18.2f} | {ewma_metrics['arl_1']:<18.2f} | {conv_metrics['arl_1']:<20.2f} | {rbult_metrics['arl_1']:<20.2f}")
    print(f"{'Peak Memory Footprint (KB)':<32} | {shewhart_metrics['peak_memory_kb']:<18.2f} | {ewma_metrics['peak_memory_kb']:<18.2f} | {conv_metrics['peak_memory_kb']:<20.2f} | {rbult_metrics['peak_memory_kb']:<20.2f}")
    print(f"{'Avg Latency per Chunk (ms)':<32} | {shewhart_metrics['avg_latency_ms']:<18.4f} | {ewma_metrics['avg_latency_ms']:<18.4f} | {conv_metrics['avg_latency_ms']:<20.4f} | {rbult_metrics['avg_latency_ms']:<20.4f}")
    print("=====================================================================================================================")

    # Save CSV and Markdown report
    os.makedirs('results', exist_ok=True)
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv('results/spc_pump_benchmark_results.csv', index=False)

    chunk_note = ('**UNDEFINED — no in-control chunks** (see label quality note)'
                  if in_control_chunks == 0 else 'Batch-level alarm behaviour')

    md_table = f"""# Full SPC Benchmark Results: Large Industrial Pump Dataset

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Full-History Bootstrap | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | {shewhart_metrics['overall_coverage_pct']:.2f}% | {ewma_metrics['overall_coverage_pct']:.2f}% | {conv_metrics['overall_coverage_pct']:.2f}% | **{rbult_metrics['overall_coverage_pct']:.2f}%** | Non-Gaussian Adaptive Bounds |
| **Sample-level FAR (%)** | {shewhart_metrics['sample_far_pct']:.2f}% | {ewma_metrics['sample_far_pct']:.2f}% | {conv_metrics['sample_far_pct']:.2f}% | **{rbult_metrics['sample_far_pct']:.2f}%** | Controlled near Bonferroni $\\alpha_{{dim}}$ |
| **Chunk-level FAR (%)** | {_fmt(shewhart_metrics['false_alarm_rate'] * 100, '%')} | {_fmt(ewma_metrics['false_alarm_rate'] * 100, '%')} | {_fmt(conv_metrics['false_alarm_rate'] * 100, '%')} | **{_fmt(rbult_metrics['false_alarm_rate'] * 100, '%')}** | {chunk_note} |
| **ARL0 (In-Control Run Length)** | {_fmt(shewhart_metrics['arl_0'])} | {_fmt(ewma_metrics['arl_0'])} | {_fmt(conv_metrics['arl_0'])} | **{_fmt(rbult_metrics['arl_0'])}** | {chunk_note} |
| **ARL1 (Detection Delay)** | {_fmt(shewhart_metrics['arl_1'])} | {_fmt(ewma_metrics['arl_1'])} | {_fmt(conv_metrics['arl_1'])} | **{_fmt(rbult_metrics['arl_1'])}** | {chunk_note} |
| **Peak Memory Footprint (KB)** | {shewhart_metrics['peak_memory_kb']:.2f} KB | {ewma_metrics['peak_memory_kb']:.2f} KB | {conv_metrics['peak_memory_kb']:.2f} KB | **{rbult_metrics['peak_memory_kb']:.2f} KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | {shewhart_metrics['avg_latency_ms']:.4f} ms | {ewma_metrics['avg_latency_ms']:.4f} ms | {conv_metrics['avg_latency_ms']:.4f} ms | **{rbult_metrics['avg_latency_ms']:.4f} ms** | Real-time Streaming (< 70 ms) |
"""
    with open('results/spc_pump_benchmark_comparison.md', 'w') as f:
        f.write(md_table)

    print("\nSaved benchmark summary to: results/spc_pump_benchmark_results.csv")
    print("Saved comparison table to:  results/spc_pump_benchmark_comparison.md")

    return rbult_metrics


def _fmt(value: float, suffix: str = '') -> str:
    """Render a metric, showing an em dash for an undefined (NaN) value.

    Chunk FAR, ARL0 and ARL1 are undefined when a dataset has no in-control chunks.
    Printing 'nan' beside a caption like "Low Batch False Alarm Rate" invites the
    reader to treat it as a result, so undefined values are rendered explicitly.
    """
    import math
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '—'
    return f'{value:.2f}{suffix}'


def _compute_arl0(history: list, labels: list, in_control_chunks: int) -> float:
    runs = []
    curr = 0
    for h, label in zip(history, labels):
        if label == 0:
            if h['any_ooc']:
                runs.append(curr)
                curr = 0
            else:
                curr += 1
    if curr > 0:
        runs.append(curr)
    # NaN when there is no in-control data; otherwise, with no false alarm the
    # value is right-censored at the observation window and returned as a bound.
    if in_control_chunks == 0:
        return float('nan')
    return float(np.mean(runs)) if runs else float(in_control_chunks)


def _compute_arl1(history: list, labels: list) -> float:
    delays = []
    delay = 0
    for h, label in zip(history, labels):
        if label == 1:
            delay += 1
            if h['any_ooc']:
                delays.append(delay)
                delay = 0
        else:
            delay = 0
    # NaN, not 1.0, when nothing was ever detected -- the old fallback was
    # indistinguishable from instant detection.
    return float(np.mean(delays)) if delays else float('nan')


if __name__ == '__main__':
    run_pump_spc_benchmark(chunk_size=200)
