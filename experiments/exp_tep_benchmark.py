"""
SPC Benchmark Experiment: Tennessee Eastman Process (TEP) Dataset
===================================================================

Evaluates RBULT-SPC (Memory-Bounded Adaptive Control Chart) against
three benchmark SPC baselines on time-series telemetry from TEP Mode 1:
  1. Baseline Shewhart X-Bar Control Chart
  2. Baseline EWMA Control Chart
  3. Baseline Conventional Full-History Bootstrap Chart
  4. Proposed RBULT-SPC Framework

Monitored Telemetry Channels (D = 34):
  - 34 sensor telemetry variables (measured & manipulated variables)

Ground-Truth Target Label:
  - Label 0 = In-Control Normal Operation
  - Label 1..28 = 28 distinct industrial disturbance scenarios
"""

import os
import sys
import time
import pickle
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from online_bootstrap.spc_rbult import RBULTControlChart


def load_and_preprocess_tep_data(pickle_path: str = 'TEPDataset_M1_M5/TEPDataset_Mode1.pickle') -> pd.DataFrame:
    """Load TEP dataset pickle and flatten runs into continuous time-series stream."""
    print(f"Loading TEP dataset: {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    signals = data['Signals']  # Shape: (2900, 600, 34)
    labels = data['Labels']    # Shape: (2900,)

    num_runs, seq_len, num_vars = signals.shape
    print(f"Loaded TEP dataset shape: {num_runs} runs x {seq_len} time points x {num_vars} variables")

    # Reshape signals into continuous time-series DataFrame
    flattened_signals = signals.reshape(-1, num_vars)
    feature_names = [f"sensor_{i:02d}" for i in range(num_vars)]
    
    df = pd.DataFrame(flattened_signals, columns=feature_names)

    # Broadcast labels across 600 time points per run
    flattened_labels = np.repeat(labels, seq_len)
    df['fault_class'] = flattened_labels
    df['failure_label'] = (flattened_labels > 0).astype(int)

    num_failures = df['failure_label'].sum()
    print(f"Total stream samples: {len(df):,}")
    print(f"Fault/Disturbance samples: {num_failures:,} ({num_failures / len(df) * 100:.2f}% of stream)")

    return df, feature_names


def run_tep_spc_benchmark(pickle_path: str = 'TEPDataset_M1_M5/TEPDataset_Mode1.pickle',
                          mode_label: str = 'Mode 1',
                          csv_output: str = 'results/spc_tep_benchmark_results.csv',
                          md_output: str = 'results/spc_tep_benchmark_comparison.md',
                          chunk_size: int = 100,
                          window_size: int = 2000,
                          outlier_filter: bool = True,
                          ooc_threshold_count: int = 3) -> dict:
    """Run streaming SPC benchmark comparing 4 methods on TEP dataset.

    Args:
        pickle_path: Path to TEP pickle file.
        mode_label: Label for the TEP mode being evaluated.
        csv_output: Path to save result metrics CSV.
        md_output: Path to save result markdown table.
        chunk_size: Streaming chunk size (samples per batch, default 100).
        window_size: Sliding window size W for conventional bootstrap (default 2000).
        outlier_filter: Whether Z-score spike filtering (Algorithm 4) is enabled.
        ooc_threshold_count: Minimum sample violations per chunk to flag chunk OOC.

    Returns:
        Dict of results for all evaluated methods.
    """
    df, features = load_and_preprocess_tep_data(pickle_path)

    label_col = 'failure_label'
    num_chunks = int(np.ceil(len(df) / chunk_size))

    print(f"\nMonitored features ({len(features)}): D = {len(features)} channels")
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
    ooc_chunks = sum(1 for label in chunk_labels if label == 1)
    print(f"Chunk distribution -> In-Control Chunks: {in_control_chunks:,} | Fault Chunks: {ooc_chunks:,}")

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
        # Handle zero variance if any
        if sd == 0:
            sd = 1e-6
        shewhart_bounds[feat] = (mu - 3 * sd, mu + 3 * sd)

    shewhart_history = []
    shewhart_covered = 0
    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        t_start = time.perf_counter()

        ooc_cnt_chunk = 0
        for feat in features:
            vals = chunk_df[feat].values
            lcl, ucl = shewhart_bounds[feat]
            ooc_cnt_chunk += np.sum((vals < lcl) | (vals > ucl))
            shewhart_covered += np.sum((vals >= lcl) & (vals <= ucl))

        any_ooc = (ooc_cnt_chunk >= ooc_threshold_count)
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
        'false_alarm_rate': shewhart_fa / max(1, in_control_chunks),
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
        if sd == 0:
            sd = 1e-6
        margin = L_ewma * sd * np.sqrt(lam / (2.0 - lam))
        ewma_bounds[feat] = (mu - margin, mu + margin)

    ewma_history = []
    ewma_covered = 0

    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        t_start = time.perf_counter()
        ooc_cnt_chunk = 0

        for feat in features:
            vals = chunk_df[feat].values
            lcl, ucl = ewma_bounds[feat]
            for v in vals:
                ewma_stats[feat] = lam * v + (1.0 - lam) * ewma_stats[feat]
                if ewma_stats[feat] < lcl or ewma_stats[feat] > ucl:
                    ooc_cnt_chunk += 1
                else:
                    ewma_covered += 1

        any_ooc = (ooc_cnt_chunk >= ooc_threshold_count)
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
        'false_alarm_rate': ewma_fa / max(1, in_control_chunks),
        'arl_0': _compute_arl0(ewma_history, chunk_labels, in_control_chunks),
        'arl_1': _compute_arl1(ewma_history, chunk_labels),
        'total_time_sec': ewma_total_time
    }

    # ================================================================== #
    # 4. Baseline Method: Conventional Sliding-Window Bootstrap Chart    #
    # ================================================================== #
    print(f"\n--- [4/4] Running Baseline 3: Conventional Sliding-Window Bootstrap (W={window_size}) ---")
    start_time = time.perf_counter()
    conv_history = []
    conv_covered = 0
    history_buffer = {feat: deque(maxlen=window_size) for feat in features}
    conv_peak_memory = 0.0

    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        t_start = time.perf_counter()
        ooc_cnt_chunk = 0

        for feat in features:
            history_buffer[feat].extend(chunk_df[feat].tolist())

        # Memory footprint for sliding window buffer
        current_mem = sys.getsizeof(history_buffer) + sum(sys.getsizeof(v) for v in history_buffer.values())
        conv_peak_memory = max(conv_peak_memory, current_mem / 1024.0)

        for feat in features:
            hist_vals = np.array(history_buffer[feat])
            lcl, ucl = np.percentile(hist_vals, [0.5, 99.5])
            vals = chunk_df[feat].values
            ooc_cnt_chunk += np.sum((vals < lcl) | (vals > ucl))
            conv_covered += np.sum((vals >= lcl) & (vals <= ucl))

        any_ooc = (ooc_cnt_chunk >= ooc_threshold_count)
        t_latency = (time.perf_counter() - t_start) * 1000.0
        conv_history.append({'any_ooc': any_ooc, 'latency_ms': t_latency})

    conv_total_time = time.perf_counter() - start_time
    conv_fa = sum(1 for h, label in zip(conv_history, chunk_labels) if h['any_ooc'] and label == 0)
    conv_coverage = (conv_covered / (len(df) * len(features))) * 100.0

    conv_metrics = {
        'method': f'Baseline Sliding-Window Bootstrap (W={window_size})',
        'total_chunks': num_chunks,
        'total_samples': len(df),
        'avg_latency_ms': np.mean([h['latency_ms'] for h in conv_history]),
        'peak_memory_kb': conv_peak_memory,
        'total_ooc_chunks': sum(1 for h in conv_history if h['any_ooc']),
        'ooc_chunk_rate': sum(1 for h in conv_history if h['any_ooc']) / num_chunks,
        'overall_coverage_pct': conv_coverage,
        'sample_far_pct': 100.0 - conv_coverage,
        'false_alarm_rate': conv_fa / max(1, in_control_chunks),
        'arl_0': _compute_arl0(conv_history, chunk_labels, in_control_chunks),
        'arl_1': _compute_arl1(conv_history, chunk_labels),
        'total_time_sec': conv_total_time
    }

    # ================================================================== #
    # 5. Benchmark Summary Display & Export                              #
    # ================================================================== #
    all_metrics = [shewhart_metrics, ewma_metrics, conv_metrics, rbult_metrics]

    print("\n=====================================================================================================================")
    print("                                TENNESSEE EASTMAN PROCESS SPC BENCHMARK COMPARISON MATRIX                            ")
    print("=====================================================================================================================")
    print(f"{'Evaluation Metric':<32} | {'Shewhart Chart':<18} | {'EWMA Chart':<18} | {'Sliding-Win Bootstrap':<20} | {'Proposed RBULT-SPC':<20}")
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
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv(csv_output, index=False)

    md_table = f"""# Full SPC Benchmark Results: Tennessee Eastman Process ({mode_label})

| Evaluation Metric | Baseline Shewhart Chart | Baseline EWMA Chart | Baseline Sliding-Window Bootstrap (W={window_size}) | Proposed RBULT-SPC | Advantage / Key Discussion |
|---|---|---|---|---|---|
| **Overall Coverage Rate (%)** | {shewhart_metrics['overall_coverage_pct']:.2f}% | {ewma_metrics['overall_coverage_pct']:.2f}% | {conv_metrics['overall_coverage_pct']:.2f}% | **{rbult_metrics['overall_coverage_pct']:.2f}%** | High-Dimensional Non-Gaussian Coverage |
| **Sample-level FAR (%)** | {shewhart_metrics['sample_far_pct']:.2f}% | {ewma_metrics['sample_far_pct']:.2f}% | {conv_metrics['sample_far_pct']:.2f}% | **{rbult_metrics['sample_far_pct']:.2f}%** | Controlled near Bonferroni $\\alpha_{{dim}}$ |
| **Chunk-level FAR (%)** | {shewhart_metrics['false_alarm_rate'] * 100:.2f}% | {ewma_metrics['false_alarm_rate'] * 100:.2f}% | {conv_metrics['false_alarm_rate'] * 100:.2f}% | **{rbult_metrics['false_alarm_rate'] * 100:.2f}%** | Low Batch False Alarm Rate |
| **ARL0 (In-Control Run Length)** | {shewhart_metrics['arl_0']:.2f} | {ewma_metrics['arl_0']:.2f} | {conv_metrics['arl_0']:.2f} | **{rbult_metrics['arl_0']:.2f}** | Boundary Stability |
| **ARL1 (Detection Delay)** | {shewhart_metrics['arl_1']:.2f} | {ewma_metrics['arl_1']:.2f} | {conv_metrics['arl_1']:.2f} | **{rbult_metrics['arl_1']:.2f}** | Fast Failure Response |
| **Peak Memory Footprint (KB)** | {shewhart_metrics['peak_memory_kb']:.2f} KB | {ewma_metrics['peak_memory_kb']:.2f} KB | {conv_metrics['peak_memory_kb']:.2f} KB | **{rbult_metrics['peak_memory_kb']:.2f} KB** | **Constant $O(D)$ RAM Footprint** |
| **Avg Latency per Chunk (ms)** | {shewhart_metrics['avg_latency_ms']:.4f} ms | {ewma_metrics['avg_latency_ms']:.4f} ms | {conv_metrics['avg_latency_ms']:.4f} ms | **{rbult_metrics['avg_latency_ms']:.4f} ms** | Real-time Stream Execution |
"""
    with open(md_output, 'w') as f:
        f.write(md_table)

    print(f"\nSaved benchmark summary to: {csv_output}")
    print(f"Saved comparison table to:  {md_output}")

    return rbult_metrics


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
    return float(np.mean(delays)) if delays else 1.0


if __name__ == '__main__':
    # Execution target: Mode 5 (10/90 Mass Ratio, Max Production Rate)
    run_tep_spc_benchmark(
        pickle_path='TEPDataset_M1_M5/TEPDataset_Mode5.pickle',
        mode_label='Mode 5 (10/90 Mass Ratio, Max Rate)',
        csv_output='results/spc_tep_mode5_benchmark_results.csv',
        md_output='results/spc_tep_mode5_benchmark_comparison.md',
        chunk_size=500
    )


