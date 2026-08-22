"""
SPC Benchmark Experiment: AI4I 2020 Predictive Maintenance Dataset
===================================================================

Runs RBULT-SPC (Memory-Bounded Adaptive Control Chart) on streaming sensor telemetry
from the AI4I 2020 Predictive Maintenance Dataset.

Monitored Features:
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]

Target: Machine failure
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from online_bootstrap.spc_rbult import RBULTControlChart



def run_spc_benchmark(csv_path: str = 'ai4i2020_Predictive Maintenance Dataset.csv',
                      chunk_size: int = 100,
                      outlier_filter: bool = True) -> dict:
    """Run real-time streaming SPC benchmark on the AI4I dataset.

    Args:
        csv_path: Path to dataset CSV file.
        chunk_size: Streaming chunk size (samples per batch).
        outlier_filter: Whether Z-score spike filtering (Algorithm 4) is enabled.

    Returns:
        Dict containing experiment metrics and summary results.
    """
    print(f"Loading dataset: {csv_path}...")
    df = pd.read_csv(csv_path)

    features = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    label_col = 'Machine failure'

    print(f"Total samples: {len(df)}")
    print(f"Monitored features ({len(features)}): {features}")
    print(f"Chunk size: {chunk_size}")

    # Initialize RBULT Control Chart
    chart = RBULTControlChart(
        features=features,
        minmax_flag=False,
        outlier_filter=outlier_filter,
        alpha_sys=0.05,
        fwer_correction='bonferroni'
    )

    num_chunks = int(np.ceil(len(df) / chunk_size))
    chunk_labels = []

    print("\n--- Starting Streaming Evaluation ---")
    start_total_time = time.perf_counter()

    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        true_ooc = 1 if (label_col in chunk_df.columns and chunk_df[label_col].sum() > 0) else 0
        chunk_labels.append(true_ooc)

        # Process chunk
        summary = chart.update_chunk(chunk_df)

        if (i + 1) % 20 == 0 or (i + 1) == num_chunks:
            print(f"Chunk [{i+1}/{num_chunks}] | Latency: {summary['latency_ms']:.2f} ms | "
                  f"RAM: {summary['memory_kb']:.2f} KB | OOC: {summary['any_ooc']} "
                  f"({summary['ooc_features']})")

    total_time = time.perf_counter() - start_total_time

    # Compute overall SPC metrics
    spc_metrics = chart.compute_spc_metrics(true_labels=chunk_labels)
    spc_metrics['total_experiment_time_sec'] = total_time

    print("\n========================================================")
    print("           RBULT-SPC BENCHMARK SUMMARY RESULTS           ")
    print("========================================================")
    print(f"Total Chunks Processed:  {spc_metrics['total_chunks']}")
    print(f"Total Samples Processed: {spc_metrics['total_samples']}")
    print(f"Avg Latency per Chunk:   {spc_metrics['avg_latency_ms']:.4f} ms")
    print(f"Peak Memory Footprint:   {spc_metrics['peak_memory_kb']:.2f} KB (O(D) constant)")
    print(f"Total OOC Alarm Chunks:  {spc_metrics['total_ooc_chunks']} / {spc_metrics['total_chunks']}")
    print(f"OOC Alarm Rate:          {spc_metrics['ooc_chunk_rate'] * 100:.2f}%")
    print(f"False Alarm Rate (FAR):  {spc_metrics.get('false_alarm_rate', 0.0) * 100:.2f}%")
    print(f"Total Experiment Time:   {total_time:.2f} seconds")
    print("========================================================")

    # Save summary results
    os.makedirs('results', exist_ok=True)
    res_df = pd.DataFrame([spc_metrics])
    res_df.to_csv('results/spc_ai4i_benchmark_results.csv', index=False)
    print("Saved benchmark summary to: results/spc_ai4i_benchmark_results.csv")

    return spc_metrics


if __name__ == '__main__':
    run_spc_benchmark()
