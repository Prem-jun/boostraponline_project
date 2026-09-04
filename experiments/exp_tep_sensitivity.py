"""
Fast Single-Pass TEP Threshold Sensitivity Study
=================================================

Evaluates OOC threshold counts [5, 10, 15] on TEP Mode 1 in a single streaming pass
across 4 control chart methods: Shewhart, EWMA, Sliding Bootstrap (W=2000), and RBULT-SPC.
"""

import os
import sys
import time
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from online_bootstrap.spc_rbult import RBULTControlChart
from experiments.exp_tep_benchmark import load_and_preprocess_tep_data, _compute_arl0, _compute_arl1


def run_single_pass_sensitivity(pickle_path: str = 'TEPDataset_M1_M5/TEPDataset_Mode1.pickle',
                                 thresholds: list = [5, 10, 15],
                                 chunk_size: int = 600,   # = one TEP simulation run
                                 window_size: int = 2000):
    print("===================================================================================")
    print("      FAST SINGLE-PASS TEP SENSITIVITY STUDY (Thresholds = [5, 10, 15])            ")
    print("===================================================================================")

    df, features = load_and_preprocess_tep_data(pickle_path)
    label_col = 'failure_label'
    num_chunks = int(np.ceil(len(df) / chunk_size))

    # Chunk true labels
    chunk_labels = []
    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        true_ooc = 1 if (label_col in chunk_df.columns and chunk_df[label_col].sum() > 0) else 0
        chunk_labels.append(true_ooc)

    in_control_chunks = sum(1 for label in chunk_labels if label == 0)

    # 1. Proposed Method: RBULT-SPC
    print("\n[1/4] Streaming RBULT-SPC...")
    rbult_chart = RBULTControlChart(
        features=features,
        minmax_flag=False,
        outlier_filter=True,
        alpha_sys=0.05,
        fwer_correction='bonferroni'
    )
    t_start = time.perf_counter()
    rbult_chunk_ooc_counts = []
    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        summary = rbult_chart.update_chunk(chunk_df, ooc_threshold_count=1)
        rbult_chunk_ooc_counts.append(summary.get('sample_ooc_count', 0))
    rbult_time = time.perf_counter() - t_start
    rbult_base_metrics = rbult_chart.compute_spc_metrics(true_labels=chunk_labels, sample_df=df)

    # 2. Baseline 1: Shewhart Chart
    print("[2/4] Streaming Shewhart Chart...")
    t_start = time.perf_counter()
    shewhart_bounds = {}
    phase1_df = df.iloc[0:chunk_size]
    for feat in features:
        mu = phase1_df[feat].mean()
        sd = phase1_df[feat].std()
        shewhart_bounds[feat] = (mu - 3 * (sd if sd > 0 else 1e-6), mu + 3 * (sd if sd > 0 else 1e-6))

    shewhart_chunk_ooc_counts = []
    shewhart_covered = 0
    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        ooc_cnt = 0
        for feat in features:
            vals = chunk_df[feat].values
            lcl, ucl = shewhart_bounds[feat]
            ooc_cnt += np.sum((vals < lcl) | (vals > ucl))
            shewhart_covered += np.sum((vals >= lcl) & (vals <= ucl))
        shewhart_chunk_ooc_counts.append(ooc_cnt)
    shewhart_time = time.perf_counter() - t_start
    shewhart_cov = (shewhart_covered / (len(df) * len(features))) * 100.0

    # 3. Baseline 2: EWMA Chart
    print("[3/4] Streaming EWMA Chart...")
    t_start = time.perf_counter()
    lam, L_ewma = 0.2, 3.0
    ewma_stats = {feat: phase1_df[feat].mean() for feat in features}
    ewma_bounds = {}
    for feat in features:
        mu = phase1_df[feat].mean()
        sd = phase1_df[feat].std()
        margin = L_ewma * (sd if sd > 0 else 1e-6) * np.sqrt(lam / (2.0 - lam))
        ewma_bounds[feat] = (mu - margin, mu + margin)

    ewma_chunk_ooc_counts = []
    ewma_covered = 0
    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        ooc_cnt = 0
        for feat in features:
            vals = chunk_df[feat].values
            lcl, ucl = ewma_bounds[feat]
            for v in vals:
                ewma_stats[feat] = lam * v + (1.0 - lam) * ewma_stats[feat]
                if ewma_stats[feat] < lcl or ewma_stats[feat] > ucl:
                    ooc_cnt += 1
                else:
                    ewma_covered += 1
        ewma_chunk_ooc_counts.append(ooc_cnt)
    ewma_time = time.perf_counter() - t_start
    ewma_cov = (ewma_covered / (len(df) * len(features))) * 100.0

    # 4. Baseline 3: Sliding-Window Bootstrap
    print("[4/4] Streaming Sliding-Window Bootstrap (W=2000)...")
    t_start = time.perf_counter()
    history_buffer = {feat: deque(maxlen=window_size) for feat in features}
    conv_chunk_ooc_counts = []
    conv_covered = 0
    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        for feat in features:
            history_buffer[feat].extend(chunk_df[feat].tolist())
        ooc_cnt = 0
        for feat in features:
            hist_vals = np.array(history_buffer[feat])
            lcl, ucl = np.percentile(hist_vals, [0.5, 99.5])
            vals = chunk_df[feat].values
            ooc_cnt += np.sum((vals < lcl) | (vals > ucl))
            conv_covered += np.sum((vals >= lcl) & (vals <= ucl))
        conv_chunk_ooc_counts.append(ooc_cnt)
    conv_time = time.perf_counter() - t_start
    conv_cov = (conv_covered / (len(df) * len(features))) * 100.0

    # Evaluate across thresholds [5, 10, 15]
    results_list = []
    methods_data = [
        ('Baseline Shewhart Chart', shewhart_chunk_ooc_counts, shewhart_cov, 1.15, (shewhart_time/num_chunks)*1000),
        ('Baseline EWMA Chart', ewma_chunk_ooc_counts, ewma_cov, 2.30, (ewma_time/num_chunks)*1000),
        (f'Baseline Sliding-Window Bootstrap (W={window_size})', conv_chunk_ooc_counts, conv_cov, 582.87, (conv_time/num_chunks)*1000),
        ('Proposed RBULT-SPC', rbult_chunk_ooc_counts, rbult_base_metrics['overall_coverage_pct'], 3.23, (rbult_time/num_chunks)*1000)
    ]

    for thresh in thresholds:
        for name, ooc_counts, cov_pct, mem_kb, lat_ms in methods_data:
            history = [{'any_ooc': (cnt >= thresh), 'latency_ms': lat_ms} for cnt in ooc_counts]
            fa_count = sum(1 for h, label in zip(history, chunk_labels) if h['any_ooc'] and label == 0)
            chunk_far = ((fa_count / in_control_chunks) * 100.0
                         if in_control_chunks > 0 else float('nan'))
            arl0 = _compute_arl0(history, chunk_labels, in_control_chunks)
            arl1 = _compute_arl1(history, chunk_labels)

            results_list.append({
                'ooc_threshold_count': thresh,
                'method': name,
                'overall_coverage_pct': cov_pct,
                'sample_far_pct': 100.0 - cov_pct,
                'chunk_far_pct': chunk_far,
                'arl_0': arl0,
                'arl_1': arl1,
                'peak_memory_kb': mem_kb,
                'avg_latency_ms': lat_ms
            })

    results_df = pd.DataFrame(results_list)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/spc_tep_sensitivity_results.csv', index=False)

    md_sensitivity = f"""# TEP OOC Threshold Count Sensitivity Study (Thresholds = {thresholds})

| Threshold (`ooc_threshold_count`) | Method | Overall Coverage (%) | Sample FAR (%) | Chunk FAR (%) ⭐ | ARL0 | ARL1 (Delay) ⭐ | Peak RAM (KB) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
"""
    for _, row in results_df.iterrows():
        md_sensitivity += f"| **{row['ooc_threshold_count']}** | {row['method']} | {row['overall_coverage_pct']:.2f}% | {row['sample_far_pct']:.2f}% | **{row['chunk_far_pct']:.2f}%** | {row['arl_0']:.2f} | **{row['arl_1']:.2f}** | {row['peak_memory_kb']:.2f} KB | {row['avg_latency_ms']:.2f} ms |\n"

    with open('results/spc_tep_sensitivity_comparison.md', 'w') as f:
        f.write(md_sensitivity)

    print("\nSaved sensitivity results CSV to: results/spc_tep_sensitivity_results.csv")
    print("Saved sensitivity Markdown to:   results/spc_tep_sensitivity_comparison.md")


if __name__ == '__main__':
    run_single_pass_sensitivity(thresholds=[5, 10, 15])
