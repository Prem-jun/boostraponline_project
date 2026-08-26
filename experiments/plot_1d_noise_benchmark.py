"""
1D Synthetic Benchmark Visualization Script (7-Scenario Gold Standard Suite)
=============================================================================
Generates publication-quality charts for Section 10 of research_plan.md:
  1. Coverage Comparison Bar Chart across 7 Noise Scenarios (Dual Protocols)
  2. Noise Sensitivity Ratio (NSR) & Stability Chart (Dual Protocols)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def clean_sc_label(sc: str) -> str:
    """Format scenario names cleanly for tick labels."""
    if 'Clean' in sc:
        return 'Clean'
    elif 'GAWN' in sc:
        if '0.1' in sc:
            return 'GAWN 0.1\u03c3'
        elif '0.3' in sc:
            return 'GAWN 0.3\u03c3'
        else:
            return 'GAWN 0.2\u03c3'
    elif '1%' in sc:
        return 'Spikes 1%'
    elif '5%' in sc:
        return 'Spikes 5%'
    elif '10%' in sc:
        return 'Spikes 10%'
    return sc

def plot_1d_benchmark_charts(csv_path: str, output_dir: str):
    """Generate summary figures from 1D benchmark CSV results for dual protocols."""
    csv_file = Path(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        print(f"Error: Results CSV not found at {csv_file}")
        return

    df = pd.read_csv(csv_file)

    # Set matplotlib aesthetic style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 8.5,
        'axes.labelsize': 9.5,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11.5
    })

    # Filter for default chunk size = 100 and target alpha = 0.05 (95% coverage)
    subset = df[(df['Chunk_Size'] == 100) & (df['Target_Alpha'] == 0.05)]
    if subset.empty:
        subset = df

    # ------------------------------------------------------------------ #
    #  Figure 1: Dual Protocol Empirical Coverage Rate (%)               #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    eval_modes = ['In-Sample Adaptation', 'One-Step-Ahead Pre-Sequential']
    scenarios = subset['Noise_Scenario'].unique()
    methods = subset['Method'].unique()
    x = np.arange(len(scenarios))
    width = 0.25

    colors = {'Traditional Offline': '#94a3b8', 'Cumulative Online': '#f59e0b', 'Proposed RBULT': '#2563eb'}

    for idx, emode in enumerate(eval_modes):
        ax = axes[idx]
        emode_df = subset[subset['Eval_Mode'] == emode]

        for i, method in enumerate(methods):
            method_df = emode_df[emode_df['Method'] == method]
            means = [method_df[method_df['Noise_Scenario'] == sc]['Empirical_Coverage_Pct'].mean() for sc in scenarios]
            rects = ax.bar(x + (i - 1) * width, means, width, label=method, color=colors.get(method, '#64748b'))

            for rect in rects:
                height = rect.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.axhline(y=95.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Target Coverage (95.0%)')
        ax.set_ylabel('Empirical Coverage Rate (%)' if idx == 0 else '')
        ax.set_title(f'Protocol: {emode}')
        ax.set_xticks(x)
        ax.set_xticklabels([clean_sc_label(sc) for sc in scenarios], rotation=15)
        ax.set_ylim(70, 103)
        ax.legend(loc='lower right', frameon=True)

    fig.suptitle('1D Stream Empirical Coverage Rate Across 7 Scenarios: In-Sample vs. One-Step-Ahead Pre-Sequential (\u03b1 = 0.05)', y=1.02)
    plt.tight_layout()

    fig1_path = out_dir / "fig_1d_coverage_comparison.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 to: {fig1_path}")

    # ------------------------------------------------------------------ #
    #  Figure 2: Dual Protocol Noise Sensitivity Ratio (NSR)            #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    for idx, emode in enumerate(eval_modes):
        ax = axes[idx]
        emode_df = subset[subset['Eval_Mode'] == emode]
        nsr_data = emode_df.groupby(['Noise_Scenario', 'Method'])['Noise_Sensitivity_Ratio_NSR'].mean().reset_index()

        for i, method in enumerate(methods):
            method_nsr = nsr_data[nsr_data['Method'] == method]
            nsr_vals = [method_nsr[method_nsr['Noise_Scenario'] == sc]['Noise_Sensitivity_Ratio_NSR'].values[0]
                        if not method_nsr[method_nsr['Noise_Scenario'] == sc].empty else 1.0
                        for sc in scenarios]
            rects = ax.bar(x + (i - 1) * width, nsr_vals, width, label=method, color=colors.get(method, '#64748b'))

            for rect in rects:
                height = rect.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=7)

        ax.axhline(y=1.0, color='#16a34a', linestyle=':', linewidth=1.5, label='Ideal Baseline (NSR = 1.0)')
        ax.set_ylabel('Noise Sensitivity Ratio (NSR)' if idx == 0 else '')
        ax.set_title(f'Protocol: {emode}')
        ax.set_xticks(x)
        ax.set_xticklabels([clean_sc_label(sc) for sc in scenarios], rotation=15)
        ax.set_ylim(0.8, max(subset['Noise_Sensitivity_Ratio_NSR'].max() * 1.15, 4.2))
        ax.legend(loc='upper left', frameon=True)

    fig.suptitle('Noise Sensitivity Ratio (NSR) Across 7 Scenarios: In-Sample vs. One-Step-Ahead Pre-Sequential Protocol', y=1.02)
    plt.tight_layout()

    fig2_path = out_dir / "fig_1d_nsr_stability.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to: {fig2_path}")


if __name__ == '__main__':
    csv_path = "results_1d_noise_benchmark/exp_1d_noise_benchmark_results.csv"
    output_dir = "results_1d_noise_benchmark"
    plot_1d_benchmark_charts(csv_path, output_dir)
