"""
Plotting Script: Tennessee Eastman Process (TEP) Telemetry Visualization
========================================================================

Generates visual control chart figures comparing key TEP telemetry signals
and ground-truth industrial disturbance/fault intervals.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.exp_tep_benchmark import load_and_preprocess_tep_data


def plot_tep_control_charts(pickle_path: str = 'TEPDataset_M1_M5/TEPDataset_Mode4.pickle',
                             mode_label: str = 'Mode 4 (Max Production Rate)',
                             output_path: str = 'results/spc_tep_mode4_chart.png'):
    """Generate control chart telemetry visualization for TEP dataset."""
    df, features = load_and_preprocess_tep_data(pickle_path)

    # Key representative sensor channels to plot (e.g. sensor_00, sensor_06, sensor_08)
    target_features = features[:3]
    
    fig, axes = plt.subplots(len(target_features), 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Tennessee Eastman Process ({mode_label}) Telemetry & Ground Truth',
                 fontsize=14, fontweight='bold')

    sample_x = df.index
    failure_mask = df['failure_label'] == 1

    for ax, feat in zip(axes, target_features):
        ax.plot(sample_x, df[feat], color='#2980b9', alpha=0.6, label=f'TEP Telemetry ({feat})', linewidth=0.5)
        
        # Highlight disturbance/fault states in light red
        ax.fill_between(sample_x, df[feat].min(), df[feat].max(), where=failure_mask,
                        color='#e74c3c', alpha=0.15, label='Disturbance / Fault Active')

        ax.set_ylabel(feat, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Sample Index (Sequential Stream Order)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved TEP control chart plot to: {output_path}")
    plt.close()


if __name__ == '__main__':
    plot_tep_control_charts(
        pickle_path='TEPDataset_M1_M5/TEPDataset_Mode5.pickle',
        mode_label='Mode 5 (10/90 Mass Ratio, Max Rate)',
        output_path='results/spc_tep_mode5_chart.png'
    )

