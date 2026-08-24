"""
Plotting Script: MetroPT-3 SPC Control Chart Visualization
============================================================

Generates visual control chart figures comparing telemetry sensor signals,
dynamic RBULT LCL/UCL bounds, and ground-truth failure windows.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.exp_metropt3_benchmark import load_and_label_metropt3


def plot_metropt3_control_charts(csv_path: str = 'MetroPT3/MetroPT3_AirCompressor.csv',
                                 output_path: str = 'results/spc_metropt3_chart.png'):
    """Generate control chart visualization for MetroPT-3 key features."""
    df = load_and_label_metropt3(csv_path)

    # Key features to visualize
    target_features = ['TP3', 'Motor_current', 'Oil_temperature']
    
    fig, axes = plt.subplots(len(target_features), 1, figsize=(14, 10), sharex=True)
    fig.suptitle('MetroPT-3 Air Compressor Telemetry & SPC Anomaly Ground Truth', fontsize=14, fontweight='bold')

    time_x = df['datetime']

    # Highlight ground-truth failure regions
    failure_mask = df['failure_label'] == 1

    for ax, feat in zip(axes, target_features):
        ax.plot(time_x, df[feat], color='#2b5c8f', alpha=0.7, label=f'Sensor Stream ({feat})', linewidth=0.6)
        
        # Shade ground truth failure windows in red
        ax.fill_between(time_x, df[feat].min(), df[feat].max(), where=failure_mask,
                        color='#e74c3c', alpha=0.3, label='Ground Truth Failure Window')

        ax.set_ylabel(feat, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Timestamp (Feb 2020 - Sep 2020)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved control chart plot to: {output_path}")
    plt.close()


if __name__ == '__main__':
    plot_metropt3_control_charts()
