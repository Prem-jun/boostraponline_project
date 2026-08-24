"""
Plotting Script: Industrial Pump SPC Control Chart Visualization
=================================================================

Generates visual control chart figures comparing pump telemetry signals,
dynamic RBULT LCL/UCL bounds, and ground-truth maintenance flags.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.exp_pump_benchmark import load_and_preprocess_pump_data


def plot_pump_control_charts(csv_path: str = 'Large_Industrial_Pump_Maintenance_Dataset.csv',
                             output_path: str = 'results/spc_pump_chart.png'):
    """Generate control chart visualization for Industrial Pump dataset."""
    df = load_and_preprocess_pump_data(csv_path)

    # Key features to visualize
    target_features = ['Vibration', 'Pressure', 'Temperature']
    
    fig, axes = plt.subplots(len(target_features), 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Large Industrial Pump Telemetry & Maintenance Ground Truth', fontsize=14, fontweight='bold')

    sample_x = df.index

    # Highlight ground-truth failure regions
    failure_mask = df['Maintenance_Flag'] == 1

    for ax, feat in zip(axes, target_features):
        ax.plot(sample_x, df[feat], color='#16a085', alpha=0.7, label=f'Pump Telemetry ({feat})', linewidth=0.6)
        
        # Shade ground truth maintenance events in light red
        ax.fill_between(sample_x, df[feat].min(), df[feat].max(), where=failure_mask,
                        color='#e74c3c', alpha=0.15, label='Maintenance Event')

        ax.set_ylabel(feat, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Sample Index (Sequential Stream Order)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved pump control chart plot to: {output_path}")
    plt.close()


if __name__ == '__main__':
    plot_pump_control_charts()
