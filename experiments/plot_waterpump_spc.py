"""
Plotting Script: Water Pump Sensor SPC Control Chart Visualization
===================================================================

Generates visual control chart figures comparing water pump telemetry signals,
dynamic RBULT LCL/UCL bounds, and ground-truth machine status events.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.exp_waterpump_benchmark import load_and_preprocess_waterpump_data


def plot_waterpump_control_charts(csv_path: str = 'sensor.csv',
                                  output_path: str = 'results/spc_waterpump_chart.png'):
    """Generate control chart visualization for Water Pump Sensor dataset."""
    df, features = load_and_preprocess_waterpump_data(csv_path)

    # Key features to visualize
    target_features = ['sensor_00', 'sensor_02', 'sensor_04', 'sensor_10']
    
    fig, axes = plt.subplots(len(target_features), 1, figsize=(14, 11), sharex=True)
    fig.suptitle('Water Pump Telemetry & Machine Failure/Recovery Ground Truth', fontsize=14, fontweight='bold')

    sample_x = df.index

    # Highlight ground-truth failure/recovery regions in light red
    failure_mask = df['failure_label'] == 1

    for ax, feat in zip(axes, target_features):
        ax.plot(sample_x, df[feat], color='#34495e', alpha=0.7, label=f'Sensor Stream ({feat})', linewidth=0.6)
        
        # Shade failure/recovery events
        ax.fill_between(sample_x, df[feat].min(), df[feat].max(), where=failure_mask,
                        color='#e74c3c', alpha=0.25, label='Broken / Recovering Event')

        ax.set_ylabel(feat, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Sample Index (April 2018 - August 2018)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved water pump control chart plot to: {output_path}")
    plt.close()


if __name__ == '__main__':
    plot_waterpump_control_charts()
