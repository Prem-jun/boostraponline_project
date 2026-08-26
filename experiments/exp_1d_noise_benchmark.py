"""
1D Synthetic Benchmark Script with Noise Contamination & Full Sensitivity Suite
=================================================================================
Executes Section 10 of research_plan.md under two evaluation protocols:
  1. In-Sample Adaptation Mode (Current / Updated Boundary Evaluation)
  2. One-Step-Ahead Pre-Sequential Mode (Predictive Evaluation prior to chunk update)

Evaluates RBULT Online Bootstrap on 1D Non-Gaussian Data Streams under 7 Scenarios:
  - 5 Distributions: F-dist, Uniform, Wald, Gamma, Gaussian Normal
  - 7 Noise & Outlier Scenarios (Gold Standard Suite):
      * Scenario A (Clean Stream)
      * Scenario B1 (GAWN Noise 0.1*sigma - Light Continuous Noise)
      * Scenario B2 (GAWN Noise 0.2*sigma - Moderate Continuous Noise)
      * Scenario B3 (GAWN Noise 0.3*sigma - Heavy Continuous Noise)
      * Scenario C1 (Impulse Spikes 1% - Light Outliers)
      * Scenario C2 (Impulse Spikes 5% - Severe Outliers)
      * Scenario C3 (Impulse Spikes 10% - Extreme Outliers)
  - Chunk Sizes: 50, 100, 500
  - Target Alpha: 0.05 (95% Target Coverage), 0.01 (99% Target Coverage)

Evaluates 6 Quantitative Metrics:
  1. Empirical Coverage Rate (%)
  2. Left Tail Violation Rate (%)
  3. Right Tail Violation Rate (%)
  4. Mean Interval Width (Mean Range / Efficiency)
  5. Boundary Stability (Std of Left L and Right R bounds)
  6. Noise Sensitivity Ratio (NSR = Range_noise / Range_clean)

Outputs saved to dedicated folder: results_1d_noise_benchmark/
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from online_bootstrap.bootstrap_online import BootstrapOnline


def generate_synthetic_1d_population(dist_name: str, n_samples: int = 10000, seed: int = 42) -> np.ndarray:
    """Generate 1D population for synthetic distributions."""
    np.random.seed(seed)
    if dist_name == 'F-Distribution':
        data = np.random.f(dfnum=5, dfden=10, size=n_samples)
    elif dist_name == 'Uniform':
        data = np.random.uniform(low=0.0, high=100.0, size=n_samples)
    elif dist_name == 'Wald':
        data = np.random.wald(mean=1.0, scale=2.0, size=n_samples)
    elif dist_name == 'Gamma':
        data = np.random.gamma(shape=2.0, scale=2.0, size=n_samples)
    elif dist_name == 'Normal':
        data = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    return data


def inject_noise_scenario(clean_data: np.ndarray, scenario: str, seed: int = 42) -> np.ndarray:
    """Inject noise into 1D stream based on Scenario A, B1-B3, C1-C3."""
    np.random.seed(seed)
    data = clean_data.copy()
    n = len(data)
    std_val = np.std(data)
    mean_val = np.mean(data)

    if scenario == 'Scenario A (Clean)':
        return data
    elif scenario.startswith('Scenario B'):
        if '0.1sigma' in scenario:
            scale_factor = 0.1
        elif '0.3sigma' in scenario:
            scale_factor = 0.3
        else:
            scale_factor = 0.2
        gawn = np.random.normal(loc=0.0, scale=scale_factor * std_val, size=n)
        return data + gawn
    elif scenario.startswith('Scenario C'):
        if '1%' in scenario:
            p_spike = 0.01
        elif '10%' in scenario:
            p_spike = 0.10
        else:
            p_spike = 0.05

        n_spikes = int(p_spike * n)
        spike_indices = np.random.choice(n, size=n_spikes, replace=False)
        for idx in spike_indices:
            direction = np.random.choice([-1.0, 1.0])
            magnitude = np.random.uniform(4.0, 6.0) * std_val
            data[idx] = mean_val + direction * magnitude
        return data
    else:
        raise ValueError(f"Unknown noise scenario: {scenario}")


def run_1d_simulation_engine(data_stream: np.ndarray,
                               chunk_size: int,
                               method: str,
                               alpha_target: float = 0.05) -> Tuple[Dict[str, Tuple[List[float], List[float]]], float]:
    """Execute 1D online bootstrap simulation and return bound histories for both In-Sample and Pre-Sequential modes."""
    n_chunks = len(data_stream) // chunk_size

    l_history_insample = []
    r_history_insample = []

    l_history_preseq = []
    r_history_preseq = []

    # Target quantiles per tail
    tail_prob = alpha_target / 2.0
    lower_quantile = tail_prob * 100.0
    upper_quantile = (1.0 - tail_prob) * 100.0

    eng = BootstrapOnline()
    eng.set_online()

    accumulated_data = []
    start_time = time.perf_counter()

    for i in range(n_chunks):
        chunk = data_stream[i * chunk_size : (i + 1) * chunk_size].tolist()
        accumulated_data.extend(chunk)

        # 1. Capture Pre-Sequential Boundaries (predictive bounds prior to chunk update)
        if i == 0:
            preseq_l = float(np.percentile(chunk, lower_quantile))
            preseq_r = float(np.percentile(chunk, upper_quantile))
        else:
            preseq_l = l_history_insample[-1]
            preseq_r = r_history_insample[-1]

        l_history_preseq.append(preseq_l)
        r_history_preseq.append(preseq_r)

        # 2. Update / Expand Boundaries for chunk i
        if method == 'Proposed RBULT':
            # Run RBULT lazy boundary expansion with Z-score outlier filtering
            eng.expand_bt_online(chunk, outlier=True)
            l_val = eng.exp_l
            r_val = eng.exp_r

        elif method == 'Cumulative Online':
            # Run online bootstrap without spike filtering
            eng.expand_bt_online(chunk, outlier=False)
            l_val = eng.exp_l
            r_val = eng.exp_r

        elif method == 'Traditional Offline':
            # Traditional percentiles over all historical samples
            l_val = float(np.percentile(accumulated_data, lower_quantile))
            r_val = float(np.percentile(accumulated_data, upper_quantile))
        else:
            raise ValueError(f"Unknown method: {method}")

        # 3. Capture In-Sample Boundaries (post-update bounds for chunk i)
        l_history_insample.append(l_val)
        r_history_insample.append(r_val)

    latency_ms = ((time.perf_counter() - start_time) / n_chunks) * 1000.0

    histories = {
        'In-Sample Adaptation': (l_history_insample, r_history_insample),
        'One-Step-Ahead Pre-Sequential': (l_history_preseq, r_history_preseq)
    }

    return histories, latency_ms


def compute_expanded_metrics(ground_truth_stream: np.ndarray,
                             eval_stream: np.ndarray,
                             l_history: List[float],
                             r_history: List[float],
                             chunk_size: int,
                             clean_mean_width: float = None) -> dict:
    """Compute all 6 expanded quantitative evaluation metrics."""
    n_chunks = len(l_history)
    total_eval_samples = n_chunks * chunk_size
    eval_data = eval_stream[:total_eval_samples]

    # Map chunk bounds to individual sample predictions
    l_expanded = np.repeat(l_history, chunk_size)
    r_expanded = np.repeat(r_history, chunk_size)

    # 1. Empirical Coverage Rate (%)
    covered_mask = (eval_data >= l_expanded) & (eval_data <= r_expanded)
    coverage_pct = (np.sum(covered_mask) / total_eval_samples) * 100.0

    # 2. Left Tail Violation Rate (%)
    left_violations = np.sum(eval_data < l_expanded)
    left_far_pct = (left_violations / total_eval_samples) * 100.0

    # 3. Right Tail Violation Rate (%)
    right_violations = np.sum(eval_data > r_expanded)
    right_far_pct = (right_violations / total_eval_samples) * 100.0

    # 4. Mean Interval Width (Mean Range / Efficiency)
    ranges = np.array(r_history) - np.array(l_history)
    mean_width = float(np.mean(ranges))

    # 5. Boundary Stability (Std of L and R)
    sigma_l = float(np.std(l_history))
    sigma_r = float(np.std(r_history))

    # 6. Noise Sensitivity Ratio (NSR)
    if clean_mean_width is not None and clean_mean_width > 0:
        nsr = mean_width / clean_mean_width
    else:
        nsr = 1.0

    return {
        'coverage_pct': coverage_pct,
        'left_far_pct': left_far_pct,
        'right_far_pct': right_far_pct,
        'mean_width': mean_width,
        'sigma_l': sigma_l,
        'sigma_r': sigma_r,
        'nsr': nsr
    }


def execute_full_1d_noise_benchmark():
    """Run full 1D synthetic experiment suite across 7 Scenarios Gold Standard Suite."""
    output_dir = Path("results_1d_noise_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================================")
    print("  Starting Gold Standard 1D Benchmark (7 Noise Scenarios)")
    print(f"  Output Directory: {output_dir.resolve()}")
    print("=======================================================================\n")

    distributions = ['F-Distribution', 'Uniform', 'Wald', 'Gamma', 'Normal']
    scenarios = [
        'Scenario A (Clean)',
        'Scenario B1 (GAWN Noise 0.1sigma)',
        'Scenario B2 (GAWN Noise 0.2sigma)',
        'Scenario B3 (GAWN Noise 0.3sigma)',
        'Scenario C1 (Impulse Spikes 1%)',
        'Scenario C2 (Impulse Spikes 5%)',
        'Scenario C3 (Impulse Spikes 10%)'
    ]
    methods = ['Traditional Offline', 'Cumulative Online', 'Proposed RBULT']
    eval_modes = ['In-Sample Adaptation', 'One-Step-Ahead Pre-Sequential']
    chunk_sizes = [50, 100, 500]
    alpha_levels = [0.05, 0.01]

    results_records = []

    total_runs = len(distributions) * len(scenarios) * len(chunk_sizes) * len(alpha_levels) * len(methods) * len(eval_modes)
    current_run = 0

    # Cache clean mean widths per (dist, chunk_size, alpha, method, eval_mode) to calculate NSR
    clean_widths_cache = {}

    # First pass: Clean scenario to compute baseline clean widths for both evaluation modes
    for dist in distributions:
        clean_pop = generate_synthetic_1d_population(dist)
        for chunk_size in chunk_sizes:
            for alpha in alpha_levels:
                for method in methods:
                    histories, _ = run_1d_simulation_engine(
                        clean_pop, chunk_size, method, alpha_target=alpha
                    )
                    for emode in eval_modes:
                        l_h, r_h = histories[emode]
                        metrics = compute_expanded_metrics(clean_pop, clean_pop, l_h, r_h, chunk_size)
                        clean_widths_cache[(dist, chunk_size, alpha, method, emode)] = metrics['mean_width']

    # Main Benchmark Execution
    for dist in distributions:
        clean_pop = generate_synthetic_1d_population(dist)

        for scenario in scenarios:
            noisy_pop = inject_noise_scenario(clean_pop, scenario)

            for chunk_size in chunk_sizes:
                for alpha in alpha_levels:
                    target_coverage = (1.0 - alpha) * 100.0
                    target_tail_far = (alpha / 2.0) * 100.0

                    for method in methods:
                        histories, latency_ms = run_1d_simulation_engine(
                            noisy_pop, chunk_size, method, alpha_target=alpha
                        )

                        for emode in eval_modes:
                            current_run += 1
                            print(f"[{current_run}/{total_runs}] Dist: {dist:<14} | Scenario: {scenario:<34} | Mode: {emode:<30} | Chunk: {chunk_size:<3} | Alpha: {alpha} | Method: {method}")

                            clean_w = clean_widths_cache.get((dist, chunk_size, alpha, method, emode), 1.0)
                            l_h, r_h = histories[emode]

                            metrics = compute_expanded_metrics(
                                clean_pop, noisy_pop, l_h, r_h, chunk_size, clean_mean_width=clean_w
                            )

                            record = {
                                'Distribution': dist,
                                'Noise_Scenario': scenario,
                                'Eval_Mode': emode,
                                'Chunk_Size': chunk_size,
                                'Target_Alpha': alpha,
                                'Target_Coverage_Pct': target_coverage,
                                'Target_Tail_FAR_Pct': target_tail_far,
                                'Method': method,
                                'Empirical_Coverage_Pct': round(metrics['coverage_pct'], 2),
                                'Left_FAR_Pct': round(metrics['left_far_pct'], 2),
                                'Right_FAR_Pct': round(metrics['right_far_pct'], 2),
                                'Mean_Interval_Width': round(metrics['mean_width'], 4),
                                'Sigma_L_Stability': round(metrics['sigma_l'], 4),
                                'Sigma_R_Stability': round(metrics['sigma_r'], 4),
                                'Noise_Sensitivity_Ratio_NSR': round(metrics['nsr'], 4),
                                'Latency_per_Chunk_ms': round(latency_ms, 4)
                            }
                            results_records.append(record)

    # Convert to DataFrame
    df_results = pd.DataFrame(results_records)

    # Save CSV
    csv_file = output_dir / "exp_1d_noise_benchmark_results.csv"
    df_results.to_csv(csv_file, index=False)
    print(f"\nSaved raw CSV results to: {csv_file}")
    
    # Generate reports
    generate_reports_from_csv(csv_file, output_dir)


def df_to_markdown_string(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join([str(val) for val in row]) + " |"
        rows.append(row_str)
    return "\n".join([header_line, separator_line] + rows)


def generate_reports_from_csv(csv_path: Path, output_dir: Path):
    """Generate Markdown and HTML reports from existing CSV results."""
    df_results = pd.read_csv(csv_path)

    md_file = output_dir / "exp_1d_noise_summary_table.md"
    
    # Filter for chunk_size=100 and target_alpha=0.05
    subset_df = df_results[(df_results['Chunk_Size'] == 100) & (df_results['Target_Alpha'] == 0.05)]

    # Aggregated Summaries per Evaluation Mode
    agg_df = df_results.groupby(['Eval_Mode', 'Noise_Scenario', 'Method']).agg({
        'Empirical_Coverage_Pct': 'mean',
        'Left_FAR_Pct': 'mean',
        'Right_FAR_Pct': 'mean',
        'Mean_Interval_Width': 'mean',
        'Sigma_L_Stability': 'mean',
        'Noise_Sensitivity_Ratio_NSR': 'mean',
        'Latency_per_Chunk_ms': 'mean'
    }).round(2).reset_index()

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 1D Synthetic Benchmark Results: 7-Scenario Gold Standard Suite\n\n")
        f.write("This report presents the 1D Synthetic Benchmark results evaluated under **both protocols** across 7 Noise & Outlier Scenarios:\n")
        f.write("1. **Scenario A (Clean Stream)**: Baseline pure stream without noise.\n")
        f.write("2. **Scenario B1 (GAWN Noise 0.1\\sigma)**: Light continuous Gaussian noise.\n")
        f.write("3. **Scenario B2 (GAWN Noise 0.2\\sigma)**: Moderate continuous Gaussian noise.\n")
        f.write("4. **Scenario B3 (GAWN Noise 0.3\\sigma)**: Heavy continuous Gaussian noise.\n")
        f.write("5. **Scenario C1 (Impulse Spikes 1%)**: Light outlier spike contamination.\n")
        f.write("6. **Scenario C2 (Impulse Spikes 5%)**: Severe outlier spike contamination.\n")
        f.write("7. **Scenario C3 (Impulse Spikes 10%)**: Extreme stress test outlier contamination.\n\n")
        f.write("Subset: Chunk Size = 100, Target Alpha = 0.05 (Target Coverage = 95.00%, Target Tail FAR = 2.50%)\n\n")

        for emode in ['In-Sample Adaptation', 'One-Step-Ahead Pre-Sequential']:
            f.write(f"## Protocol: {emode}\n\n")
            emode_df = subset_df[subset_df['Eval_Mode'] == emode]
            f.write(df_to_markdown_string(emode_df))
            f.write("\n\n---\n\n")

        f.write("## Overall Aggregated Comparative Summary Across All Distributions & Protocols\n\n")
        f.write(df_to_markdown_string(agg_df))

    print(f"Saved Markdown summary table to: {md_file}")

    print("\n=======================================================================")
    print("      Benchmark Completed Successfully! 7-Scenario Aggregated Summary:")
    print("=======================================================================\n")
    print(agg_df.to_string(index=False))
    print(f"\nResults persisted in dedicated directory: {output_dir.resolve()}")


if __name__ == '__main__':
    execute_full_1d_noise_benchmark()
