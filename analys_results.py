import pandas as pd
import numpy as np
from scipy import stats
import warnings
from typing import Tuple, Dict, List
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

def load_and_prepare_data(file_path: str,chunk_size: List =[50,500]) -> pd.DataFrame:
    """Load the performance data from CSV file."""
    df = pd.read_csv(file_path)
    filtered_df = df[df['chunk_size'].isin(chunk_size)]
    print(f"Data loaded: {len(filtered_df)} rows, {len(filtered_df.columns)} columns")
    print(f"Unique config_files: {filtered_df['config_file'].unique()}")
    print(f"Unique chunk_sizes: {filtered_df['chunk_size'].unique()}")
    print(f"Unique estimators: {filtered_df['estimator'].unique()}")
    return filtered_df

def plot_exp_l_values(df):
    """
    Create comprehensive plots of exp_l values for all estimators.
    
    Args:
        df: DataFrame containing the performance data
    """
    print("\nCreating plots of exp_l values...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analysis of exp_l Values Across Estimators', fontsize=16, fontweight='bold')
    
    # Plot 1: Box plot of exp_l by estimator (overall)
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='estimator', y='exp_l', ax=ax1)
    ax1.set_title('Distribution of exp_l by Estimator (Overall)')
    ax1.set_xlabel('Estimator')
    ax1.set_ylabel('exp_l')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Box plot by estimator and config_file
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='estimator', y='exp_l', hue='config_file', ax=ax2)
    ax2.set_title('Distribution of exp_l by Estimator and Config File')
    ax2.set_xlabel('Estimator')
    ax2.set_ylabel('exp_l')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Config File', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Box plot by estimator and chunk_size
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='estimator', y='exp_l', hue='chunk_size', ax=ax3)
    ax3.set_title('Distribution of exp_l by Estimator and Chunk Size')
    ax3.set_xlabel('Estimator')
    ax3.set_ylabel('exp_l')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Chunk Size')
    
    # Plot 4: Violin plot for detailed distribution
    ax4 = axes[1, 1]
    sns.violinplot(data=df, x='estimator', y='exp_l', ax=ax4)
    ax4.set_title('Detailed Distribution of exp_l by Estimator (Violin Plot)')
    ax4.set_xlabel('Estimator')
    ax4.set_ylabel('exp_l')
    ax4.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('exp_l_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'exp_l_analysis_plots.png'")
    
    # Show the plot
    plt.show()
    
    # Create additional detailed plots by config and chunk combinations
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('exp_l Values by Config File and Chunk Size Combinations', fontsize=16, fontweight='bold')
    
    # Get unique combinations
    config_chunk_combinations = df.groupby(['config_file', 'chunk_size']).size().index
    
    for i, (config_file, chunk_size) in enumerate(config_chunk_combinations):
        if i >= 6:  # Limit to 6 subplots
            break
            
        row = i // 3
        col = i % 3
        ax = axes2[row, col]
        
        # Filter data for this combination
        subset = df[(df['config_file'] == config_file) & (df['chunk_size'] == chunk_size)]
        
        # Create box plot
        sns.boxplot(data=subset, x='estimator', y='exp_l', ax=ax)
        ax.set_title(f'{config_file}\nChunk Size: {chunk_size}')
        ax.set_xlabel('Estimator')
        ax.set_ylabel('exp_l')
        ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplots if any
    for i in range(len(config_chunk_combinations), 6):
        row = i // 3
        col = i % 3
        fig2.delaxes(axes2[row, col])
    
    plt.tight_layout()
    plt.savefig('exp_l_by_config_chunk.png', dpi=300, bbox_inches='tight')
    print("Detailed plots saved as 'exp_l_by_config_chunk.png'")
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics for exp_l by Estimator:")
    print("=" * 50)
    summary_stats = df.groupby('estimator')['exp_l'].agg(['count', 'mean', 'std', 'min', 'max']).round(6)
    print(summary_stats)
    
    return summary_stats

def check_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Check normality using Shapiro-Wilk test.
    Returns (is_normal, p_value)
    """
    if len(data) < 3:
        return False, np.nan
    
    # Use Shapiro-Wilk test for normality
    statistic, p_value = stats.shapiro(data)
    is_normal = p_value > alpha
    return is_normal, p_value

def paired_statistical_test(group1: np.ndarray, group2: np.ndarray, 
                          alpha: float = 0.05) -> Dict:
    """
    Perform paired statistical test (t-test or Wilcoxon) based on normality.
    
    Args:
        group1: First group (bt_est_on)
        group2: Second group (other estimator)
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    # Calculate differences
    differences = group1 - group2
    
    # Check normality of differences
    is_normal, normality_p = check_normality(differences, alpha)
    
    results = {
        'n_pairs': len(differences),
        'mean_diff': np.mean(differences),
        'std_diff': np.std(differences, ddof=1),
        'normality_test_p': normality_p,
        'differences_normal': is_normal
    }
    
    if is_normal and len(differences) >= 3:
        # Use paired t-test
        statistic, p_value = stats.ttest_rel(group1, group2,alternative ='less')
        test_name = "Paired t-test"
        
        # Calculate confidence interval for mean difference
        se_diff = stats.sem(differences)
        t_critical = stats.t.ppf(1 - alpha/2, len(differences) - 1)
        ci_lower = np.mean(differences) - t_critical * se_diff
        ci_upper = np.mean(differences) + t_critical * se_diff
        
    else:
        # Use Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(group1, group2, alternative ='less')
        test_name = "Wilcoxon signed-rank test"
        ci_lower, ci_upper = np.nan, np.nan
    
    results.update({
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
    
    return results

def conduct_analysis(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Conduct the complete statistical analysis as specified.
    
    Steps:
    1. Group by config_file and chunk_size
    2. For each group, group by estimator
    3. Compare bt_est_on with other estimators using paired tests
    """
    
    results_list = []
    
    # Step 1: Group by config_file and chunk_size
    config_chunk_groups = df.groupby(['config_file', 'chunk_size'])
    
    print(f"\nFound {len(config_chunk_groups)} config_file-chunk_size combinations:")
    for name, group in config_chunk_groups:
        print(f"  {name}: {len(group)} rows")
    
    # Step 2 & 3: For each config-chunk group, compare estimators
    for (config_file, chunk_size), config_chunk_group in config_chunk_groups:
        print(f"\n--- Analyzing {config_file}, chunk_size={chunk_size} ---")
        
        # Group by estimator within this config-chunk combination
        estimator_groups = config_chunk_group.groupby('estimator')
        
        # Get bt_est_on group
        if 'bt_est_on' not in estimator_groups.groups:
            print(f"Warning: bt_est_on not found in {config_file}, chunk_size={chunk_size}")
            continue
            
        bt_est_on_data = estimator_groups.get_group('bt_est_on')['poperr_range'].values
        # bt_est_on_data = np.sort(bt_est_on_data)  # Sort for consistent pairing
        
        print(f"bt_est_on group: {len(bt_est_on_data)} observations")
        
        # Compare with other estimators
        other_estimators = [est for est in estimator_groups.groups.keys() if est != 'bt_est_on']
        
        for other_estimator in other_estimators:
            other_data = estimator_groups.get_group(other_estimator)['poperr_range'].values
            # other_data = np.sort(other_data)  # Sort for consistent pairing
            
            print(f"\nComparing bt_est_on vs {other_estimator}")
            print(f"  bt_est_on: n={len(bt_est_on_data)}, mean={np.mean(bt_est_on_data):.6f}")
            print(f"  {other_estimator}: n={len(other_data)}, mean={np.mean(other_data):.6f}")
            
            # Check if we have equal sample sizes for pairing
            if len(bt_est_on_data) != len(other_data):
                print(f"  Warning: Unequal sample sizes ({len(bt_est_on_data)} vs {len(other_data)})")
                min_len = min(len(bt_est_on_data), len(other_data))
                bt_est_on_paired = bt_est_on_data[:min_len]
                other_paired = other_data[:min_len]
            else:
                bt_est_on_paired = bt_est_on_data
                other_paired = other_data
            
            # Perform paired statistical test
            if len(bt_est_on_paired) >= 3:  # Minimum sample size for meaningful test
                test_results = paired_statistical_test(bt_est_on_paired, other_paired, alpha)
                
                print(f"  Test: {test_results['test_name']}")
                print(f"  Mean difference: {test_results['mean_diff']:.6f}")
                print(f"  p-value: {test_results['p_value']:.6f}")
                print(f"  Significant (α={alpha}): {test_results['significant']}")
                
                # Store results
                result_row = {
                    'config_file': config_file,
                    'chunk_size': chunk_size,
                    'estimator_1': 'bt_est_on',
                    'estimator_2': other_estimator,
                    'n_pairs': test_results['n_pairs'],
                    'mean_bt_est_on': np.mean(bt_est_on_paired),
                    'mean_other': np.mean(other_paired),
                    'mean_difference': test_results['mean_diff'],
                    'std_difference': test_results['std_diff'],
                    'test_name': test_results['test_name'],
                    'test_statistic': test_results['statistic'],
                    'p_value': test_results['p_value'],
                    'significant': test_results['significant'],
                    'normality_p': test_results['normality_test_p'],
                    'differences_normal': test_results['differences_normal'],
                    'ci_lower': test_results['ci_lower'],
                    'ci_upper': test_results['ci_upper']
                }
                results_list.append(result_row)
            else:
                print(f"  Skipped: Insufficient data (n={len(bt_est_on_paired)})")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df

def print_summary(results_df: pd.DataFrame, alpha: float = 0.05):
    """Print a summary of all test results."""
    print("\n" + "="*80)
    print("SUMMARY OF STATISTICAL TESTS")
    print("="*80)
    
    print(f"\nTotal comparisons: {len(results_df)}")
    print(f"Significant results (α={alpha}): {results_df['significant'].sum()}")
    print(f"Tests using paired t-test: {(results_df['test_name'] == 'Paired t-test').sum()}")
    print(f"Tests using Wilcoxon signed-rank: {(results_df['test_name'] == 'Wilcoxon signed-rank test').sum()}")
    
    print(f"\nDetailed Results:")
    print("-" * 80)
    
    for idx, row in results_df.iterrows():
        print(f"\n{row['config_file']}, chunk_size={row['chunk_size']}")
        print(f"  {row['estimator_1']} vs {row['estimator_2']}")
        print(f"  Test: {row['test_name']}")
        print(f"  n_pairs: {row['n_pairs']}")
        print(f"  Mean difference: {row['mean_difference']:.6f}")
        print(f"  p-value: {row['p_value']:.6f}")
        print(f"  Significant: {'Yes' if row['significant'] else 'No'}")
        if pd.notna(row['ci_lower']) and pd.notna(row['ci_upper']):
            print(f"  95% CI for difference: [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]")

def plot_exp_l_exp_r_progression(df):
    """
    Create progression line plots similar to the provided example,
    showing exp_l and exp_r values for each estimator method.
    
    Args:
        df: DataFrame containing the performance data
    """
    print("\nCreating progression plots for exp_l and exp_r by method...")
    
    # Get unique combinations of config_file and chunk_size
    config_chunk_combinations = df.groupby(['config_file', 'chunk_size']).size().index.tolist()
    
    # Create plots for each config-chunk combination
    for config_file, chunk_size in config_chunk_combinations:
        subset = df[(df['config_file'] == config_file) & (df['chunk_size'] == chunk_size)].copy()
        
        # Sort by observation index for consistent progression
        subset = subset.reset_index(drop=True)
        subset['obs_index'] = subset.index
        
        # Create figure with subplots for exp_l and exp_r
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'exp_l Progression: {config_file}, Chunk Size: {chunk_size}',
                           f'exp_r Progression: {config_file}, Chunk Size: {chunk_size}'],
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        # Color palette
        colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']  # Red, Blue, Green, Orange
        estimators = sorted(df['estimator'].unique())
        
        # Plot exp_l for each estimator
        for i, estimator in enumerate(estimators):
            est_data = subset[subset['estimator'] == estimator].sort_values('obs_index')
            
            if len(est_data) > 0:
                # exp_l plot
                fig.add_trace(
                    go.Scatter(
                        x=est_data['obs_index'],
                        y=est_data['exp_l'],
                        mode='lines+markers',
                        name=f'exp_l_{estimator}',
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6, symbol='circle'),
                        legendgroup=f'group{i}',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # exp_r plot
                fig.add_trace(
                    go.Scatter(
                        x=est_data['obs_index'],
                        y=est_data['exp_r'],
                        mode='lines+markers',
                        name=f'exp_r_{estimator}',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                        marker=dict(size=6, symbol='diamond'),
                        legendgroup=f'group{i}',
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        # Add reference lines for min/max values
        exp_l_min, exp_l_max = subset['exp_l'].min(), subset['exp_l'].max()
        exp_r_min, exp_r_max = subset['exp_r'].min(), subset['exp_r'].max()
        
        # Add horizontal reference lines
        fig.add_hline(y=exp_l_min, line_dash="dot", line_color="gray", 
                     annotation_text=f"min exp_l: {exp_l_min:.4f}", row=1, col=1)
        fig.add_hline(y=exp_l_max, line_dash="dot", line_color="gray",
                     annotation_text=f"max exp_l: {exp_l_max:.4f}", row=1, col=1)
        
        fig.add_hline(y=exp_r_min, line_dash="dot", line_color="gray",
                     annotation_text=f"min exp_r: {exp_r_min:.4f}", row=2, col=1)
        fig.add_hline(y=exp_r_max, line_dash="dot", line_color="gray",
                     annotation_text=f"max exp_r: {exp_r_max:.4f}", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title=f'Progression Analysis: {config_file}, Chunk Size: {chunk_size}',
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Observation Index", row=2, col=1)
        fig.update_yaxes(title_text="exp_l Value", row=1, col=1)
        fig.update_yaxes(title_text="exp_r Value", row=2, col=1)
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')
        
        # Save plot
        filename = f"progression_{config_file}_chunk{chunk_size}.html"
        fig.write_html(filename)
        print(f"Progression plot saved as '{filename}'")
        fig.show()
    
    # Create overall summary progression plot
    print("\nCreating overall progression summary...")
    
    # Sort all data by exp_l for overall progression
    df_sorted = df.sort_values(['exp_l', 'exp_r']).reset_index(drop=True)
    df_sorted['global_index'] = df_sorted.index
    
    # Create summary figure
    fig_summary = make_subplots(
        rows=2, cols=1,
        subplot_titles=['exp_l Values Across All Conditions (Sorted by exp_l)',
                       'exp_r Values Across All Conditions (Sorted by exp_l)'],
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    # Colors for estimators
    estimator_colors = {
        'bt_est_on': '#d62728',      # Red
        'bt_est_onmm': '#1f77b4',    # Blue  
        'bt_est_trad': '#2ca02c',    # Green
        'bt_est_on_out': '#ff7f0e'   # Orange
    }
    
    # Plot progression for each estimator
    for estimator in sorted(df['estimator'].unique()):
        est_data = df_sorted[df_sorted['estimator'] == estimator]
        
        if len(est_data) > 0:
            # exp_l progression
            fig_summary.add_trace(
                go.Scatter(
                    x=est_data['global_index'],
                    y=est_data['exp_l'],
                    mode='lines+markers',
                    name=f'exp_l_{estimator}',
                    line=dict(color=estimator_colors.get(estimator, '#1f77b4'), width=2),
                    marker=dict(size=4, symbol='circle'),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            # exp_r progression
            fig_summary.add_trace(
                go.Scatter(
                    x=est_data['global_index'],
                    y=est_data['exp_r'],
                    mode='lines+markers',
                    name=f'exp_r_{estimator}',
                    line=dict(color=estimator_colors.get(estimator, '#1f77b4'), width=2, dash='dash'),
                    marker=dict(size=4, symbol='diamond'),
                    opacity=0.8
                ),
                row=2, col=1
            )
    
    # Add reference lines
    fig_summary.add_hline(y=df['exp_l'].min(), line_dash="dot", line_color="gray",
                         annotation_text=f"Global min exp_l: {df['exp_l'].min():.4f}", row=1, col=1)
    fig_summary.add_hline(y=df['exp_l'].max(), line_dash="dot", line_color="gray",
                         annotation_text=f"Global max exp_l: {df['exp_l'].max():.4f}", row=1, col=1)
    
    fig_summary.add_hline(y=df['exp_r'].min(), line_dash="dot", line_color="gray",
                         annotation_text=f"Global min exp_r: {df['exp_r'].min():.4f}", row=2, col=1)
    fig_summary.add_hline(y=df['exp_r'].max(), line_dash="dot", line_color="gray",
                         annotation_text=f"Global max exp_r: {df['exp_r'].max():.4f}", row=2, col=1)
    
    # Update summary layout
    fig_summary.update_layout(
        height=800,
        width=1400,
        title='Overall Progression: exp_l and exp_r Across All Methods and Conditions',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top", 
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update axes
    fig_summary.update_xaxes(title_text="Global Observation Index (Sorted by exp_l)", row=2, col=1)
    fig_summary.update_yaxes(title_text="exp_l Value", row=1, col=1)
    fig_summary.update_yaxes(title_text="exp_r Value", row=2, col=1)
    
    # Add grid
    fig_summary.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig_summary.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    # Save and show summary
    fig_summary.write_html("exp_progression_summary.html")
    print("Overall progression plot saved as 'exp_progression_summary.html'")
    fig_summary.show()
    
    # Print progression statistics
    print("\nProgression Statistics:")
    print("=" * 50)
    for estimator in sorted(df['estimator'].unique()):
        est_data = df[df['estimator'] == estimator]
        print(f"\n{estimator}:")
        print(f"  exp_l range: {est_data['exp_l'].min():.4f} - {est_data['exp_l'].max():.4f}")
        print(f"  exp_r range: {est_data['exp_r'].min():.4f} - {est_data['exp_r'].max():.4f}")
        print(f"  exp_l mean±std: {est_data['exp_l'].mean():.4f}±{est_data['exp_l'].std():.4f}")
        print(f"  exp_r mean±std: {est_data['exp_r'].mean():.4f}±{est_data['exp_r'].std():.4f}")


# Main analysis function
def main(results_dir,file_results):
    """Run the complete analysis."""
    # Load data
    df = load_and_prepare_data(f'{results_dir}/{file_results}')
    
    # Create progression plots like the example shown
    # plot_exp_l_exp_r_progression(df)
    
    # Conduct analysis
    results = conduct_analysis(df, alpha=0.05)
    
    # # Create plots of exp_l values
    # exp_l_stats = plot_exp_l_values(df)
    
    
    
    # Print summary
    print_summary(results, alpha=0.05)
    
    # Save results
    results.to_csv(f'{results_dir}/statistical_test_results.csv', index=False)
    print(f"\nResults saved to 'statistical_test_results.csv'")
    
    return results

# Example usage
if __name__ == "__main__":
    # Run the analysis
    results_dir = "config_sim_data/fdist"
    file_results = "performance_summary.csv"
    results, exp_l_summary = main(results_dir,file_results)
    
    # Display the results DataFrame
    print("\nResults DataFrame:")
    print(results.to_string())