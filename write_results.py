import os
import yaml
import pickle
import pandas as pd

def load_config_list(config_list_path):
    """Load the list of result YAML files from the config list YAML."""
    with open(config_list_path, 'r') as f:
        config_files = yaml.safe_load(f)
    return config_files

def load_results(result_file_path):
    """Load the results from a result YAML file (multiple documents)."""
    with open(result_file_path, 'r') as f:
        results = list(yaml.safe_load_all(f))
    return results

def evaluate_results(config_list_path, results_dir):
    config_files = load_config_list(config_list_path)
    all_results = []
    perf_rows = []
    for result_file in config_files:
        # Get the population data file name from the result file name
        pop_file = result_file.split('_')[0] + '.pkl'
        pop_data = pd.read_pickle(os.path.join(results_dir, pop_file))
        pop_min = min(pop_data)
        pop_max = max(pop_data)
        pop_range = pop_max-pop_min
        result_path = os.path.join(results_dir, result_file)
        
        if not os.path.exists(result_path):
            print(f"Warning: {result_path} not found.")
            continue
        results = load_results(result_path)
        for result in results:
            # Store all history entries for all three estimators
            for key in ['bt_est_on', 'bt_est_onmm', 'bt_est_trad','bt_est_on_out']:
                est = result.get(key, {})
                history = est.get('history', [])
                for hist_entry in history:
                    tmp_exp_l = hist_entry.get('exp_l')
                    tmp_exp_r = hist_entry.get('exp_r')
                    perf_rows.append({
                        'config_file': result.get('config_file'),
                        'chunk_size': result.get('chunk_size'),
                        'estimator': key,
                        'pop_min'   : pop_min,
                        'exp_l': tmp_exp_l,
                        'pop_max'   : pop_max,
                        'exp_r': tmp_exp_r,
                        'pop_range': pop_range,
                        'exp_range': tmp_exp_r - tmp_exp_l,
                        'nlearn_l': hist_entry.get('nlearn_l'),
                        'nlearn_r': hist_entry.get('nlearn_r'),
                        'ch_min': hist_entry.get('ch_min'),
                        'ch_max': hist_entry.get('ch_max'),
                        'poperr_min': pop_min - tmp_exp_l,
                        'poperr_max': pop_max - tmp_exp_r,
                        'poperr_range': pop_range - (tmp_exp_r - tmp_exp_l)
                    })
        all_results.extend(results)
        print(f"Loaded {len(results)} results from {result_file}")
    # Create DataFrame from performance rows
    perf_df = pd.DataFrame(perf_rows)
    print(perf_df.head())
    if 'outlier' in config_list_path:
        perf_df.to_csv(os.path.join(results_dir, "performance_summary_outlier.csv"), index=False)
    else:
        perf_df.to_csv(os.path.join(results_dir, "performance_summary.csv"), index=False)
    return perf_df

if __name__ == "__main__":
    # Example usage: adjust these paths as needed
    # config_list_path = "config_sim_data/fdist/results_config_files.yaml"
    config_list_path = "config_sim_data/fdist/results_config_files_outlier.yaml"
    results_dir = "config_sim_data/fdist"
    perf_df = evaluate_results(config_list_path, results_dir)