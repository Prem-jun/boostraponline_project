import os
import yaml

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
    for result_file in config_files:
        result_path = os.path.join(results_dir, result_file)
        if not os.path.exists(result_path):
            print(f"Warning: {result_path} not found.")
            continue
        results = load_results(result_path)
        all_results.extend(results)
        print(f"Loaded {len(results)} results from {result_file}")
    # Example: print summary statistics for bt_est_on['history'] length
    for idx, entry in enumerate(all_results):
        bt_est_on = entry.get('bt_est_on', {})
        history = bt_est_on.get('history', [])
        print(f"Result {idx+1}: config_file={entry.get('config_file')}, history_len={len(history)}")
    # You can add more evaluation/analysis code here

if __name__ == "__main__":
    # Example usage: adjust these paths as needed
    config_list_path = "config_sim_data/fdist/results_config_files.yaml"
    results_dir = "config_sim_data/fdist"
    evaluate_results(config_list_path, results_dir)