
# get the dataclass field of the class : __dataclass__fields__.keys()
import argparse, os
import json, pickle, yaml
import pandas as pd
import numpy as np
from typing import List, Union, Dict
from dataclasses import dataclass, field
from online_bootstrap import bootstrap_online
from pathlib import Path

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load the data from the file
    return data

def read_yaml_config(file_path):
    """
    Read YAML configuration file that contains multiple documents separated by '---'
    Returns a list of configuration dictionaries
    """
    with open(file_path, 'r') as file:
        # yaml.safe_load_all returns a generator of all documents in the YAML file
        configs = list(yaml.safe_load_all(file))
    return configs

def parse_opt():
    """
    Args:
        --source (str | list[str], optional): Configure file results parth. Defaults to ROOT /'config_sim_data/config_results_normal.yaml'.
        
    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        
        ```
    """
    parser = argparse.ArgumentParser(
        description='Bootstraping running results',
        epilog='Example: python script.py --source config_sim_data/config_results_fdist.yaml',  # ข้อความท้าย help
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # แสดง default values
        prog='Bootstrap-Tool'
        )
    ROOT = Path(__file__).parent
    parser.add_argument("--dir", type = str, default=ROOT/"config_sim_data/fdist", help = 'working directory')
    parser.add_argument("--file", type = str, default="config_fdist_simulate.yaml", help = 'config file')
    # parser.add_argument("--outlier", action = 'store_true')
    # parser.add_argument("--savename", type = str, default="result-fdist-statanal2.csv", help = 'source for loading config file results')
    opt = parser.parse_args()
    return opt

def run(dir:str,file:str):
    yaml_path = os.path.join(dir,file)
    configs = read_yaml_config(yaml_path)
    config_files_used = []
    for config in configs:
        res_all = []
        json_data = read_json_file(os.path.join(dir,config['file_data_chunk']+'.json'))
        for count, data in enumerate(json_data):
            
            chunk_data = data['samp_chuck']
            bt_est_on = bootstrap_online.BootstrapOnline()\
            .pipe(lambda x: x.set_online())
            bt_est_onmm = bootstrap_online.BootstrapOnline()\
            .pipe(lambda x: x.set_online(minmax_boost = True))
            bt_est_trad = bootstrap_online.BootstrapOnline()\
            .pipe(lambda x: x.set_trad())        
            sample_whole = []
            count2 = 0    
            for samples_chunk in chunk_data:
                count2 +=1
                sample_whole = sample_whole + samples_chunk
                bt_est_on.bt_online(new_data_chunk = samples_chunk,ndata=len(samples_chunk))
                bt_est_onmm.bt_online(new_data_chunk = samples_chunk,ndata=len(samples_chunk))
                bt_est_trad.bt_trad(new_data_chunk = sample_whole,ndata=len(sample_whole))
                print(f'Complete running chunk: {count2} size: {len(samples_chunk)}')
        # estimator.pipe(lambda x: x.set_online(online_cum=True))
            # Collect dataclass fields as dicts for YAML
            res_all.append({
                'config_file': config['file_data_chunk'],
                'chunk_size': data['chunk_size'],
                'bt_est_on': {k: getattr(bt_est_on, k) for k in bt_est_on.__dataclass_fields__},
                'bt_est_onmm': {k: getattr(bt_est_onmm, k) for k in bt_est_onmm.__dataclass_fields__},
                'bt_est_trad': {k: getattr(bt_est_trad, k) for k in bt_est_trad.__dataclass_fields__}
            })
        # Write all results for this config to a YAML file
        yaml_outfile = os.path.join(dir, config['file_data_chunk']+"_results.yaml")
        with open(yaml_outfile, 'w') as f:
            yaml.dump_all(res_all, f, sort_keys=False)
        # Collect config file name
        config_files_used.append(config['file_data_chunk']+"_results.yaml")
    # Write the list of config result files to a separate YAML file
    config_list_file = os.path.join(dir, "results_config_files.yaml")
    with open(config_list_file, 'w') as f:
        yaml.dump(config_files_used, f)
        
def main(opt):
    run(**vars(opt))
    # run(**vars(opt))
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)