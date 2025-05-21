""" Get the format from a full filename.
    Args:
    
    Returns:
       
    Raises:
        -   
    
    Examples:
    
""" 
import json, pickle
import os
import pandas as pd
import numpy as np
from online_bootstrap import *
# import lib_boostrap
from typing import List, Union, Dict
from dataclasses import dataclass, field
from online_bootstrap import res_bootstrap
from online_bootstrap import BatchOutlierDetection
from pathlib import Path
import argparse
import yaml


# Function to read a JSON file and return its contents as a Python object
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load the data from the file
    return data
    

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
        description='sample chunks generator',
        epilog='Example: python sim_data_samp_chunk.py --dir config_sim_data/uniform --file config_uniform_simulate.yaml',  # ข้อความท้าย help
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # แสดง default values
        prog='Sample generator-Tool'
        )
    ROOT = Path(__file__).parent
    parser.add_argument("--dir", type = str, default=ROOT/"config_sim_data/fdist", help = 'working directory')
    parser.add_argument("--file", type = str, default="config_fdist_simulate.yaml", help = 'config file')
    parser.add_argument("--outlier", action = 'store_true')
    opt = parser.parse_args()
    return opt

def read_yaml_config(file_path):
    """
    Read YAML configuration file that contains multiple documents separated by '---'
    Returns a list of configuration dictionaries
    """
    with open(file_path, 'r') as file:
        # yaml.safe_load_all returns a generator of all documents in the YAML file
        configs = list(yaml.safe_load_all(file))
    return configs

def run(dir:str,file:str,outlier:bool):
    # outlier = True
    yaml_path = os.path.join(dir,file)
    configs = read_yaml_config(yaml_path)
    # if outlier:
    #     detector = BatchOutlierDetection.ZBatchOutlierDetector()
    #     # detector.add_init_params(alpha=0.05, threshold=4.0)
    #     detector.add_init_threshold(threshold=0.3)
    for config in configs:
        res_all = []
        json_data = read_json_file(os.path.join(dir,config['file_data_chunk']+'.json'))
        for count, data in enumerate(json_data):
            # chunk_size.append(data['chunk_size'])
            chunk_data = data['samp_chuck']
            # create the initial network
            net_online = boot_stream.booststream()
            net_online_mm = boot_stream.booststream()
            net_online_cum = boot_stream.booststream()
            net_trad_cum = boot_stream.booststream()
            
            # Setting for online manner
            net_online.set_online()
            net_online_mm.set_online(minmax_flag=True)
            net_online_cum.set_online()
            
            # Setting results 
            res4 = res_bootstrap.Res_boostrap()
            res4.add_init_params(net = net_trad_cum)
            res1 = res_bootstrap.Res_boostrap()
            res1.add_init_params(net = net_online)
            res2 = res_bootstrap.Res_boostrap()
            res2.add_init_params(net = net_online_mm)
            res3 = res_bootstrap.Res_boostrap()
            res3.add_init_params(net = net_online_cum,cum=True)
            sample_whole = []
            count2 = 0 
            for samples_chunk in chunk_data:
                # if outlier:
                #     samples_chunk = detector.get_clean_data(samples_chunk)
                
                count2 +=1
                sample_whole = sample_whole + samples_chunk
                # Train online chunk
                if outlier:
                    print(f'Online running round:{count}/{count2}')
                    expandsion = net_online.expand_bt_online(samples_chunk,outlier=outlier) 
                    res1.add_params(net_online)
                    print(f'Online minmax running round:{count}/{count2}')
                    expandsion_mm = net_online_mm.expand_bt_online(samples_chunk,outlier=outlier)
                    res2.add_params(net_online_mm)
                else:   
                    print(f'Online running round:{count}/{count2}')
                    expandsion = net_online.expand_bt_online(samples_chunk)
                    res1.add_params(net_online)
                    print(f'Online minmax running round:{count}/{count2}')
                    expandsion_mm = net_online_mm.expand_bt_online(samples_chunk)
                    res2.add_params(net_online_mm)
                    # print(f'Online whole running round:{count}/{count2}')
                    # expansion_ocum = net_online_cum.expand_bt_online(sample_whole,cum=True)
                    # res3.add_params(net_online_cum)
                print(f'Offline running round:{count}/{count2}')
                expansion_tcum = net_trad_cum.expand_bt_trad(sample_whole)
                res4.add_params(net_trad_cum)
            
            res_all.append(res1)
            res_all.append(res2)
            # res_all.append(res3)
            res_all.append(res4)
            print(f'Round running:{count}')
        data_wb = {
                'result_all': res_all
            }
        # Specify the name of the pickle file
        if outlier:
            pickle_file = os.path.join(dir,config['file_data_chunk']+'_re_outlier.pkl')
        else:    
            pickle_file = os.path.join(dir,config['file_data_chunk']+'_re.pkl')
        # Save the variables to the pickle file
        with open(pickle_file, 'wb') as file:
            pickle.dump(data_wb, file)
        # print(json_data)  # Print the contents of the JSON file
        
    
def main(opt):
    run(**vars(opt))
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)    