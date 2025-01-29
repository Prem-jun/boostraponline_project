import argparse
import yaml
import pickle, os
import pandas as pd
import numpy as np
from typing import List, Union, Dict
from dataclasses import dataclass, field
from online_bootstrap import res_bootstrap
from pathlib import Path

    

def ensure_directory_exists(directory_path):
    """
    Ensure a directory exists, create it if it doesn't
    
    Args:
        directory_path (str): Path to the directory
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")
        return False

def print_args(args):
    """Print arguments nicely formatted"""
    print("Arguments:")
    for k, v in args.items():
        print(f"  {k}: {v}")

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
        epilog='Example: python script.py --source config_sim_data/normal',  # ข้อความท้าย help
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # แสดง default values
        prog='Bootstrap-Tool'
        )
    ROOT = Path(__file__).parent
    parser.add_argument("--source", type = str, default=ROOT/"config_sim_data/config_results_chi2.yaml", help = 'source for loading config file results')
    opt = parser.parse_args()
    return opt

def run(source:str):
    with open(source, 'r') as file:
        config = yaml.safe_load(file)
    folder_path = config['folder_path']
    filename_list = config['filename_list']
    res_all = []
    pop_data = []
    for filename in filename_list:
        
        # Save the result figures.
        figure_path = os.path.join(folder_path,filename)
        
        if not(ensure_directory_exists(figure_path)):
            break
        
        file_re = os.path.join(folder_path,filename+'_re.pkl') # file results
        file_pop = os.path.join(folder_path,filename) # file name of population data.
        # load all instances.
        with open(file_re, 'rb') as file:
            loaded_data = pickle.load(file)
        temp = loaded_data['result_all']    
        res_all.append(temp)    
        pop_data.append(pd.read_pickle(file_pop+'.pkl'))    
        pop_min = np.min(pop_data[-1])
        pop_max = np.max(pop_data[-1])
        pop_range = pop_max - pop_min
        print(f" Distribution: {filename}: min = {pop_min:.4f} and max = {pop_max:.4f}")
        # print population distribution
        res_bootstrap.plot_hist(pop_data[-1],filesave = os.path.join(figure_path,filename))
        
        res = res_all[-1]
        ch_size = list(set([res1.chunk_size for res1 in res]))
        exp_l = []
        exp_r = []
        exp_range = []
        nlearn = []
        name_l = []
        name_r = []
        error_l = []
        error_r = []
        error_range = []
        name = []
        for size in ch_size:
            name_l.append(['min_'+res1.net_name+str(size) for res1 in res if res1.chunk_size==size])
            name_r.append(['max_'+res1.net_name+str(size) for res1 in res if res1.chunk_size==size])
            exp_l.append([res1.exp_l for res1 in res if res1.chunk_size==size])
            exp_r.append([res1.exp_r for res1 in res if res1.chunk_size==size])
            exp_range.append([res1.exp_range for res1 in res if res1.chunk_size==size])
            nlearn.append([[a+b for a,b in zip(res1.nlearnl,res1.nlearnr)] for res1 in res if res1.chunk_size==size])
            name.append([res1.net_name+str(size) for res1 in res if res1.chunk_size==size])
            error_l.append([list(map(lambda x: pop_min - x, res1.exp_l)) for res1 in res if res1.chunk_size==size])
            error_r.append([list(map(lambda x: pop_max - x, res1.exp_r)) for res1 in res if res1.chunk_size==size])
            error_range.append([list(map(lambda x: pop_range - x, res1.exp_range)) for res1 in res if res1.chunk_size==size])    
        popminmax = [pop_min,pop_max]
        for idx in range(len(exp_l)):
            res_bootstrap.plot_minmax_line(ch_size[idx],exp_l[idx], exp_r[idx], name_l[idx],name_r[idx], popminmax,
                            filesave = os.path.join(figure_path,filename+name[idx][0]))
            res_bootstrap.plot_err_line(ch_size[idx],error_range[idx], name[idx],
                        filesave = os.path.join(figure_path,filename+name[idx][0]+'error'),position = 'top-right')
            res_bootstrap.plot_nlearn_line(ch_size[idx],nlearn[idx], name[idx], 
                            filesave = os.path.join(figure_path,filename+name[idx][0]+'_n'))
    
def main(opt):
    run(**vars(opt))
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)    
    