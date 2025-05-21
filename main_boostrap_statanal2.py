import argparse
import yaml, json
import pickle, os
import pandas as pd
import numpy as np
from typing import List, Union, Dict
from dataclasses import dataclass, field
from online_bootstrap import res_bootstrap
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon


def paired_ttest(error:List):
    t_val = []
    p_val = []
    for i in range(1, len(error)):
        # t_stat, p_value = ttest_rel(np.array(error[0]), np.array(error[i]),alternative = 'less')
        t_stat, p_value = ttest_rel(error[0], error[i],alternative = 'less')
        t_val.append(t_stat)
        p_val.append(p_value)
    return list(zip(t_val,p_val))     

def paired_wilcoxon(error:List):
    t_val = []
    p_val = []
    for i in range(1, len(error)):
        t_stat, p_value = wilcoxon(error[0], error[i],alternative = 'less')
        t_val.append(t_stat)
        p_val.append(p_value)
    return list(zip(t_val,p_val))

   

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
        epilog='Example: python script.py --source config_sim_data/config_results_fdist.yaml',  # ข้อความท้าย help
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # แสดง default values
        prog='Bootstrap-Tool'
        )
    ROOT = Path(__file__).parent
    parser.add_argument("--source", type = str, default=ROOT/"config_sim_data/config_results_fdist.yaml", help = 'source for loading config file results')
    parser.add_argument("--chunk_file", type = str, default=ROOT/"config_sim_data/fdist/config_fdist_simulate.yaml", help = 'source for loading config file results')
    parser.add_argument("--outlier", action = 'store_true')
    # parser.add_argument("--savename", type = str, default="result-fdist-statanal2.csv", help = 'source for loading config file results')
    opt = parser.parse_args()
    return opt

def run(source:str,chunk_file:str,outlier:bool,select_ch=[50,500]):
    outlier = True
    # Read the network files.
    with open(source, 'r') as file:
        config = yaml.safe_load(file)
        
    with open(chunk_file, 'r') as file:
        # yaml.safe_load_all returns a generator of all documents in the YAML file
        chunk_info = list(yaml.safe_load_all(file))
    folder_path, file_name = os.path.split(chunk_file)
    
    
    for info in chunk_info:
        res = []
        # Read streaming data chunk data.
        chunkfile = os.path.join(folder_path,info['file_data_chunk']+'.json')
        with open(chunkfile, 'r') as file:
            data = json.load(file)  # Load the data from the file
        # Read population data.
        pop_data = pd.read_pickle(os.path.join(folder_path,info['file_data_chunk']+'.pkl'))
        pop_min = min(pop_data)
        pop_max = max(pop_data)
        pop_range = pop_max-pop_min
            
        if outlier:
            resultfile = os.path.join(folder_path,info['file_data_chunk']+'_re_outlier.pkl') # file results    
        else:
            resultfile = os.path.join(folder_path,info['file_data_chunk']+'_re.pkl') # file results
        
        with open(resultfile, 'rb') as file:
            result_data = pickle.load(file)
              
        for size in select_ch:
            # res.append(dict(file = info['file_data_chunk'],method  = [],chunk_size = None 
            #             ,err1stepl = [],err1stepr = [],errpopl=[]
            #             ,errpopr = [],errpop_range = [],nlearn = []))    
            # res.append(dict(file = info['file_data_chunk'],method  = [],chunk_size = None 
            #             ,test_pop_errl = None, test_pop_errr = None,test_pop_range=None
            #             ,errpopr = [],errpop_range = [],nlearn = []))    
            res.append(dict(file = info['file_data_chunk'],method  = [],chunk_size = None 
                        ,test_pop_errl = None, test_pop_errr = None,test_pop_errrange = None
            ))    
            
            # res[-1]['method'].append([])
            res[-1]['chunk_size']= size
            err1stepr = []
            err1stepl = []
            errpopr = []
            errpopl = []
            errpop_range = []
            # res[-1]['err1stepr'].append([])
            # res[-1]['err1stepl'].append([])
            # res[-1]['errpopr'].append([])
            # res[-1]['errpopl'].append([])
            # res[-1]['errpop_range'].append([])
            sample_ch = [ dat['samp_chuck'] for dat in data if dat['chunk_size']==size][0] 
            net_list = [net for net in result_data['result_all'] if getattr(net,'chunk_size') == size] 
            min_list = [min(samp) for samp in sample_ch]
            max_list = [max(samp) for samp in sample_ch]
            for net in net_list:
                res[-1]['method'].append(net.net_name)
                err1stepr.append([b - a for a, b in zip(net.exp_r[:-1], max_list[1:])])
                err1stepl.append([b - a for a, b in zip(net.exp_l[:-1], min_list[1:])])
                errpopr.append([ (pop_max - exp) for exp in net.exp_r])
                errpopl.append([ (pop_min - exp) for exp in net.exp_l])
                errpop_range.append([pop_range - exp for exp in net.exp_range])
                # res[-1]['err1stepr'].append([b - a for a, b in zip(net.exp_r[:-1], max_list[1:])]) 
                # res[-1]['err1stepl'].append([b - a for a, b in zip(net.exp_l[:-1], min_list[1:])])
                # res[-1]['errpopr'].append([ (pop_max - exp) for exp in net.exp_r]) 
                # res[-1]['errpopl'].append([ (pop_min - exp) for exp in net.exp_l])
                # res[-1]['errpop_range'].append([ (pop_range - exp) for exp in net.exp_range]) 
                # res[-1]['nlearn'].append([[a+b for a,b in zip(net.nlearnl,net.nlearnr)]])
                # print(result)
            if len(errpopr[0]) > 30:    
                t_res = paired_ttest(errpopr)
                res[-1]['test_pop_errr'] = {'stat':'t-test',
                                            'values':t_res}
            else:
                w_res = paired_ttest(errpopr)
                res[-1]['test_pop_errr'] = {'stat':'w-test',
                                            'values':w_res}
            if len(errpopl[0]) > 30:    
                t_res = paired_ttest(errpopl)
                res[-1]['test_pop_errl'] = {'stat':'t-test',
                                            'values':t_res}
            else:
                w_res = paired_ttest(errpopl)
                res[-1]['test_pop_errl'] = {'stat':'w-test',
                                            'values':w_res}    
            if len(errpop_range[0]) > 30:    
                t_res = paired_ttest(errpop_range)
                res[-1]['test_pop_errrange'] = {'stat':'t-test',
                                            'values':t_res}
            else:
                w_res = paired_ttest(errpop_range)
                res[-1]['test_pop_errrange'] = {'stat':'w-test',
                                            'values':w_res}        
                
        if outlier:
            with open(os.path.join(folder_path,info['file_data_chunk']+'_anal_outlier.json'), 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)
        else:            
            with open(os.path.join(folder_path,info['file_data_chunk']+'_anal.json'), 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)

    print("Write resutls.json, completedly")    
        
    
                        
def main(opt):
    run(**vars(opt))
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)    