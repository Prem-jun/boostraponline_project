''' 
 Decription: Create the streaming data chunk from specified number of propotion of the 
            population data. 
 Arguments: folder_config => a folder containing population data file. 
            file_config => a config yaml file containing decription of the population file. 
 Output: streaming data chunk saved as the json file type. 
''' 
import pandas as pd
import numpy as np 
import os, yaml
import json
from dataclasses import asdict
from online_bootstrap import samp1d
from pathlib import Path
import argparse
from typing import List

def get_file_format(filename: str) -> str:
    """ Get the format from a full filename.
    Args:
        filename: a string of filename.
    
    Returns:
       a file format as a string. 
       
    Raises:
        ValueError: If the full filename does not contain a period.   
    
    Examples:
    
    """     
    # Extract the file extension
    if '.' in filename:
        file_extension = filename.rsplit('.', 1)[1].lower()
        return file_extension
    else:
        raise ValueError("Cannot extract the format file")
        #return None

def inject_outliers(data_chunk, outlier_ratio=0.03, outlier_strength=5):
    """
    Inject outlier data into a sample chunk.

    Args:
        data_chunk (List[float] or np.ndarray): Original sample data chunk.
        outlier_ratio (float): Ratio of outliers to inject (e.g., 0.03 for 3%).
        outlier_strength (float): How far outliers deviate from the mean in terms of std.

    Returns:
        np.ndarray: New data chunk with injected outliers.
    """
    data_chunk = np.array(data_chunk)
    n_outliers = max(1, int(len(data_chunk) * outlier_ratio))

    # Calculate mean and std of original data
    mean = np.mean(data_chunk)
    std = np.std(data_chunk)

    # Create outliers (half on left, half on right)
    outliers = np.concatenate([
        np.random.normal(loc=mean - outlier_strength * std, scale=std * 0.5, size=n_outliers // 2),
        np.random.normal(loc=mean + outlier_strength * std, scale=std * 0.5, size=n_outliers - n_outliers // 2)
    ])

    # Combine and shuffle
    combined_data = np.concatenate([data_chunk, outliers])
    np.random.shuffle(combined_data)
    return outliers.tolist()  # Convert back to list for consistency

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
    parser.add_argument("--dir", type = str, default="./config_sim_data/chi2", help = 'working directory')
    parser.add_argument("--file", type = str, default="config_chi2_simulate.yaml", help = 'config file')
    parser.add_argument("--pfeed", type = int, default=30, help = 'percent of samples out of population')
    parser.add_argument("--outlier", action = 'store_true', help = 'Run with outlier contamination')
    # parser.add_argument("--savename", type = str, default="result-fdist-statanal2.csv", help = 'source for loading config file results')
    parser.set_defaults(outlier=True)
    opt = parser.parse_args()
    return opt

def run(dir:str,file:str,pfeed:int,outlier:bool,ch_size:List = [50,500]):
    pathfile = os.path.join(dir,file)
    with open(pathfile,'r') as f:
        docs = yaml.safe_load_all(f)
        # Create filename list.
        list_filename = []
        for doc in docs:
            list_filename.append(doc['file_data_chunk'])
    
    for file in list_filename:
        pop_sim = pd.read_pickle(os.path.join(dir,file+'.pkl'))
        if outlier:
            # Inject outliers into the population data
            outlier_sim = inject_outliers(pop_sim, outlier_ratio=0.0005, outlier_strength=10)
            samp_list = []
            for size in ch_size:
                samp_list.append(samp1d(file_config=dir, nsim=len(pop_sim),name = file, 
                                percent_feed=pfeed))
                samp_list[-1].split2chunk(pop_sim,size)
                if len(samp_list[-1].samp_chuck)>7:
                    samp_list[-1].samp_chuck[9].extend(outlier_sim)  # Inject outliers into the first chunk
                else:
                    samp_list[-1].samp_chuck[4].extend(outlier_sim)  # Inject outliers into the first chunk
            samp_list_dict = [asdict(samp) for samp in samp_list]  
            samp_json = json.dumps(samp_list_dict, indent=4)
            with open(os.path.join(dir,file+'_outlier.json'), 'w') as json_file:
                json_file.write(samp_json)  
        else:
            samp_list = []
            for size in ch_size:
                samp_list.append(samp1d(file_config=dir, nsim=len(pop_sim),name = file, 
                                percent_feed=pfeed))
                samp_list[-1].split2chunk(pop_sim,size) 
            samp_list_dict = [asdict(samp) for samp in samp_list]  
            samp_json = json.dumps(samp_list_dict, indent=4)
            with open(os.path.join(dir,file+'.json'), 'w') as json_file:
                json_file.write(samp_json)  
    return 0
def main(opt):
    run(**vars(opt))
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)