''' 
 Decription: Create the streaming data chunk from specified number of propotion of the 
            population data. 
 Arguments: folder_config => a folder containing population data file. 
            file_config => a config yaml file containing decription of the population file. 
 Output: streaming data chunk saved as the json file type. 
''' 
import pandas as pd 
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
    parser.add_argument("--dir", type = str, default="./config_sim_data/uniform", help = 'working directory')
    parser.add_argument("--file", type = str, default="config_uniform_simulate.yaml", help = 'config file')
    parser.add_argument("--pfeed", type = int, default=30, help = 'percent of samples out of population')
    opt = parser.parse_args()
    return opt

def run(dir:str,file:str,pfeed:int,ch_size:List = [50,500]):
    pathfile = os.path.join(dir,file)
    with open(pathfile,'r') as f:
        docs = yaml.safe_load_all(f)
        # Create filename list.
        list_filename = []
        for doc in docs:
            list_filename.append(doc['file_data_chunk'])
    
    for file in list_filename:
        pop_sim = pd.read_pickle(os.path.join(dir,file+'.pkl'))
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