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
from lib_boostrap import samp1d
from pathlib import Path
import argparse
from typing import List, Union

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
     
# def main(folder_config:str = None,file_config:str = None, percent_feed:int = 30,
#          chunk_size:list = [50,100,500]):
#     """ Get the format from a full filename.
#     Args:
#         folder_config: a string of folder containing configuration file of simulated population.
#         file_config: a string of file name of configuration files.
#         percent_feed: A percent of samples selected as training, as int.
#         chunk_size: A list of number of samples for a chunk.
    
#     Returns:
#        No return, the results are the streaming data chunk in 1-D. 
       
#     Raises:
#         -   
    
#     Examples:
    
#     """
#     if folder_config is None:
#        folder_config = './config_sim_data/wiebull/' 
#     if file_config is None:
#        file_config = 'config_wiebull_simulate.yaml'
#     pathfile = os.path.join(folder_config,file_config)
#     with open(pathfile,'r') as f:
#         docs = yaml.safe_load_all(f)
#         list_filename = []
#         list_filepath = []
#         for doc in docs:
#             #path_file = os.path.join(folder_config,doc['file_data_chunk']+'.pkl')
#             list_filename.append(doc['file_data_chunk'])
#             # list_filepath.append(path_file)
    
#     for file in list_filename:
#         pop_sim = pd.read_pickle(os.path.join(folder_config,file+'.pkl'))
#         samp_list = []
#         for size in chunk_size:
#             samp_list.append(samp1d(file_config=file_config, nsim=len(pop_sim),name = file, 
#                                percent_feed=percent_feed))
#             samp_list[-1].split2chunk(pop_sim,size) 
#         samp_list_dict = [asdict(samp) for samp in samp_list]  
#         samp_json = json.dumps(samp_list_dict, indent=4)
#         with open(os.path.join(folder_config,file+'.json'), 'w') as json_file:
#             json_file.write(samp_json)  
#     return 0

# if __name__=='__main__':
#     dist_select = 'wiebull'
    
#     # program part
#     if dist_select == 'realworld':
#         folder_config = './config_sim_data/realworld/' 
#         file_config = 'config_realworld_simulate.yaml'
#         main(folder_config,file_config,chunk_size=[50,100])
#     else:
#         if dist_select == 'fdist':
#             # ======== Normal distribution   
#             folder_config = './config_sim_data/fdist/' 
#             file_config = 'config_fdist_simulate.yaml'
#             # ========
#         if dist_select == 'normal':
#             # ======== Normal distribution   
#             folder_config = './config_sim_data/normal/' 
#             file_config = 'config_normal_simulate.yaml'
#             # ========
            
#         if dist_select == 'wiebull':
#             # ======== Wiebull distribution   
#             folder_config = './config_sim_data/wiebull/' 
#             file_config = 'config_wiebull_simulate.yaml'
#             # ========
#         if dist_select == 'wald':
#             # ======== Wald distribution   
#             folder_config = './config_sim_data/wald/' 
#             file_config = 'config_wald_simulate.yaml'
#             # ========
        
#         main(folder_config,file_config)

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
        description='Population simulation',
        epilog='Example: python script.py --dir config_sim_data --file config_uniform.yaml',  # ข้อความท้าย help
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # แสดง default values
        prog='Population generator-Tool'
        )
    ROOT = Path(__file__).parent
    parser.add_argument("--dir", type = str, default=ROOT/"config_sim_data/normal", help = 'working directory')
    parser.add_argument("--file", type = str, default="config_normal_simulate.yaml", help = 'config file')
    opt = parser.parse_args()
    return opt

def run(dir:str,file:str):
    pathfile = os.path.join(dir,file)
    with open(pathfile,'r') as f:
        docs = yaml.safe_load_all(f)
        # Create filename list.
        list_filename = []
        for doc in docs:
            list_filename.append(doc['file_data_chunk'])
    
    for file in list_filename:
        pop_sim = pd.read_pickle(os.path.join(folder_config,file+'.pkl'))
        samp_list = []
        for size in chunk_size:
            samp_list.append(samp1d(file_config=file_config, nsim=len(pop_sim),name = file, 
                               percent_feed=percent_feed))
            samp_list[-1].split2chunk(pop_sim,size) 
        samp_list_dict = [asdict(samp) for samp in samp_list]  
        samp_json = json.dumps(samp_list_dict, indent=4)
        with open(os.path.join(folder_config,file+'.json'), 'w') as json_file:
            json_file.write(samp_json)  
    return 0
def main(opt):
    run(**vars(opt))
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)