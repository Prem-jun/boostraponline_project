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
import lib_boostrap
from typing import List, Union, Dict
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import yaml

@dataclass
class Res_boostrap:
    net_name: str = ''
    net:lib_boostrap.booststream = field(default_factory=lib_boostrap.booststream) 
    chunk_size: int=0
    num_chunk: int =0
    exp_l:List[float] = field(default_factory=list)
    exp_r:List[float] = field(default_factory=list)
    exp_range:List[float] =field(default_factory=list)
    nlearnl:List[float] =field(default_factory=list)
    nlearnr:List[float] =field(default_factory=list)
    
    def add_init_params(self, net:lib_boostrap.booststream, cum:bool=False):
        # add net_name attributes.
        self.net = net
        if self.net.online:
            if not self.net.minmax_boost:
                self.net_name = 'online' if not cum else 'online_cum'
            else:
                self.net_name = 'online_mm' if not cum else 'online_mm_cum'
        else:
            self.net_name = 'offline'
        
            
    
    def add_params(self,net:lib_boostrap.booststream):
        self.net = net
        if self.num_chunk == 0:
            self.chunk_size = self.net.chunk_size
        self.num_chunk +=1
        self.exp_l.append(self.net.exp_l)
        self.exp_r.append(self.net.exp_r)
        self.exp_range.append(self.net.range)
        if len(self.nlearnl) == 0:
            self.nlearnl.append(0)
        else: 
            self.nlearnl.append(self.net.nlearn_l[-1])
        if len(self.nlearnr) == 0:
            self.nlearnr.append(0)
        else:    
            self.nlearnr.append(self.net.nlearn_r[-1])
            
# Function to read a JSON file and return its contents as a Python object
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load the data from the file
    return data
     
# def main(folder_path:str ,filename:str):
    
#     # Define the path to the JSON file
#     filetype_re = '.pkl'
#     filetype_samp = '.json'
    
#     # filepath_pop = os.path.join(folder_path,filename+filetype_pop)
#     filepath_samp = os.path.join(folder_path,filename+filetype_samp)
    
#     # Read sample file
#     file_path = filepath_samp
#     # fold_re = filename+'_re'
    
#     # file_path_re = os.path.join(folder_path,fold_re) 
#     res_all = []
#     try:
#         # pop_data =  pd.read_pickle(filepath_pop)
#         # pop_min = np.min(pop_data)
#         # pop_max = np.max(pop_data)
#         # pop_range = pop_max-pop_min
#         json_data = read_json_file(file_path)
#         for count, data in enumerate(json_data):
            
#             # chunk_size.append(data['chunk_size'])
#             chunk_data = data['samp_chuck']
            
#             # create the initial network
#             net_online = lib_boostrap.booststream()
#             net_online_mm = lib_boostrap.booststream()
#             net_online_cum = lib_boostrap.booststream()
#             net_trad_cum = lib_boostrap.booststream()
            
#             # Setting for online manner
#             net_online.set_online()
#             net_online_mm.set_online(minmax_flag=True)
#             net_online_cum.set_online()
            
#             # Setting results 
#             res4 = Res_boostrap()
#             res4.add_init_params(net = net_trad_cum)
#             res1 = Res_boostrap()
#             res1.add_init_params(net = net_online)
#             res2 = Res_boostrap()
#             res2.add_init_params(net = net_online_mm)
#             res3 = Res_boostrap()
#             res3.add_init_params(net = net_online_cum,cum=True)
#             sample_whole = []
#             count2 = 0 
#             for samples_chunk in chunk_data:
#                 count2 +=1
#                 sample_whole = sample_whole + samples_chunk
#                 # Train online chunk
#                 print(f'Online running round:{count}/{count2}')
#                 expandsion = net_online.expand_bt_online(samples_chunk)
#                 res1.add_params(net_online)
#                 print(f'Online minmax running round:{count}/{count2}')
#                 expandsion_mm = net_online_mm.expand_bt_online(samples_chunk)
#                 res2.add_params(net_online_mm)
#                 # print(f'Online whole running round:{count}/{count2}')
#                 # expansion_ocum = net_online_cum.expand_bt_online(sample_whole,cum=True)
#                 # res3.add_params(net_online_cum)
#                 print(f'Offline running round:{count}/{count2}')
#                 expansion_tcum = net_trad_cum.expand_bt_trad(sample_whole)
#                 res4.add_params(net_trad_cum)
            
#             res_all.append(res1)
#             res_all.append(res2)
#             # res_all.append(res3)
#             res_all.append(res4)
#             print(f'Round running:{count}')
#         data_wb = {
#                 'result_all': res_all
#             }
#         # Specify the name of the pickle file
#         pickle_file = os.path.join(folder_path,filename+'_re.pkl')
#         # Save the variables to the pickle file
#         with open(pickle_file, 'wb') as file:
#             pickle.dump(data_wb, file)
#         # print(json_data)  # Print the contents of the JSON file
#     except FileNotFoundError:
#         print(f"The file {file_path} was not found.")
#     except json.JSONDecodeError:
#         print(f"The file {file_path} is not a valid JSON.")
    
    
#     #return 0
            
# if __name__=='__main__':
#     dist_list = ['normal','wald','wiebull','fdist','realworld']
#     idx = 4
#     dist_select = dist_list[idx]
#     # dist_select = 'realworld'
#     # ====== program part
#     if dist_select == 'realworld':
#         # normal distribution
#         folder_path = './config_sim_data/realworld/'
#         # filename_list = ['laptop_prices','Electronic_sales_Sep2023-Sep2024',
#         #                  'Ecommerce_Sales_Prediction_Dataset','world_tourism_economy_data']
#         filename_list = ['world_tourism_economy_data']
    
#     if dist_select == 'fdist':
#         # normal distribution
#         folder_path = './config_sim_data/fdist/'
#         # filename_list = ['normalm0sd1n10000','normalm0sd25n10000',
#         #                  'normalm0sd25n50000','normalm0sd100n50000']
#         filename_list = ['fdistdfn5dfd10n10000','fdistdfn5dfd15n10000','fdistdfn5dfd20n10000']
    
#     if dist_select == 'normal':
#         # normal distribution
#         folder_path = './config_sim_data/normal/'
#         # filename_list = ['normalm0sd1n10000','normalm0sd25n10000',
#         #                  'normalm0sd25n50000','normalm0sd100n50000']
#         filename_list = ['normalm0sd1n10000','normalm0sd4n10000']
#         # ======
#     if dist_select == 'wiebull':
#         # wiebull distribution
#         folder_path = './config_sim_data/wiebull/'
#         # filename_list = ['wiebullshape5n50000']
#         filename_list = ['wiebullshape1n10000','wiebullshape5n10000']
        
#         # ======
#     if dist_select == 'wald':    
#         # wald distribution
#         folder_path = './config_sim_data/wald/'
#         filename_list = ['waldm1sd2n10000','waldm1sd05n10000']
#         # filename_list = ['waldm1sd2n10000','waldm1sd05n10000',
#         #             'waldm3sd2n10000','waldm3sd05n10000','waldm3sd05n10000']
#         # filename_list = ['waldm1sd2n10000','waldm1sd2n50000','waldm1sd05n10000','waldm1sd05n50000',
#         #             'waldm3sd2n10000','waldm3sd2n50000','waldm3sd05n10000','waldm3sd05n10000']
#         # ======
#     print(f"Your selected distribution is: {dist_select}")
#     flag = input("Running y/n: ")
#     if (flag == 'y') or (flag == 'Y'):    
#         for filename in filename_list:
#             main(folder_path,filename)
#     else:
#         print(f'Please, change the selected distribution')    

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
    parser.add_argument("--dir", type = str, default=ROOT/"config_sim_data/uniform", help = 'working directory')
    parser.add_argument("--file", type = str, default="config_uniform_simulate.yaml", help = 'config file')
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

def run(dir:str,file:str):
    yaml_path = os.path.join(dir,file)
    configs = read_yaml_config(yaml_path)
    for config in configs:
        res_all = []
        json_data = read_json_file(os.path.join(dir,config['file_data_chunk']+'.json'))
        for count, data in enumerate(json_data):
            # chunk_size.append(data['chunk_size'])
            chunk_data = data['samp_chuck']
            # create the initial network
            net_online = lib_boostrap.booststream()
            net_online_mm = lib_boostrap.booststream()
            net_online_cum = lib_boostrap.booststream()
            net_trad_cum = lib_boostrap.booststream()
            
            # Setting for online manner
            net_online.set_online()
            net_online_mm.set_online(minmax_flag=True)
            net_online_cum.set_online()
            
            # Setting results 
            res4 = Res_boostrap()
            res4.add_init_params(net = net_trad_cum)
            res1 = Res_boostrap()
            res1.add_init_params(net = net_online)
            res2 = Res_boostrap()
            res2.add_init_params(net = net_online_mm)
            res3 = Res_boostrap()
            res3.add_init_params(net = net_online_cum,cum=True)
            sample_whole = []
            count2 = 0 
            for samples_chunk in chunk_data:
                count2 +=1
                sample_whole = sample_whole + samples_chunk
                # Train online chunk
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