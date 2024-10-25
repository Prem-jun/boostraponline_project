''' 
 Decription: Create the streaming data chunk from specified number of propotion of the 
            population data. 
 Arguments:
 Output:
''' 
import polars as pl
import pandas as pd 
import os, yaml
import json
from dataclasses import asdict
from lib_boostrap import samp1d

     
def main(folder_config:str = None,file_config:str = None, percent_feed:int = 30,
         chunk_size:list = [50,100,500]):
    if folder_config is None:
       folder_config = './config_sim_data/wiebull/' 
    if file_config is None:
       file_config = 'config_wiebull_simulate.yaml'
    pathfile = os.path.join(folder_config,file_config)
    with open(pathfile,'r') as f:
        docs = yaml.safe_load_all(f)
        list_filename = []
        list_filepath = []
        for doc in docs:
            #path_file = os.path.join(folder_config,doc['file_data_chunk']+'.pkl')
            list_filename.append(doc['file_data_chunk'])
            # list_filepath.append(path_file)
    
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

if __name__=='__main__':
    dist_select = 'normal'
    
    # program part
    if dist_select == 'normal':
        # ======== Normal distribution   
        folder_config = './config_sim_data/normal/' 
        file_config = 'config_normal_simulate.yaml'
        # ========
        
    if dist_select == 'wiebull':
        # ======== Wiebull distribution   
        folder_config = './config_sim_data/wiebull/' 
        file_config = 'config_wiebull_simulate.yaml'
        # ========
    if dist_select == 'wald':
        # ======== Wald distribution   
        folder_config = './config_sim_data/wald/' 
        file_config = 'config_wald_simulate.yaml'
        # ========
    main(folder_config,file_config)