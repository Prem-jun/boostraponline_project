''' 
Decription: 
    - The script file is created referring to main_boost.py 
Arguments:
Output:

''' 
import json, pickle
import os
import pandas as pd
import numpy as np
import lib_boostrap
import dataclasses

# Function to read a JSON file and return its contents as a Python object
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load the data from the file
    return data
     
def main(folder_path,filename):

    # Define the path to the JSON file
    filetype_re = '.pkl'
    filetype_samp = '.json'
    # filepath_pop = os.path.join(folder_path,filename+filetype_pop)
    filepath_samp = os.path.join(folder_path,filename+filetype_samp)
    # Read sample file
    file_path = filepath_samp
    fold_re = filename+'_re' 
    chunk_size = []
    net_online = []
    try:
        # pop_data =  pd.read_pickle(filepath_pop)
        # pop_min = np.min(pop_data)
        # pop_max = np.max(pop_data)
        # pop_range = pop_max-pop_min
        json_data = read_json_file(file_path)
        for count, data in enumerate(json_data):
            chunk_size.append(data['chunk_size'])
            chunk_data = data['samp_chuck']
            net_online1 = lib_boostrap.booststream()
            net_online_whole = lib_boostrap.booststream()
            net_offline = lib_boostrap.booststream()
            net_online1.set_online()
            net_online_whole.set_online()
            if count == 0:
                sample_whole = [] 
                for samples_chunk in chunk_data:
                    sample_whole = sample_whole + samples_chunk
                    expandsion = net_online1.expand_bt_online(samples_chunk)
                net_online_whole.expand_bt_online(sample_whole)
                net_offline.expand_bt_trad(sample_whole)
                net_online.append(net_online1)    
            else:
                for samples_chunk in chunk_data:
                    expandsion = net_online1.expand_bt_online(samples_chunk)
                net_online.append(net_online1)    
            
        
        # print(json_data)  # Print the contents of the JSON file
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"The file {file_path} is not a valid JSON.")
    
    
    return 0
            
if __name__=='__main__':
    folder_path = './config_sim_data/wiebull/'
    filename = 'wiebullshape1n10000'
    main(folder_path,filename)