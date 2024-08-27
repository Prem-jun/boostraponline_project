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
    # file_path_re = os.path.join(folder_path,fold_re) 
    chunk_size = []
    net_online_list = []
    net_online_mm_list = []
    samples_size = 0
    try:
        # pop_data =  pd.read_pickle(filepath_pop)
        # pop_min = np.min(pop_data)
        # pop_max = np.max(pop_data)
        # pop_range = pop_max-pop_min
        json_data = read_json_file(file_path)
        for count, data in enumerate(json_data):
            chunk_size.append(data['chunk_size'])
            chunk_data = data['samp_chuck']
            # create the initial network
            net_online = lib_boostrap.booststream()
            net_online_mm = lib_boostrap.booststream()
            net_online_whole = lib_boostrap.booststream()
            net_offline = lib_boostrap.booststream()
            
            # Setting for online manner
            net_online.set_online()
            net_online_mm.set_online()
            net_online_whole.set_online()
            if count == 0:
                sample_whole = [] 
                for samples_chunk in chunk_data:
                    sample_whole = sample_whole + samples_chunk
                    expandsion = net_online.expand_bt_online(samples_chunk)
                    expandsion_mm = net_online_mm.expand_bt_online(samples_chunk)
                samples_size = len(sample_whole)
                net_online_whole.expand_bt_online(sample_whole)
                net_offline.expand_bt_trad(sample_whole)
                net_online_list.append(net_online)
                net_online_mm_list.append(net_online_mm)    
            else:
                for samples_chunk in chunk_data:
                    expandsion = net_online.expand_bt_online(samples_chunk)
                    expandsion_mm = net_online_mm.expand_bt_online(samples_chunk)
                net_online_list.append(net_online)
                net_online_mm_list.append(net_online_mm)    
                    
        # if not os.path.exists(file_path_re):
        #     # Create the directory
        #     os.makedirs(file_path_re)    
        data_wb = {
                'samples_size': samples_size,
                'chunk_sizes': chunk_size,
                'net_online': net_online_list,
                'net_online_mm': net_online_mm_list,
                'net_online_whole': net_online_whole,
                'net_offline': net_offline
            }
        # Specify the name of the pickle file
        pickle_file = os.path.join(folder_path,filename+'_re.pkl')
        # Save the variables to the pickle file
        with open(pickle_file, 'wb') as file:
            pickle.dump(data_wb, file)
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