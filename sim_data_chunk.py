''' 
 Decription: This python script is used to create stream chunks from the given popupation.
 Arguments: population data folder.
 Output: streaming chunk with relavant statistical description.
''' 

import numpy as np
import pandas as pd
import os, math, random, pickle, yaml #, h5py

import glob

def split2chunk(pop: list, chunk_size, percent_feed):
    pop_n = len(pop)
    samp_num = math.ceil((percent_feed/100)*pop_n)
    num_ch = math.floor(samp_num/chunk_size)
    list_chunk = [pop[i:i+chunk_size] for i in range(0,(chunk_size*(num_ch-1)),chunk_size)]
    list_chunk.append(pop[(chunk_size*(num_ch-1)):(chunk_size*(num_ch-1))+chunk_size])
    return list_chunk
    

# create data chunk
def main(filewd = None, dist_folder = None,file_config = None,percent_feed = None, ch_size = None,num_ch_list = None):
    if filewd  is None:
        filewd = './sim_data/'
    if dist_folder is None: 
        dist_folder = 'wald'
    if file_config is None:
        file_config = 'config_wald_simulate.yaml'
    if percent_feed is None:
        percent_feed = 30
    if ch_size is None:
        ch_size = [50,100,500]
    if num_ch_list is None:
        num_ch_list = 1
    res_folder = 'Chunk'
    path_res = os.path.join(filewd, dist_folder,res_folder)
    path_name = os.path.join(filewd, dist_folder,file_config)
    if not os.path.isdir(path_res):
        os.makedirs(path_res)
    
    with open(path_name,'r') as f:
        docs = yaml.safe_load_all(f)
        list_ = []
        list_filename = []
        for doc in docs:
            list_filename.append(doc['file_data_chunk'])
            path_file = os.path.join(filewd, dist_folder,doc['file_data_chunk']+'.p')
            list_.append(path_file)
    count = 0    
    for file in list_:
        pop_data = pickle.load(open(file,"rb"))
        pop_n = len(pop_data)
        pop_stat = {'min':min(pop_data),'max':max(pop_data),'n':pop_n}        
        samp_num = math.ceil((percent_feed/100)*pop_n)
        samp_stat = {'min':min(pop_data[0:samp_num]),'max':max(pop_data[0:samp_num]),'n':samp_num}
        list_sch = []
        for item in ch_size:  
            list_sch.append(split2chunk(pop_data, item, percent_feed))
            
        file_save = list_filename[count].rsplit('.',1)[0]+"_ch.p"
        pickle.dump([pop_data,list_sch,percent_feed,pop_stat,samp_stat], \
                            open(os.path.join(path_res,file_save), "wb"))
        count +=1

if __name__=='__main__':
    filewd = './config_sim_data'
    dist_folder = 'weibull'
    file_config = 'config_wiebull_simulate.yaml'
    # dist_folder =  'wiebull'
    # file_config = 'config_wiebull_simulate.yaml'
    res = main(filewd=filewd,dist_folder=dist_folder,file_config=file_config) 
    
    
    
    
    
    