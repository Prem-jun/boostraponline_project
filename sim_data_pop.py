  
import numpy as np
import pandas as pd
import os, yaml
from scipy.stats import weibull_min
import polars as pl
import pickle
import matplotlib.pyplot as plt   
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

def read_yaml(ypath: str)->List:
    """ Read .yaml file.
    Args:
        filename: a string of filename.
    
    Returns:
       A list of data. 
       
    Raises:
        -
    
    Examples:
    
    """
    with open(ypath,'r') as f:
         return yaml.safe_load(f)
     
def sim_1d(config_file_path):
    with open(config_file_path,'r') as f:
        yaml.safe_load(f)
    return 0
 
def main(config_path: str, fileload: str):
    """ Create the 1-D simulated data from the specified config file. This data is considered as
    population data. For an example of config file: 
                - config_wald.yaml, 
                - config_wiebull.yaml
    Args:
        config_path: a string of a path containing config file
        fileload: a string of a path containing config file
    
    Returns:
        - The folder contain pickle files (pkl) and their related figure of the simulated data.
        - The description of each file written in .yaml format. 
                
    """
    ypath = os.path.join(config_path, fileload)
    doc = read_yaml(ypath=ypath)    
    
    path = os.path.join(config_path, doc['name'])
    if not os.path.isdir(path):
        os.makedirs(path)
        
    filesave = 'config_'+doc['name']+'_simulate.yaml'
    
    if doc['name'] == 'realworld':
        doc_sim = []
        for i in range(len(doc['parameters']['filename'])):
            filename = doc['parameters']['filename'][i]
            file_format = doc['parameters']['filetype'][i]
            if file_format == 'csv':
                df = pd.read_csv(filename+'.'+file_format)
                np_data = df['Price_euros'].to_numpy()
                list_data = list(np_data)
                path_fig =os.path.join(path, filename)
                pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
                # create config file ended wigh `.yaml`
                dict_tmp = {'file_config':ypath,
                            'file_data_chunk':filename,
                            'filetype':file_format
                            }
                doc_sim.append(dict_tmp)
                     
        
    
    if doc['name'] == 'normal':
        doc_sim = []
        for amount_sample in doc['parameters']['nsim']:
            for mean in doc['parameters']['mean']:
                for scale in doc['parameters']['scale']:
                    np_data  = np.random.normal(loc=mean,scale=scale,size = amount_sample)
                    filename = doc['name']+'m'+str(mean)+'sd'+str(scale)+'n'+str(amount_sample)
                    fig, ax = plt.subplots(1, 1)
                    ax.hist(np_data, density=False, bins='auto', histtype='stepfilled', alpha=0.2,label = 'Wald')
                    plt.show()
                    path_fig =os.path.join(path, filename)
                    fig.savefig(path_fig)
                    list_data = list(np_data)
                    print('Histrogram figure done.')
                    
                    pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
                    # create config file ended wigh `.yaml`
                    dict_tmp = {'file_config':ypath,
                                'file_data_chunk':filename,
                                'nsim':amount_sample,
                                'mean':mean,
                                'scale':scale,
                                }
                    doc_sim.append(dict_tmp)   
                    
    if doc['name'] == 'wald':
        doc_sim = []
        for amount_sample in doc['parameters']['nsim']:
            for mean in doc['parameters']['mean']:
                for scale in doc['parameters']['scale']:
                    np_data  = np.random.wald(mean,scale,amount_sample)
                    if scale<1:
                        filename = doc['name']+'m'+str(mean)+'sd0'+str(int(scale*10))+'n'+str(amount_sample)
                    else:    
                        filename = doc['name']+'m'+str(mean)+'sd'+str(scale)+'n'+str(amount_sample)
                    fig, ax = plt.subplots(1, 1)
                    ax.hist(np_data, density=False, bins='auto', histtype='stepfilled', alpha=0.2,label = 'Wald')
                    plt.show()
                    path_fig =os.path.join(path, filename)
                    fig.savefig(path_fig)
                    list_data = list(np_data)
                    print('Histrogram figure done.')
                    
                    pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
                    # create config file ended wigh `.yaml`
                    dict_tmp = {'file_config':ypath,
                                'file_data_chunk':filename,
                                'nsim':amount_sample,
                                'mean':mean,
                                'scale':scale,
                                }
                    doc_sim.append(dict_tmp)   
                    
    if doc['name'] == 'wiebull':
        doc_sim = []
        for amount_sample in doc['parameters']['nsim']:
            for shape_param in doc['parameters']['shape']: 
                np_data  = np.random.weibull(shape_param,amount_sample)
                filename = doc['name']+'shape'+str(shape_param)+'n'+str(amount_sample)
                fig, ax = plt.subplots(1, 1)
                ax.hist(np_data, density=False, bins='auto', histtype='stepfilled', alpha=0.2,label = 'Weibull')
                plt.show()
                path_fig =os.path.join(path, filename)      
                list_data = list(np_data)
                pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
                dict_tmp = {'file_config':ypath,
                            'file_data_chunk':filename,
                            'nsim':amount_sample,
                            'shape':shape_param
                            }
                doc_sim.append(dict_tmp)
                
    path_save = os.path.join(path, filesave)
    # Writting config file `yaml`
    with open(path_save, 'w') as file:
        yaml.dump_all(doc_sim, file, sort_keys=False)
    print('Simulation data has been done.')

if __name__=='__main__':
    config_path = './config_sim_data/' # main path
    dist_select = 'realworld' # specify the distribution
    if dist_select == 'wiebull':
        file_config = 'config_wiebull.yaml'
    if dist_select == 'wald':
        file_config = 'config_wald.yaml'
    if dist_select == 'normal':
        file_config = 'config_normal.yaml'
    if dist_select == 'realworld':
        file_config = 'config_real_labtop.yaml'
                
    main(config_path,file_config)
