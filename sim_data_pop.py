
''' 
 Decription: The python script is the main script for createing the 1-D simulation data from the config file.
            For an example of config file: config_wald.yaml, config_wiebull.yaml  
 Arguments: 
          - config_path: a path containing config file.
          - config_file: the .yaml file.     
 Output:
        - The folder contain pickle files of the simulated data.
        - The detail of each file written in .yaml format.
'''  
import numpy as np
import pandas as pd
import os, yaml
from scipy.stats import weibull_min
import polars as pl
import pickle
import matplotlib.pyplot as plt   

def read_yaml(ypath):
    with open(ypath,'r') as f:
         return yaml.safe_load(f)
def sim_1d(config_file_path):
    with open(config_file_path,'r') as f:
        yaml.safe_load(f)
    return 0
 
def main(config_path: str, fileload: str):
    ypath = os.path.join(config_path, fileload)

    # with open(ypath,'r') as f:
    #     doc = yaml.safe_load(f)
    doc = read_yaml(ypath=ypath)    
    
    path = os.path.join(config_path, doc['name'])
    if not os.path.isdir(path):
        os.makedirs(path)
    filesave = 'config_'+doc['name']+'_simulate.yaml'
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
    config_path = './config_sim_data/' # 
    file_config_wald = 'config_wald.yaml'
    file_config_wiebull = 'config_wiebull.yaml'
    main(config_path,file_config_wald)
    
    
