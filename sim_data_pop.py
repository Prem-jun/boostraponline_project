  
import numpy as np
import pandas as pd
import os, yaml
from scipy.stats import weibull_min
import polars as pl
import pickle
import matplotlib.pyplot as plt   
from typing import List, Union
from pathlib import Path
import argparse

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
 
# def main(config_path: str, fileload: str):
#     """ Create the 1-D simulated data from the specified config file. This data is considered as
#     population data. For an example of config file: 
#                 - config_wald.yaml, 
#                 - config_wiebull.yaml
#     Args:
#         config_path: a string of a path containing config file
#         fileload: a string of a path containing config file
    
#     Returns:
#         - The folder contain pickle files (pkl) and their related figure of the simulated data.
#         - The description of each file written in .yaml format. 
                
#     """
#     ypath = os.path.join(config_path, fileload)
#     doc = read_yaml(ypath=ypath)    
    
#     path = os.path.join(config_path, doc['name'])
#     if not os.path.isdir(path):
#         os.makedirs(path)
        
#     filesave = 'config_'+doc['name']+'_simulate.yaml'
    
#     if doc['name'] == 'realworld':
#         doc_sim = []
#         for i in range(len(doc['parameters']['filename'])):
#             filename = doc['parameters']['filename'][i]
#             file_format = doc['parameters']['filetype'][i]
#             if (file_format == 'csv') and (filename == 'laptop_prices'):
#                 df = pd.read_csv(filename+'.'+file_format)
#                 np_data = df['Price_euros'].to_numpy()
#                 list_data = list(np_data)
#                 path_fig =os.path.join(path, filename)
#                 pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
#                 # create config file ended wigh `.yaml`
#                 dict_tmp = {'file_config':ypath,
#                             'file_data_chunk':filename,
#                             'filetype':file_format
#                             }
#                 doc_sim.append(dict_tmp)
#             elif (file_format == 'csv') and (filename == 'Electronic_sales_Sep2023-Sep2024'):
#                 df = pd.read_csv(filename+'.'+file_format)
#                 np_data = df['Total Price'].to_numpy()
#                 list_data = list(np_data)
#                 path_fig =os.path.join(path, filename)
#                 pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
#                 # create config file ended wigh `.yaml`
#                 dict_tmp = {'file_config':ypath,
#                             'file_data_chunk':filename,
#                             'filetype':file_format
#                             }
#                 doc_sim.append(dict_tmp)
#             elif (file_format == 'csv') and (filename == 'Ecommerce_Sales_Prediction_Dataset'):    
#                 df = pd.read_csv(filename+'.'+file_format)
#                 np_data = df['Marketing_Spend'].to_numpy()
#                 list_data = list(np_data)
#                 path_fig =os.path.join(path, filename)
#                 pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
#                 # create config file ended wigh `.yaml`
#                 dict_tmp = {'file_config':ypath,
#                             'file_data_chunk':filename,
#                             'filetype':file_format
#                             }
#                 doc_sim.append(dict_tmp)
                
#             elif (file_format == 'csv') and (filename == 'world_tourism_economy_data'):    
#                 df = pd.read_csv(filename+'.'+file_format)
#                 df = df.dropna(subset=['tourism_expenditures'])
#                 np_data = df['tourism_expenditures'].to_numpy()
#                 list_data = list(np_data)
#                 path_fig =os.path.join(path, filename)
#                 pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
#                 # create config file ended wigh `.yaml`
#                 dict_tmp = {'file_config':ypath,
#                             'file_data_chunk':filename,
#                             'filetype':file_format
#                             }
#                 doc_sim.append(dict_tmp)    
                     
#     if doc['name'] == 'fdist':
#         doc_sim = []
#         for amount_sample in doc['parameters']['nsim']:
#             for dfd in doc['parameters']['dfd']:
#                 for dfn in doc['parameters']['dfn']:
#                     np_data = np.random.f(dfnum = dfn, dfden=dfn, size=amount_sample)
                    
#                     filename = doc['name']+'dfn'+str(dfn)+'dfd'+str(dfd)+'n'+str(amount_sample)
#                     fig, ax = plt.subplots(1, 1)
#                     ax.hist(np_data, density=False, bins='auto', histtype='stepfilled', alpha=0.2,label = 'Wald')
#                     plt.show()
#                     path_fig =os.path.join(path, filename)
#                     fig.savefig(path_fig)
#                     list_data = list(np_data)
#                     print('Histrogram figure done.')
                    
#                     pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
#                     # create config file ended wigh `.yaml`
#                     dict_tmp = {'file_config':ypath,
#                                 'file_data_chunk':filename,
#                                 'nsim':amount_sample,
#                                 'dfn':dfn,
#                                 'dfd':dfd,
#                                 }
#                     doc_sim.append(dict_tmp)    
    
#     if doc['name'] == 'normal':
#         doc_sim = []
#         for amount_sample in doc['parameters']['nsim']:
#             for mean in doc['parameters']['mean']:
#                 for scale in doc['parameters']['scale']:
#                     np_data  = np.random.normal(loc=mean,scale=scale,size = amount_sample)
#                     filename = doc['name']+'m'+str(mean)+'sd'+str(scale)+'n'+str(amount_sample)
#                     fig, ax = plt.subplots(1, 1)
#                     ax.hist(np_data, density=False, bins='auto', histtype='stepfilled', alpha=0.2,label = 'Wald')
#                     plt.show()
#                     path_fig =os.path.join(path, filename)
#                     fig.savefig(path_fig)
#                     list_data = list(np_data)
#                     print('Histrogram figure done.')
                    
#                     pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
#                     # create config file ended wigh `.yaml`
#                     dict_tmp = {'file_config':ypath,
#                                 'file_data_chunk':filename,
#                                 'nsim':amount_sample,
#                                 'mean':mean,
#                                 'scale':scale,
#                                 }
#                     doc_sim.append(dict_tmp)   
                    
#     if doc['name'] == 'wald':
#         doc_sim = []
#         for amount_sample in doc['parameters']['nsim']:
#             for mean in doc['parameters']['mean']:
#                 for scale in doc['parameters']['scale']:
#                     np_data  = np.random.wald(mean,scale,amount_sample)
#                     if scale<1:
#                         filename = doc['name']+'m'+str(mean)+'sd0'+str(int(scale*10))+'n'+str(amount_sample)
#                     else:    
#                         filename = doc['name']+'m'+str(mean)+'sd'+str(scale)+'n'+str(amount_sample)
#                     fig, ax = plt.subplots(1, 1)
#                     ax.hist(np_data, density=False, bins='auto', histtype='stepfilled', alpha=0.2,label = 'Wald')
#                     plt.show()
#                     path_fig =os.path.join(path, filename)
#                     fig.savefig(path_fig)
#                     list_data = list(np_data)
#                     print('Histrogram figure done.')
                    
#                     pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
#                     # create config file ended wigh `.yaml`
#                     dict_tmp = {'file_config':ypath,
#                                 'file_data_chunk':filename,
#                                 'nsim':amount_sample,
#                                 'mean':mean,
#                                 'scale':scale,
#                                 }
#                     doc_sim.append(dict_tmp)   
                    
#     if doc['name'] == 'wiebull':
#         doc_sim = []
#         for amount_sample in doc['parameters']['nsim']:
#             for shape_param in doc['parameters']['shape']: 
#                 np_data  = np.random.weibull(shape_param,amount_sample)
#                 filename = doc['name']+'shape'+str(shape_param)+'n'+str(amount_sample)
#                 fig, ax = plt.subplots(1, 1)
#                 ax.hist(np_data, density=False, bins='auto', histtype='stepfilled', alpha=0.2,label = 'Weibull')
#                 plt.show()
#                 path_fig =os.path.join(path, filename)      
#                 list_data = list(np_data)
#                 pickle.dump(list_data, open(path_fig+".pkl", "wb"))
                    
#                 dict_tmp = {'file_config':ypath,
#                             'file_data_chunk':filename,
#                             'nsim':amount_sample,
#                             'shape':shape_param
#                             }
#                 doc_sim.append(dict_tmp)
                
#     path_save = os.path.join(path, filesave)
#     # Writting config file `yaml`
#     with open(path_save, 'w') as file:
#         yaml.dump_all(doc_sim, file, sort_keys=False)
#     print('Simulation data has been done.')

# if __name__== '__main__':
#     config_path = './config_sim_data/' # main path
#     dist_select = 'wiebull' # specify the distribution
#     if dist_select is 'fdist':
#         file_config = 'config_fdist.yaml'
#     if dist_select == 'wiebull':
#         file_config = 'config_wiebull.yaml'
#     if dist_select == 'wald':
#         file_config = 'config_wald.yaml'
#     if dist_select == 'normal':
#         file_config = 'config_normal.yaml'
#     if dist_select == 'realworld':
#         file_config = 'config_real_labtop.yaml'
                
#     main(config_path,file_config)
def ensure_directory_exists(directory_path):
    """
    Ensure a directory exists, create it if it doesn't
    
    Args:
        directory_path (str): Path to the directory
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")
        return False
    
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
    parser.add_argument("--dir", type = str, default=ROOT/"config_sim_data", help = 'working directory')
    parser.add_argument("--file", type = str, default="config_uniform.yaml", help = 'config file')
    opt = parser.parse_args()
    return opt

def run(dir:str,file:str):
    source = os.path.join(dir, file)
    
    with open(source, 'r') as file:
        doc = yaml.safe_load(file)
    folder_save = os.path.join(dir,doc['name'])
    
    if not(ensure_directory_exists(folder_save)):
        print('Cannot create the saving foloder.')
    else:
        if doc['name'] == 'normal':
            doc_sim = []
            for amount_sample in doc['parameters']['nsim']:
                for mean in doc['parameters']['mean']:
                    for scale in doc['parameters']['scale']:
                        np_data  = np.random.normal(loc=mean,scale=scale,size = amount_sample)
                        filename = doc['name']+'m'+str(mean)+'sd'+str(scale)+'n'+str(amount_sample)
                        file_save = os.path.join(source,filename)
                        list_data = list(np_data)
                        pickle.dump(list_data, open(os.path.join(file_save,filename+".pkl"), "wb"))
                        
                        # create config file ended wigh `.yaml`
                        dict_tmp = {'file_config':source,
                                    'file_data_chunk':filename,
                                    'nsim':amount_sample,
                                    'mean':mean,
                                    'scale':scale,
                                    }
                        doc_sim.append(dict_tmp)
        if doc['name'] == 'uniform':
            doc_sim = []
            for amount_sample in doc['parameters']['nsim']:
                for max in doc['parameters']['max']:
                    for min in doc['parameters']['min']:
                        np_data = np.random.uniform(low=min, high=max, size=amount_sample)
                        filename = doc['name']+'min'+str(min)+'max'+str(max)+'n'+str(amount_sample)
                        list_data = list(np_data)
                    
                        pickle.dump(list_data, open(os.path.join(folder_save,filename+".pkl"), "wb"))
                        
                        # create config file ended wigh `.yaml`
                        dict_tmp = {'file_config':source,
                                    'file_data_chunk':filename,
                                    'nsim':amount_sample,
                                    'min':min,
                                    'max':max,
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
                        
                        list_data = list(np_data)
                        
                        pickle.dump(list_data, open(os.path.join(folder_save,filename+".pkl"), "wb"))
                        
                        # create config file ended wigh `.yaml`
                        dict_tmp = {'file_config':source,
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
                    list_data = list(np_data)
                    pickle.dump(list_data, open(os.path.join(folder_save,filename+".pkl"), "wb"))
                        
                    dict_tmp = {'file_config':source,
                                'file_data_chunk':filename,
                                'nsim':amount_sample,
                                'shape':shape_param
                                }
                    doc_sim.append(dict_tmp)
        if doc['name'] == 'fdist':
            doc_sim = []
            for amount_sample in doc['parameters']['nsim']:
                for dfd in doc['parameters']['dfd']:
                    for dfn in doc['parameters']['dfn']:
                        np_data = np.random.f(dfnum = dfn, dfden=dfn, size=amount_sample)
                        filename = doc['name']+'dfn'+str(dfn)+'dfd'+str(dfd)+'n'+str(amount_sample)
                        list_data = list(np_data)
                        pickle.dump(list_data, open(os.path.join(folder_save,filename+".pkl"), "wb"))
                        
                        # create config file ended wigh `.yaml`
                        dict_tmp = {'file_config':source,
                                    'file_data_chunk':filename,
                                    'nsim':amount_sample,
                                    'dfn':dfn,
                                    'dfd':dfd,
                                    }
                        doc_sim.append(dict_tmp)
        if doc['name'] == 'realworld':
            doc_sim = []
            for i in range(len(doc['parameters']['filename'])):
                filename = doc['parameters']['filename'][i]
                file_format = doc['parameters']['filetype'][i]
                if (file_format == 'csv') and (filename == 'laptop_prices'):
                    df = pd.read_csv(filename+'.'+file_format)
                    np_data = df['Price_euros'].to_numpy()
                    list_data = list(np_data)
                    pickle.dump(list_data, open(os.path.join(folder_save,filename+".pkl"), "wb"))
                        
                    # create config file ended wigh `.yaml`
                    dict_tmp = {'file_config':source,
                                'file_data_chunk':filename,
                                'filetype':file_format
                                }
                    doc_sim.append(dict_tmp)
                elif (file_format == 'csv') and (filename == 'Electronic_sales_Sep2023-Sep2024'):
                    df = pd.read_csv(filename+'.'+file_format)
                    np_data = df['Total Price'].to_numpy()
                    list_data = list(np_data)
                    pickle.dump(list_data, open(os.path.join(folder_save,filename+".pkl"), "wb"))
                        
                    # create config file ended wigh `.yaml`
                    dict_tmp = {'file_config':source,
                                'file_data_chunk':filename,
                                'filetype':file_format
                                }
                    doc_sim.append(dict_tmp)
                elif (file_format == 'csv') and (filename == 'Ecommerce_Sales_Prediction_Dataset'):    
                    df = pd.read_csv(filename+'.'+file_format)
                    np_data = df['Marketing_Spend'].to_numpy()
                    list_data = list(np_data)
                    
                    pickle.dump(list_data, open(os.path.join(folder_save,filename+".pkl"), "wb"))
                        
                    # create config file ended wigh `.yaml`
                    dict_tmp = {'file_config':source,
                                'file_data_chunk':filename,
                                'filetype':file_format
                                }
                    doc_sim.append(dict_tmp)
                    
                elif (file_format == 'csv') and (filename == 'world_tourism_economy_data'):    
                    df = pd.read_csv(filename+'.'+file_format)
                    df = df.dropna(subset=['tourism_expenditures'])
                    np_data = df['tourism_expenditures'].to_numpy()
                    list_data = list(np_data)
                    
                    pickle.dump(list_data, open(os.path.join(folder_save,filename+".pkl"), "wb"))
                        
                    # create config file ended wigh `.yaml`
                    dict_tmp = {'file_config':source,
                                'file_data_chunk':filename,
                                'filetype':file_format
                                }
                    doc_sim.append(dict_tmp)    
                            
        
        # Writting config file `yaml`
        with open(os.path.join(folder_save,'config_'+doc['name']+"_simulate.yaml"), 'w') as file:
            yaml.dump_all(doc_sim, file, sort_keys=False)
        print('Simulation data has been done.')
    
def main(opt):
    run(**vars(opt))
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)    