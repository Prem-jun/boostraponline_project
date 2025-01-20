import argparse
import yaml
import pickle, os
import pandas as pd
import numpy as np
import openpyxl
from typing import List, Union, Dict
from dataclasses import dataclass, field
import lib_boostrap
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

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



def plot_err_line(ch_size:int ,error_list:List, name:List,filesave:str, yaxis_title:str = 'Range error',
                  xaxis_title:str = 'Number of samples',color_list:List =  ['brown','blue','green'],
                  position:str ='middle-right'):
    # Sample data for the line plots
    x = [(i+1)*ch_size for i in range(len(error_list[0]))]  # Common x-axis

    """
    # Data for three different lines
    y1 = exp_l[0][0]     # First line
    y2 = exp_l[0][1]     # Second line
    y3 = exp_l[0][2]     # Third line
    """
    # Create a figure
    fig = go.Figure()
    color = ['brown','blue','green']
    for i in range(len(error_list)):
        fig.add_trace(go.Scatter(x=x, y=error_list[i], mode='lines+markers', name=name[i], line=dict(color=color_list[i])))  
    # Update layout with titles and labels
    if position =='middle-right':
        fig.update_layout(
            # title='Three Line Plots with Base Value Line',
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend=dict(
                font=dict(
                family="Arial",
                size=10,  # ขนาดตัวอักษรของรายการ
                color="black",
                
                ),
                yanchor="middle",
                y=0.4,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0)"
                # bgcolor="white",   # สีพื้นหลังของ legend
                # bordercolor="black", # สีขอบ legend
                # borderwidth=1      # ความหนาของเส้นขอบ
            ),
            # legend_title='Lines',
            template='plotly'
        )  
    elif position == 'top-right':
        fig.update_layout(
            # title='Three Line Plots with Base Value Line',
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend=dict(
                font=dict(
                family="Arial",
                size=10,  # ขนาดตัวอักษรของรายการ
                color="black",
                
                ),
                yanchor="top",
                y=0.9,
                xanchor="right",
                x=0.9,
                bgcolor="rgba(0,0,0,0)"
                # bgcolor="white",   # สีพื้นหลังของ legend
                # bordercolor="black", # สีขอบ legend
                # borderwidth=1      # ความหนาของเส้นขอบ
            ),
            # legend_title='Lines',
            template='plotly'
        )  
    pio.write_image(fig, filesave+'.png')  
    fig.show()

def plot_nlearn_line(ch_size:int, nlearnl:List, name:List, filesave:str,
                     yaxis_title:str = 'Numbers of samples for bootstraping',
                     xaxis_title:str = 'Number of samples',color_list:List =  ['brown','blue','green'],
                     position:str = 'middle-right'):
    # Sample data for the line plots
    x = [(i+1)*ch_size for i in range(len(nlearnl[0]))]  # Common x-axis
    # Create a figure
    fig = go.Figure()
   
    for i in range(len(nlearnl)-1):
        fig.add_trace(go.Scatter(x=x, y=nlearnl[i], mode='lines+markers', name=name[i], line=dict(color=color_list[i])))  
    # Update layout with titles and labels
    if position =='middle-right':
        fig.update_layout(
            # title='Three Line Plots with Base Value Line',
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend=dict(
                font=dict(
                family="Arial",
                size=10,  # ขนาดตัวอักษรของรายการ
                color="black",
                
                ),
                yanchor="middle",
                y=0.4,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0)"
                # bgcolor="white",   # สีพื้นหลังของ legend
                # bordercolor="black", # สีขอบ legend
                # borderwidth=1      # ความหนาของเส้นขอบ
            ),
            # legend_title='Lines',
            template='plotly'
        )  
    elif position == 'top-right':
        fig.update_layout(
            # title='Three Line Plots with Base Value Line',
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend=dict(
                font=dict(
                family="Arial",
                size=10,  # ขนาดตัวอักษรของรายการ
                color="black",
                
                ),
                yanchor="top",
                y=0.9,
                xanchor="right",
                x=0.9,
                bgcolor="rgba(0,0,0,0)"
                # bgcolor="white",   # สีพื้นหลังของ legend
                # bordercolor="black", # สีขอบ legend
                # borderwidth=1      # ความหนาของเส้นขอบ
            ),
            # legend_title='Lines',
            template='plotly'
        )  
    pio.write_image(fig, filesave+'.png')
    fig.show()
    
    
def plot_minmax_line(ch_size:int ,exp_l:List, exp_r:List, name_l:List,name_r:List, popminmax:List,filesave:str,
                     yaxis_title:str = 'Value',xaxis_title:str = 'Number of samples',
                     color_list:List =  ['brown','blue','green'],
                     position:str = 'middle-right' ):
    # Sample data for the line plots
    x = [(i+1)*ch_size for i in range(len(exp_l[0]))]  # Common x-axis

    """
    # Data for three different lines
    y1 = exp_l[0][0]     # First line
    y2 = exp_l[0][1]     # Second line
    y3 = exp_l[0][2]     # Third line
    """
    # Create a figure
    fig = go.Figure()
   
    for i in range(len(exp_l)):
        fig.add_trace(go.Scatter(x=x, y=exp_l[i], mode='lines+markers', name=name_l[i], line=dict(color=color_list[i])))  
        fig.add_trace(go.Scatter(x=x, y=exp_r[i], mode='lines+markers', name=name_r[i], line=dict(color=color_list[i])))  
    # Add horizontal line for the base value
    fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[popminmax[0], popminmax[0]], 
                            mode='lines', name=f'min pop: {popminmax[0]:.4f}', 
                            line=dict(color='grey', dash='dash')))
    fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[popminmax[1], popminmax[1]], 
                            mode='lines', name=f'max pop: {popminmax[1]:.4f}', 
                            line=dict(color='grey', dash='dash')))
    # Update layout with titles and labels
    if position =='middle-right':
        fig.update_layout(
            # title='Three Line Plots with Base Value Line',
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend=dict(
                font=dict(
                family="Arial",
                size=10,  # ขนาดตัวอักษรของรายการ
                color="black",
                
                ),
                yanchor="middle",
                y=0.4,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0)"
                # bgcolor="white",   # สีพื้นหลังของ legend
                # bordercolor="black", # สีขอบ legend
                # borderwidth=1      # ความหนาของเส้นขอบ
            ),
            # legend_title='Lines',
            template='plotly'
        )  
    elif position == 'top-right':
        fig.update_layout(
            # title='Three Line Plots with Base Value Line',
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend=dict(
                font=dict(
                family="Arial",
                size=10,  # ขนาดตัวอักษรของรายการ
                color="black",
                
                ),
                yanchor="top",
                y=0.9,
                xanchor="right",
                x=0.9,
                bgcolor="rgba(0,0,0,0)"
                # bgcolor="white",   # สีพื้นหลังของ legend
                # bordercolor="black", # สีขอบ legend
                # borderwidth=1      # ความหนาของเส้นขอบ
            ),
            # legend_title='Lines',
            template='plotly'
        )
        
    pio.write_image(fig, filesave+'.png')
    fig.show()


def plot_hist(data:List,filesave:str):


    # Generate random data for the histogram
    data = np.array(data)  # 1000 data points from a normal distribution

    # Create a histogram
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=100,  # Number of bins
        marker=dict(color='blue', opacity=0.7),  # Style
        name='Data Distribution'
    ))

    # Update layout
    fig.update_layout(
        # title='Histogram of Data',
        xaxis_title='Value',
        yaxis_title='Count',
        template='plotly'
    )
    pio.write_image(fig, filesave+'.png')
    # Show the histogram
    fig.show()
    

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


# dist_list = ['fdist','normal','wald','wiebull','realworld']
# idx = 3
# dist_select = dist_list[idx]

# print(f"Your selected distribution is: {dist_select}")
# flag = input("Continued Running y/n: ")
# if (flag == 'y') or (flag == 'Y'):    
#     if dist_select == 'normal':
#         folder_path = './config_sim_data/normal/'
#         filename_list = ['normalm0sd1n10000','normalm0sd4n10000']
#     if dist_select == 'fdist':
#         # normal distribution
#         folder_path = './config_sim_data/fdist/'
#         # filename_list = ['normalm0sd1n10000','normalm0sd25n10000',
#         #                  'normalm0sd25n50000','normalm0sd100n50000']
#         filename_list = ['fdistdfn5dfd10n10000','fdistdfn5dfd20n10000'] 
#     if dist_select == 'wald':    
#         # wald distribution
#         folder_path = './config_sim_data/wald/'
#         filename_list = ['waldm1sd2n10000','waldm1sd05n10000']    
#     if dist_select == 'wiebull':
#         # wiebull distribution
#         folder_path = './config_sim_data/wiebull/'
#         # filename_list = ['wiebullshape5n50000']
#         filename_list = ['wiebullshape1n10000','wiebullshape5n10000']    

def print_args(args):
    """Print arguments nicely formatted"""
    print("Arguments:")
    for k, v in args.items():
        print(f"  {k}: {v}")

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
        description='Bootstraping running results',
        epilog='Example: python script.py --source config_sim_data/normal',  # ข้อความท้าย help
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # แสดง default values
        prog='Bootstrap-Tool'
        )
    ROOT = Path(__file__).parent
    parser.add_argument("--source", type = str, default=ROOT/"config_sim_data/config_results_normal.yaml", help = 'source for loading config file results')
    opt = parser.parse_args()
    return opt

def run(source:str = "./config_sim_data/config_results_normal.yaml"):
    with open(source, 'r') as file:
        config = yaml.safe_load(file)
    folder_path = config['folder_path']
    filename_list = config['filename_list']
    res_all = []
    pop_data = []
    for filename in filename_list:
        
        # Save the result figures.
        figure_path = os.path.join(folder_path,filename)
        
        if not(ensure_directory_exists(figure_path)):
            break
        
        file_re = os.path.join(folder_path,filename+'_re.pkl') # file results
        file_pop = os.path.join(folder_path,filename) # file name of population data.
        # load all instances.
        with open(file_re, 'rb') as file:
            loaded_data = pickle.load(file)
        temp = loaded_data['result_all']    
        res_all.append(temp)    
        pop_data.append(pd.read_pickle(file_pop+'.pkl'))    
        pop_min = np.min(pop_data[-1])
        pop_max = np.max(pop_data[-1])
        pop_range = pop_max - pop_min
        print(f" Distribution: {config['dist']}: min = {pop_min:.4f} and max = {pop_max:.4f}")
        # print population distribution
        plot_hist(pop_data[-1],filesave = os.path.join(figure_path,filename))
        
        res = res_all[-1]
        ch_size = list(set([res1.chunk_size for res1 in res]))
        exp_l = []
        exp_r = []
        exp_range = []
        nlearn = []
        name_l = []
        name_r = []
        error_l = []
        error_r = []
        error_range = []
        name = []
        for size in ch_size:
            name_l.append(['min_'+res1.net_name+str(size) for res1 in res if res1.chunk_size==size])
            name_r.append(['max_'+res1.net_name+str(size) for res1 in res if res1.chunk_size==size])
            exp_l.append([res1.exp_l for res1 in res if res1.chunk_size==size])
            exp_r.append([res1.exp_r for res1 in res if res1.chunk_size==size])
            exp_range.append([res1.exp_range for res1 in res if res1.chunk_size==size])
            nlearn.append([[a+b for a,b in zip(res1.nlearnl,res1.nlearnr)] for res1 in res if res1.chunk_size==size])
            name.append([res1.net_name+str(size) for res1 in res if res1.chunk_size==size])
            error_l.append([list(map(lambda x: pop_min - x, res1.exp_l)) for res1 in res if res1.chunk_size==size])
            error_r.append([list(map(lambda x: pop_max - x, res1.exp_r)) for res1 in res if res1.chunk_size==size])
            error_range.append([list(map(lambda x: pop_range - x, res1.exp_range)) for res1 in res if res1.chunk_size==size])    
        popminmax = [pop_min,pop_max]
        for idx in range(len(exp_l)):
            plot_minmax_line(ch_size[idx],exp_l[idx], exp_r[idx], name_l[idx],name_r[idx], popminmax,
                            filesave = os.path.join(figure_path,filename+name[idx][0]))
            plot_err_line(ch_size[idx],error_range[idx], name[idx],
                        filesave = os.path.join(figure_path,filename+name[idx][0]+'error'),position = 'top-right')
            plot_nlearn_line(ch_size[idx],nlearn[idx], name[idx], 
                            filesave = os.path.join(figure_path,filename+name[idx][0]+'_n'))
    
def main(opt):
    run(**vars(opt))
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)    
    