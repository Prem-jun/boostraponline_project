import pandas as pd
import numpy as np
from typing import List, Union, Dict
from dataclasses import dataclass, field
import plotly.graph_objects as go
import plotly.io as pio
from online_bootstrap import boot_stream

@dataclass
class Res_boostrap:
    net_name: str = ''
    net:boot_stream.booststream = field(default_factory=boot_stream.booststream) 
    chunk_size: int=0
    num_chunk: int =0
    exp_l:List[float] = field(default_factory=list)
    exp_r:List[float] = field(default_factory=list)
    exp_range:List[float] =field(default_factory=list)
    nlearnl:List[float] =field(default_factory=list)
    nlearnr:List[float] =field(default_factory=list)
    
    def add_init_params(self, net:boot_stream.booststream, cum:bool=False):
        # add net_name attributes.
        self.net = net
        if self.net.online:
            if not self.net.minmax_boost:
                self.net_name = 'online' if not cum else 'online_cum'
            else:
                self.net_name = 'online_mm' if not cum else 'online_mm_cum'
        else:
            self.net_name = 'offline'
        
            
    
    def add_params(self,net:boot_stream.booststream):
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