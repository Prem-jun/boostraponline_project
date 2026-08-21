"""
Result Bootstrap Module (Version 2)
====================================

Result collector and plotting utilities for BootstrapOnline experiments.
Refactored from res_bootstrap.py to support the new BootstrapOnline class.

Class:
    ResBootstrap — Collects and stores bootstrap results per chunk.

Functions:
    plot_err_line()     — Plot error lines for left/right/range
    plot_nlearn_line()  — Plot number of learning samples
    plot_minmax_line()  — Plot min/max boundary progression
    plot_hist()         — Plot histogram of data distribution
"""

import pandas as pd
import numpy as np
from typing import List, Union, Dict
from dataclasses import dataclass, field
import plotly.graph_objects as go
import plotly.io as pio
from online_bootstrap.bootstrap_online import BootstrapOnline


@dataclass
class ResBootstrap:
    """Collects bootstrap results from each chunk iteration.

    Stores the progression of exp_l, exp_r, range, and learning counts
    across multiple chunks for later analysis and plotting.

    Attributes:
        net_name: Name identifier for the bootstrap method ('online', 'online_mm', 'offline').
        net: Reference to the BootstrapOnline engine instance.
        chunk_size: Size of each data chunk.
        num_chunk: Number of chunks processed so far.
        exp_l: History of left boundary values.
        exp_r: History of right boundary values.
        exp_range: History of range values.
        nlearnl: History of left-side learning counts.
        nlearnr: History of right-side learning counts.
    """
    net_name: str = ''
    net: BootstrapOnline = field(default_factory=BootstrapOnline)
    chunk_size: int = 0
    num_chunk: int = 0
    exp_l: List[float] = field(default_factory=list)
    exp_r: List[float] = field(default_factory=list)
    exp_range: List[float] = field(default_factory=list)
    nlearnl: List[float] = field(default_factory=list)
    nlearnr: List[float] = field(default_factory=list)

    def add_init_params(self, net: BootstrapOnline, cum: bool = False) -> None:
        """Initialize the result collector with a bootstrap engine reference.

        Automatically determines the method name based on engine configuration.

        Args:
            net: BootstrapOnline engine instance.
            cum: If True, marks this as a cumulative online method.
        """
        self.net = net
        if self.net.online:
            if not self.net.minmax_boost:
                self.net_name = 'online' if not cum else 'online_cum'
            else:
                self.net_name = 'online_mm' if not cum else 'online_mm_cum'
        else:
            self.net_name = 'offline'

    def add_params(self, net: BootstrapOnline) -> None:
        """Record current state of the bootstrap engine after processing a chunk.

        Appends exp_l, exp_r, range, and learning counts to history lists.

        Args:
            net: BootstrapOnline engine instance (after processing a chunk).
        """
        self.net = net
        if self.num_chunk == 0:
            self.chunk_size = self.net.chunk_size
        self.num_chunk += 1
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


# ====================================================================== #
#                         Plotting Functions                               #
# ====================================================================== #

def plot_err_line(ch_size: int, error_list: List, name: List, filesave: str,
                  yaxis_title: str = 'Range error',
                  xaxis_title: str = 'Number of samples',
                  color_list: List = ['brown', 'blue', 'green'],
                  position: str = 'middle-right') -> None:
    """Plot error progression lines across chunks.

    Args:
        ch_size: Size of each data chunk (used for x-axis scale).
        error_list: List of error series (one per method).
        name: List of method names for legend.
        filesave: Output file path (without extension).
        yaxis_title: Y-axis label.
        xaxis_title: X-axis label.
        color_list: Colors for each line.
        position: Legend position ('middle-right' or 'top-right').
    """
    x = [(i + 1) * ch_size for i in range(len(error_list[0]))]
    fig = go.Figure()

    for i in range(len(error_list)):
        fig.add_trace(go.Scatter(
            x=x, y=error_list[i], mode='lines+markers',
            name=name[i], line=dict(color=color_list[i])
        ))

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=_get_legend_config(position),
        template='plotly'
    )
    pio.write_image(fig, filesave + '.png')
    fig.show()


def plot_nlearn_line(ch_size: int, nlearnl: List, name: List, filesave: str,
                     yaxis_title: str = 'Numbers of samples for bootstraping',
                     xaxis_title: str = 'Number of samples',
                     color_list: List = ['brown', 'blue', 'green'],
                     position: str = 'middle-right') -> None:
    """Plot the number of learning samples per chunk.

    Args:
        ch_size: Size of each data chunk (used for x-axis scale).
        nlearnl: List of learning count series (one per method).
        name: List of method names for legend.
        filesave: Output file path (without extension).
        yaxis_title: Y-axis label.
        xaxis_title: X-axis label.
        color_list: Colors for each line.
        position: Legend position ('middle-right' or 'top-right').
    """
    x = [(i + 1) * ch_size for i in range(len(nlearnl[0]))]
    fig = go.Figure()

    for i in range(len(nlearnl) - 1):
        fig.add_trace(go.Scatter(
            x=x, y=nlearnl[i], mode='lines+markers',
            name=name[i], line=dict(color=color_list[i])
        ))

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=_get_legend_config(position),
        template='plotly'
    )
    pio.write_image(fig, filesave + '.png')
    fig.show()


def plot_minmax_line(ch_size: int, exp_l: List, exp_r: List,
                     name_l: List, name_r: List, popminmax: List,
                     filesave: str,
                     yaxis_title: str = 'Value',
                     xaxis_title: str = 'Number of samples',
                     color_list: List = ['brown', 'blue', 'green'],
                     position: str = 'middle-right') -> None:
    """Plot min/max boundary progression with population reference lines.

    Args:
        ch_size: Size of each data chunk (used for x-axis scale).
        exp_l: List of left boundary series (one per method).
        exp_r: List of right boundary series (one per method).
        name_l: Legend names for left boundary lines.
        name_r: Legend names for right boundary lines.
        popminmax: [population_min, population_max] reference values.
        filesave: Output file path (without extension).
        yaxis_title: Y-axis label.
        xaxis_title: X-axis label.
        color_list: Colors for each method.
        position: Legend position ('middle-right' or 'top-right').
    """
    x = [(i + 1) * ch_size for i in range(len(exp_l[0]))]
    fig = go.Figure()

    for i in range(len(exp_l)):
        fig.add_trace(go.Scatter(
            x=x, y=exp_l[i], mode='lines+markers',
            name=name_l[i], line=dict(color=color_list[i])
        ))
        fig.add_trace(go.Scatter(
            x=x, y=exp_r[i], mode='lines+markers',
            name=name_r[i], line=dict(color=color_list[i])
        ))

    # Add population min/max reference lines
    fig.add_trace(go.Scatter(
        x=[min(x), max(x)], y=[popminmax[0], popminmax[0]],
        mode='lines', name=f'min pop: {popminmax[0]:.4f}',
        line=dict(color='grey', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[min(x), max(x)], y=[popminmax[1], popminmax[1]],
        mode='lines', name=f'max pop: {popminmax[1]:.4f}',
        line=dict(color='grey', dash='dash')
    ))

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=_get_legend_config(position),
        template='plotly'
    )
    pio.write_image(fig, filesave + '.png')
    fig.show()


def plot_hist(data: List, filesave: str) -> None:
    """Plot histogram of data distribution.

    Args:
        data: Data values to histogram.
        filesave: Output file path (without extension).
    """
    data = np.array(data)
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data, nbinsx=100,
        marker=dict(color='blue', opacity=0.7),
        name='Data Distribution'
    ))

    fig.update_layout(
        xaxis_title='Value',
        yaxis_title='Count',
        template='plotly'
    )
    pio.write_image(fig, filesave + '.png')
    fig.show()


# ====================================================================== #
#                         Private Helpers                                  #
# ====================================================================== #

def _get_legend_config(position: str) -> dict:
    """Get plotly legend configuration for the specified position.

    Args:
        position: 'middle-right' or 'top-right'.

    Returns:
        Dict of legend configuration for plotly layout.
    """
    base = dict(
        font=dict(family="Arial", size=10, color="black"),
        xanchor="right",
        bgcolor="rgba(0,0,0,0)"
    )
    if position == 'top-right':
        base.update(yanchor="top", y=0.9, x=0.9)
    else:  # default: middle-right
        base.update(yanchor="middle", y=0.4, x=0.99)
    return base
