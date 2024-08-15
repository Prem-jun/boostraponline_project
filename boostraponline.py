#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:43:13 2024

@author: premjunsawang
"""
# from scipy import stats
import numpy as np
import pandas as pd
# from tabulate import tabulate
import typing
import copy
from scipy import stats as st
from scipy.stats import exponweib
from scipy.stats import wald
from scipy.stats import exponpow
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import expon
from scipy.stats import powerlaw
from scipy.stats import lognorm
# from scipy.stats import cauchy
from scipy.stats import chi2
from scipy.stats import uniform
# from scipy.stats import kurtosis
# from scipy.stats import skew
from scipy.stats import weibull_min
from scipy.stats import weibull_max
import math
# from time import sleep
# from datetime import date
# from datetime import time 
from datetime import datetime
import matplotlib.pyplot as plt


class stat_dist:
    def gamma_percent_area_in_each_std(a=None):
        if a is None:
            a = 1.99
        mean, var, skew, kurt = gamma.stats(a, moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = gamma.cdf(mean, a, loc_value, scale_value )
        cdf_area_at_rstd1 = gamma.cdf(rstd1, a, loc_value, scale_value)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = gamma.cdf(rstd1, a, loc_value, scale_value)
        cdf_area_at_rstd2 = gamma.cdf(rstd2, a, loc_value, scale_value)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = gamma.cdf(rstd2, a, loc_value, scale_value)
        cdf_area_at_rstd3 = gamma.cdf(rstd3, a, loc_value, scale_value)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = gamma.cdf(rstd3, a, loc_value, scale_value)
        cdf_area_at_rstd4 = gamma.cdf(rstd4, a, loc_value, scale_value)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = gamma.cdf(lstd1, a, loc_value, scale_value)
        cdf_area_at_mean = gamma.cdf(mean, a, loc_value, scale_value)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = gamma.cdf(lstd2, a, loc_value, scale_value)
        cdf_area_at_lstd1 = gamma.cdf(lstd1, a, loc_value, scale_value)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = gamma.cdf(lstd3, a, loc_value, scale_value)
        cdf_area_at_lstd2 = gamma.cdf(lstd2, a, loc_value, scale_value)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = gamma.cdf(lstd4, a, loc_value, scale_value)
        cdf_area_at_lstd3 = gamma.cdf(lstd3, a, loc_value, scale_value)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #----------------------------------------------------------------------------
    def norm_percent_area_in_each_std():
        
        mean, var, skew, kurt = norm.stats(moments='mvsk')
        std = math.sqrt(var)
    
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = norm.cdf(mean, loc_value, scale_value )
        cdf_area_at_rstd1 = norm.cdf(rstd1, loc_value, scale_value)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = norm.cdf(rstd1, loc_value, scale_value)
        cdf_area_at_rstd2 = norm.cdf(rstd2, loc_value, scale_value)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = norm.cdf(rstd2, loc_value, scale_value)
        cdf_area_at_rstd3 = norm.cdf(rstd3, loc_value, scale_value)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = norm.cdf(rstd3, loc_value, scale_value)
        cdf_area_at_rstd4 = norm.cdf(rstd4, loc_value, scale_value)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = norm.cdf(lstd1, loc_value, scale_value)
        cdf_area_at_mean = norm.cdf(mean, loc_value, scale_value)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = norm.cdf(lstd2, loc_value, scale_value)
        cdf_area_at_lstd1 = norm.cdf(lstd1, loc_value, scale_value)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = norm.cdf(lstd3, loc_value, scale_value)
        cdf_area_at_lstd2 = norm.cdf(lstd2, loc_value, scale_value)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = norm.cdf(lstd4, loc_value, scale_value)
        cdf_area_at_lstd3 = norm.cdf(lstd3, loc_value, scale_value)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #----------------------------------------------------------------------------
    def exponpow_percent_area_in_each_std(b = None):
        if b is None:
            b = 2.7
        mean, var, skew, kurt = exponpow.stats(b, moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = exponpow.cdf(mean, b, loc_value, scale_value )
        cdf_area_at_rstd1 = exponpow.cdf(rstd1, b, loc_value, scale_value)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = exponpow.cdf(rstd1, b, loc_value, scale_value)
        cdf_area_at_rstd2 = exponpow.cdf(rstd2, b, loc_value, scale_value)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = exponpow.cdf(rstd2, b, loc_value, scale_value)
        cdf_area_at_rstd3 = exponpow.cdf(rstd3, b, loc_value, scale_value)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = exponpow.cdf(rstd3, b, loc_value, scale_value)
        cdf_area_at_rstd4 = exponpow.cdf(rstd4, b, loc_value, scale_value)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = exponpow.cdf(lstd1, b, loc_value, scale_value)
        cdf_area_at_mean = exponpow.cdf(mean, b, loc_value, scale_value)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = exponpow.cdf(lstd2, b, loc_value, scale_value)
        cdf_area_at_lstd1 = exponpow.cdf(lstd1, b, loc_value, scale_value)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = exponpow.cdf(lstd3, b, loc_value, scale_value)
        cdf_area_at_lstd2 = exponpow.cdf(lstd2, b, loc_value, scale_value)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = exponpow.cdf(lstd4, b, loc_value, scale_value)
        cdf_area_at_lstd3 = exponpow.cdf(lstd3, b, loc_value, scale_value)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #----------------------------------------------------------------------------
    def wald_percent_area_in_each_std():
        
        mean, var, skew, kurt = wald.stats(moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = wald.cdf(mean, loc_value, scale_value )
        cdf_area_at_rstd1 = wald.cdf(rstd1, loc_value, scale_value)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = wald.cdf(rstd1, loc_value, scale_value)
        cdf_area_at_rstd2 = wald.cdf(rstd2, loc_value, scale_value)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = wald.cdf(rstd2, loc_value, scale_value)
        cdf_area_at_rstd3 = wald.cdf(rstd3, loc_value, scale_value)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = wald.cdf(rstd3, loc_value, scale_value)
        cdf_area_at_rstd4 = wald.cdf(rstd4, loc_value, scale_value)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = wald.cdf(lstd1, loc_value, scale_value)
        cdf_area_at_mean = wald.cdf(mean, loc_value, scale_value)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = wald.cdf(lstd2, loc_value, scale_value)
        cdf_area_at_lstd1 = wald.cdf(lstd1, loc_value, scale_value)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = wald.cdf(lstd3, loc_value, scale_value)
        cdf_area_at_lstd2 = wald.cdf(lstd2, loc_value, scale_value)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = wald.cdf(lstd4, loc_value, scale_value)
        cdf_area_at_lstd3 = wald.cdf(lstd3, loc_value, scale_value)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #----------------------------------------------------------------------------
    def exponweib_percent_area_in_each_std(a = None,c = None):
        if a is None:   
            a = 2.89
        if c is None:
            c = 1.95
        mean, var, skew, kurt = exponweib.stats(a, c, moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = exponweib.cdf(mean, a, c, 0, 1)
        cdf_area_at_rstd1 = exponweib.cdf(rstd1, a, c, 0, 1)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = exponweib.cdf(rstd1, a, c, 0, 1)
        cdf_area_at_rstd2 = exponweib.cdf(rstd2, a, c, 0, 1)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = exponweib.cdf(rstd2, a, c, 0, 1)
        cdf_area_at_rstd3 = exponweib.cdf(rstd3, a, c, 0, 1)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = exponweib.cdf(rstd3, a, c, 0, 1)
        cdf_area_at_rstd4 = exponweib.cdf(rstd4, a, c, 0, 1)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = exponweib.cdf(lstd1, a, c, 0, 1)
        cdf_area_at_mean = exponweib.cdf(mean, a, c, 0, 1)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = exponweib.cdf(lstd2, a, c, 0, 1)
        cdf_area_at_lstd1 = exponweib.cdf(lstd1, a, c, 0, 1)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = exponweib.cdf(lstd3, a, c, 0, 1)
        cdf_area_at_lstd2 = exponweib.cdf(lstd2, a, c, 0, 1)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = exponweib.cdf(lstd4, a, c, 0, 1)
        cdf_area_at_lstd3 = exponweib.cdf(lstd3, a, c, 0, 1)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #------------------------------------------------------------------------------
    def rayleigh_percent_area_in_each_std():
        
        mean, var, skew, kurt = rayleigh.stats(moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = rayleigh.cdf(mean, loc_value, scale_value )
        cdf_area_at_rstd1 = rayleigh.cdf(rstd1, loc_value, scale_value)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = rayleigh.cdf(rstd1, loc_value, scale_value)
        cdf_area_at_rstd2 = rayleigh.cdf(rstd2, loc_value, scale_value)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = rayleigh.cdf(rstd2, loc_value, scale_value)
        cdf_area_at_rstd3 = rayleigh.cdf(rstd3, loc_value, scale_value)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = rayleigh.cdf(rstd3, loc_value, scale_value)
        cdf_area_at_rstd4 = rayleigh.cdf(rstd4, loc_value, scale_value)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = exponweib.cdf(lstd1, loc_value, scale_value)
        cdf_area_at_mean = exponweib.cdf(mean, loc_value, scale_value)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = exponweib.cdf(lstd2, loc_value, scale_value)
        cdf_area_at_lstd1 = exponweib.cdf(lstd1, loc_value, scale_value)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = exponweib.cdf(lstd3, loc_value, scale_value)
        cdf_area_at_lstd2 = exponweib.cdf(lstd2, loc_value, scale_value)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = exponweib.cdf(lstd4, loc_value, scale_value)
        cdf_area_at_lstd3 = exponweib.cdf(lstd3, loc_value, scale_value)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #---------------------------------------------------------------------------
    def powerlaw_percent_area_in_each_std():
        
        a = 0.659
        mean, var, skew, kurt = powerlaw.stats(a, moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = powerlaw.cdf(mean, a, loc_value, scale_value)
        cdf_area_at_rstd1 = powerlaw.cdf(rstd1, a, loc_value, scale_value)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = powerlaw.cdf(rstd1, a, loc_value, scale_value)
        cdf_area_at_rstd2 = powerlaw.cdf(rstd2, a, loc_value, scale_value)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = powerlaw.cdf(rstd2, a, loc_value, scale_value)
        cdf_area_at_rstd3 = powerlaw.cdf(rstd3, a, loc_value, scale_value)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = powerlaw.cdf(rstd3, a, loc_value, scale_value)
        cdf_area_at_rstd4 = powerlaw.cdf(rstd4, a, loc_value, scale_value)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = powerlaw.cdf(lstd1, a, loc_value, scale_value)
        cdf_area_at_mean = powerlaw.cdf(mean, a, loc_value, scale_value)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = powerlaw.cdf(lstd2, a, loc_value, scale_value)
        cdf_area_at_lstd1 = powerlaw.cdf(lstd1, a, loc_value, scale_value)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = powerlaw.cdf(lstd3, a, loc_value, scale_value)
        cdf_area_at_lstd2 = powerlaw.cdf(lstd2, a, loc_value, scale_value)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = powerlaw.cdf(lstd4, a, loc_value, scale_value)
        cdf_area_at_lstd3 = powerlaw.cdf(lstd3, a, loc_value, scale_value)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #-------------------------------------------------------------------------
    def expon_percent_area_in_each_std():
        
        mean, var, skew, kurt = expon.stats(moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = expon.cdf(mean, loc_value, scale_value )
        cdf_area_at_rstd1 = expon.cdf(rstd1, loc_value, scale_value)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = expon.cdf(rstd1, loc_value, scale_value)
        cdf_area_at_rstd2 = expon.cdf(rstd2, loc_value, scale_value)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = expon.cdf(rstd2, loc_value, scale_value)
        cdf_area_at_rstd3 = expon.cdf(rstd3, loc_value, scale_value)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = expon.cdf(rstd3, loc_value, scale_value)
        cdf_area_at_rstd4 = expon.cdf(rstd4, loc_value, scale_value)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = expon.cdf(lstd1, loc_value, scale_value)
        cdf_area_at_mean = expon.cdf(mean, loc_value, scale_value)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = expon.cdf(lstd2, loc_value, scale_value)
        cdf_area_at_lstd1 = expon.cdf(lstd1, loc_value, scale_value)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = expon.cdf(lstd3, loc_value, scale_value)
        cdf_area_at_lstd2 = expon.cdf(lstd2, loc_value, scale_value)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = expon.cdf(lstd4, loc_value, scale_value)
        cdf_area_at_lstd3 = expon.cdf(lstd3, loc_value, scale_value)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #-------------------------------------------------------------------------
    def uniform_percent_area_in_each_std():
        
        mean, var, skew, kurt = uniform.stats(moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = uniform.cdf(mean, loc_value, scale_value )
        cdf_area_at_rstd1 = uniform.cdf(rstd1, loc_value, scale_value)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = uniform.cdf(rstd1, loc_value, scale_value)
        cdf_area_at_rstd2 = uniform.cdf(rstd2, loc_value, scale_value)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = uniform.cdf(rstd2, loc_value, scale_value)
        cdf_area_at_rstd3 = uniform.cdf(rstd3, loc_value, scale_value)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = uniform.cdf(rstd3, loc_value, scale_value)
        cdf_area_at_rstd4 = uniform.cdf(rstd4, loc_value, scale_value)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = uniform.cdf(lstd1, loc_value, scale_value)
        cdf_area_at_mean = uniform.cdf(mean, loc_value, scale_value)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = uniform.cdf(lstd2, loc_value, scale_value)
        cdf_area_at_lstd1 = uniform.cdf(lstd1, loc_value, scale_value)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = uniform.cdf(lstd3, loc_value, scale_value)
        cdf_area_at_lstd2 = uniform.cdf(lstd2, loc_value, scale_value)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = uniform.cdf(lstd4, loc_value, scale_value)
        cdf_area_at_lstd3 = uniform.cdf(lstd3, loc_value, scale_value)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #--------------------------------------------------------------------------
    def lognorm_percent_area_in_each_std():
        
        s = 0.954
        mean, var, skew, kurt = lognorm.stats(s, moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = lognorm.cdf(mean, s, loc_value, scale_value )
        cdf_area_at_rstd1 = lognorm.cdf(rstd1, s, loc_value, scale_value)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = lognorm.cdf(rstd1, s, loc_value, scale_value)
        cdf_area_at_rstd2 = lognorm.cdf(rstd2, s, loc_value, scale_value)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = lognorm.cdf(rstd2, s, loc_value, scale_value)
        cdf_area_at_rstd3 = lognorm.cdf(rstd3, s, loc_value, scale_value)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = lognorm.cdf(rstd3, s, loc_value, scale_value)
        cdf_area_at_rstd4 = lognorm.cdf(rstd4, s, loc_value, scale_value)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = lognorm.cdf(lstd1, s, loc_value, scale_value)
        cdf_area_at_mean = lognorm.cdf(mean, s, loc_value, scale_value)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = lognorm.cdf(lstd2, s, loc_value, scale_value)
        cdf_area_at_lstd1 = lognorm.cdf(lstd1, s, loc_value, scale_value)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = lognorm.cdf(lstd3, s, loc_value, scale_value)
        cdf_area_at_lstd2 = lognorm.cdf(lstd2, s, loc_value, scale_value)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = lognorm.cdf(lstd4, s, loc_value, scale_value)
        cdf_area_at_lstd3 = lognorm.cdf(lstd3, s, loc_value, scale_value)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #--------------------------------------------------------------------
    def chi2_percent_area_in_each_std():
        df = 55
        mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = chi2.cdf(mean, df, 0, 1 )
        cdf_area_at_rstd1 = chi2.cdf(rstd1, df, 0, 1)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = chi2.cdf(rstd1, df, 0, 1)
        cdf_area_at_rstd2 = chi2.cdf(rstd2, df, 0, 1)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = chi2.cdf(rstd2, df, 0, 1)
        cdf_area_at_rstd3 = chi2.cdf(rstd3, df, 0, 1)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = chi2.cdf(rstd3, df, 0, 1)
        cdf_area_at_rstd4 = chi2.cdf(rstd4, df, 0, 1)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = chi2.cdf(lstd1, df, 0, 1)
        cdf_area_at_mean = chi2.cdf(mean, df, 0, 1)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = chi2.cdf(lstd2, df, 0, 1)
        cdf_area_at_lstd1 = chi2.cdf(lstd1, df, 0, 1)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = chi2.cdf(lstd3, df, 0, 1)
        cdf_area_at_lstd2 = chi2.cdf(lstd2, df, 0, 1)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = chi2.cdf(lstd4, df, 0, 1)
        cdf_area_at_lstd3 = chi2.cdf(lstd3, df, 0, 1)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #-----------------------------------------------------------------------
    def weibull_min_percent_area_in_each_std():
        
        c = 1.79
        mean, var, skew, kurt = weibull_min.stats(c, moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = weibull_min.cdf(mean, c, 0, 1 )
        cdf_area_at_rstd1 = weibull_min.cdf(rstd1, c, 0, 1)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = weibull_min.cdf(rstd1, c, 0, 1)
        cdf_area_at_rstd2 = weibull_min.cdf(rstd2, c, 0, 1)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = weibull_min.cdf(rstd2, c, 0, 1)
        cdf_area_at_rstd3 = weibull_min.cdf(rstd3, c, 0, 1)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = weibull_min.cdf(rstd3, c, 0, 1)
        cdf_area_at_rstd4 = weibull_min.cdf(rstd4, c, 0, 1)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = chi2.cdf(lstd1, c, 0, 1)
        cdf_area_at_mean = chi2.cdf(mean, c, 0, 1)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = weibull_min.cdf(lstd2, c, 0, 1)
        cdf_area_at_lstd1 = weibull_min.cdf(lstd1, c, 0, 1)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = weibull_min.cdf(lstd3, c, 0, 1)
        cdf_area_at_lstd2 = weibull_min.cdf(lstd2, c, 0, 1)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = weibull_min.cdf(lstd4, c, 0, 1)
        cdf_area_at_lstd3 = weibull_min.cdf(lstd3, c, 0, 1)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #----------------------------------------------------------------------
    def weibull_max_percent_area_in_each_std(c = None):
        if c is None:
            c = 2.87
        mean, var, skew, kurt = weibull_max.stats(c, moments='mvsk')
        std = math.sqrt(var)
        
        loc_value = mean
        scale_value = std
        
        lstd1 = mean - std
        lstd2 = mean - 2*std
        lstd3 = mean - 3*std
        lstd4 = mean - 4*std
        rstd1 = mean + std
        rstd2 = mean + 2*std
        rstd3 = mean + 3*std
        rstd4 = mean + 4*std
        
        cdf_area_at_mean = weibull_max.cdf(mean, c, 0, 1 )
        cdf_area_at_rstd1 = weibull_max.cdf(rstd1, c, 0, 1)
        area_between_mean_rstd1 = cdf_area_at_rstd1 - cdf_area_at_mean
        
        cdf_area_at_rstd1 = weibull_max.cdf(rstd1, c, 0, 1)
        cdf_area_at_rstd2 = weibull_max.cdf(rstd2, c, 0, 1)
        area_between_rstd1_rstd2 = cdf_area_at_rstd2 - cdf_area_at_rstd1
    
        cdf_area_at_rstd2 = weibull_max.cdf(rstd2, c, 0, 1)
        cdf_area_at_rstd3 = weibull_max.cdf(rstd3, c, 0, 1)
        area_between_rstd2_rstd3 = cdf_area_at_rstd3 - cdf_area_at_rstd2
    
        cdf_area_at_rstd3 = chi2.cdf(rstd3, c, 0, 1)
        cdf_area_at_rstd4 = chi2.cdf(rstd4, c, 0, 1)
        area_between_rstd3_rstd4 = cdf_area_at_rstd4 - cdf_area_at_rstd3
     #--------------------------------------------------------------------------   
    
        cdf_area_at_lstd1 = weibull_max.cdf(lstd1, c, 0, 1)
        cdf_area_at_mean = weibull_max.cdf(mean, c, 0, 1)
        area_between_lstd1_mean = cdf_area_at_mean - cdf_area_at_lstd1
        
        cdf_area_at_lstd2 = weibull_max.cdf(lstd2, c, 0, 1)
        cdf_area_at_lstd1 = weibull_max.cdf(lstd1, c, 0, 1)
        area_between_lstd2_lstd1 = cdf_area_at_lstd1 - cdf_area_at_lstd2
    
        cdf_area_at_lstd3 = weibull_max.cdf(lstd3, c, 0, 1)
        cdf_area_at_lstd2 = weibull_max.cdf(lstd2, c, 0, 1)
        area_between_lstd3_lstd2 = cdf_area_at_lstd2 - cdf_area_at_lstd3
    
        cdf_area_at_lstd4 = weibull_max.cdf(lstd4, c, 0, 1)
        cdf_area_at_lstd3 = weibull_max.cdf(lstd3, c, 0, 1)
        area_between_lstd4_lstd3 = cdf_area_at_lstd3 - cdf_area_at_lstd4
    
        total_area = area_between_lstd4_lstd3 + area_between_lstd3_lstd2 + \
                     area_between_lstd2_lstd1 + area_between_lstd1_mean  + \
                     area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                     area_between_rstd2_rstd3 + area_between_rstd3_rstd4
    
        percent_data_rstd1 = 100*(area_between_mean_rstd1 / total_area)
        percent_data_rstd2 = 100*(area_between_rstd1_rstd2 / total_area)
        percent_data_rstd3 = 100*(area_between_rstd2_rstd3 / total_area)
        percent_data_rstd4 = 100*(area_between_rstd3_rstd4 / total_area)
    
        percent_data_lstd1 = 100*(area_between_lstd1_mean / total_area)
        percent_data_lstd2 = 100*(area_between_lstd2_lstd1 / total_area)
        percent_data_lstd3 = 100*(area_between_lstd3_lstd2 / total_area)
        percent_data_lstd4 = 100*(area_between_lstd4_lstd3 / total_area)
    
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
               percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
               percent_data_rstd3, percent_data_rstd4
    
    #def the_hist_normal():
#______________________________________________________________________________

class managefile:
    def get_file_nmae_from_hour_minute_second():
        present_time = datetime.now()
        file_name = str(present_time.hour) + str(present_time.minute) + \
                    str(present_time.second)
    
        return file_name

#---------------------------------------------------------------------------
    def read_data_from_file(input_file_name):
        class_index_name_list = []
        min_left_end_list = []
        max_right_end_list = []
        min_data_left_end_list = []
        max_data_right_end_list = []
    
        with open(input_file_name,'r') as data_file:
            for line in data_file:
                data = line.split()
                size = len(data)
                if data[0] != '\n':
                    x = float(data[0])
                    class_index_name_list.append(x)
                    x = float(data[1])
                    min_left_end_list.append(x)
                    x = float(data[2])
                    max_right_end_list.append(x)
                    x = float(data[3])
                    min_data_left_end_list.append(x)
                    x = float(data[4])
                    max_data_right_end_list.append(x)
    
                    
        return class_index_name_list, min_left_end_list, max_right_end_list, \
               min_data_left_end_list, max_data_right_end_list
    
    #----------------------------------------------------------------------------
    def show_exapnd_result():
        input_file_name = "expand-stat.txt"
        class_index_name_list, min_left_end_list, max_right_end_list, \
               min_data_left_end_list, max_data_right_end_list = \
                                                managefile.read_data_from_file(input_file_name)
    
        figure, axis = plt.subplots(2)
    
        axis[1].plot(class_index_name_list, min_data_left_end_list, 'bo', label = \
                 'min_data_left_end_list')
        axis[1].plot(class_index_name_list, min_left_end_list, 'ro', label = \
                 'expand_min_left_end_list')
        
        axis[1].legend()
        axis[1].set_title('expand left end')
        axis[1].set_xlabel('class index name')
        axis[1].set_ylabel('left end value')
    
        axis[0].plot(class_index_name_list, max_right_end_list, 'co',label = \
                 'expand_max_right_end_list')
        axis[0].plot(class_index_name_list, max_data_right_end_list, 'mo', label = \
                'max_data_right_end_list')
        
        axis[0].legend()
        axis[0].set_title('expand right end')
        axis[0].set_xlabel('class index name')
        axis[0].set_ylabel('right end value')
    
        figure.tight_layout() #----- add space between two subplots ------
        file_name = managefile.get_file_nmae_from_hour_minute_second()
        file_name = "./results/" + "expand-bound-" + str(file_name) + ".png"
        plt.savefig(file_name)
        plt.show()
class boostrap_v1:
    def bootstrap_online(input_data: list, end_side: str,
                         number_bootstrap_iteration: int = None,
                         minmax_boost: bool = None, prob: bool = None) -> float:
        
        if minmax_boost is None:
           minmax_boost = False # Set default value of minmax_boost.
        if prob is None:
            prob = False # Set default value of boostrap with probability.
        data_set = copy.deepcopy(input_data)
        if number_bootstrap_iteration is None:
            number_bootstrap_iteration = 600
    
        bootstrap_means = np.zeros(number_bootstrap_iteration)
        bootstrap_std = []
        # bootstrap_max_diff_dist_std = []
    
        size_data_set = len(data_set)
        previous_bootstrap_mean = 0
        
        bootstrap_sample_list = []
        if minmax_boost:
            bootstrap_sample_min_list = []
            bootstrap_sample_max_list = []
        
        # Setting boostrap size
        size_boost = size_data_set
        # if size_data_set < 30:
        #     size_boost = size_data_set
        # else:
        #     size_boost = 30
            
        # Perform bootstrap sampling
        if prob:
            dist = abs(data_set-np.mean(data_set))
            idist = 1/dist
            pdist = idist/np.sum(idist)
        for i in range(number_bootstrap_iteration):
     
            if prob:
                
                # bootstrap_sample = [ np.random.choice(data_set, 1,replace=True,p=pdist) for i in range(size_boost)]
                bootstrap_sample = list(np.random.choice(data_set, size_boost,replace=True,p=pdist))
            else:
                # bootstrap_sample = [ np.random.choice(data_set, 1,replace=True) for i in range(size_boost)]
                bootstrap_sample = list(np.random.choice(data_set, size_boost,replace=True))
            
            
            # bootstrap_sample = list(np.random.choice(data_set, size_data_set, \
            #                                           replace = with_replacement))
            # bootstrap_sample = list(random.sample)
            bootstrap_sample_list.append(bootstrap_sample)
            if minmax_boost:
                bootstrap_sample_min_list.append(min(bootstrap_sample))
                bootstrap_sample_max_list.append(max(bootstrap_sample))
            present_bootstrap_mean = (np.mean(bootstrap_sample) +
                                      previous_bootstrap_mean)/2
            previous_bootstrap_mean = present_bootstrap_mean
            bootstrap_means[i] = present_bootstrap_mean
    
        estimated_mean = np.mean(bootstrap_means)
        # estimated_std_of_mean = boostrap_v1.stdev(bootstrap_means,estimated_mean)
        if prob:
          input_mean = sum(np.array(pdist)*np.array(input_data))
        else:
            input_mean = np.mean(input_data) # Aj Chid mean
        
        # different_input_mean_bootstrap_mean = abs(boostrap_v1.mean(input_data) - estimated_mean)
        different_input_mean_bootstrap_mean = abs(input_mean - estimated_mean)
        
        for i in range(number_bootstrap_iteration):
            data = bootstrap_sample_list[i]
            variance = [(k-estimated_mean)**2 for k in data]
            std = math.sqrt(sum(variance)/(size_boost-1))
            # std = 0
            # for j in range(size_boost):
            #     std = std + (data[j] - estimated_mean)**2
            # std = math.sqrt(std/(size_boost-1))
            # # diff_dist_right_std = max(data) - (std + estimated_mean)
            # # diff_dist_left_std = (estimated_mean - std) - min(data)
            # # max_std = (diff_dist_right_std + diff_dist_left_std)/2
            bootstrap_std.append(std)
            # bootstrap_max_diff_dist_std.append(max_std)
    
        estimated_std = np.mean(bootstrap_std)
        # estimated_std_of_std = boostrap_v1.stdev(bootstrap_std,estimated_std)
        different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data,input_mean) - estimated_std)
        # different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data,) - estimated_std)
    
        # estimated_diff_dist_std = np.mean(bootstrap_max_diff_dist_std)
        # estimated_std_of_diff_dist_std = np.std(bootstrap_max_diff_dist_std)
    
        
        # input_std = boostrap_v1.stdev(input_data) No use in the below lines.
    
        # print("=====================================================================")
        # print(">>>>> in Bootstrap")
        # print("\n----- ", end_side)
        # print("left min:", min(data_set))
        # print("right max:", max(data_set))
        # print("\ninput mean:", input_mean)
        # print("estimated_mean:", estimated_mean)
        # print("estimated_std_of_mean:", estimated_std_of_mean)
        # print(" ")
        # print("input std:", input_std)
        # print("estimated_std:", estimated_std)
        # print("estimated_std_of_std:", estimated_std_of_std)
        # print(" ")
        # print("estimated_diff_dist_std:", estimated_diff_dist_std)
        # print("estimated_std_of_diff_dist_std:", estimated_std_of_diff_dist_std)
    
        # input_mean = np.mean(input_data)
    
        if end_side == "left":
            if minmax_boost is True:
                if prob:
                    dist_min = abs(bootstrap_sample_min_list-np.mean(bootstrap_sample_min_list))
                    idist_min = 1/dist_min
                    pdist_min = idist_min/np.sum(idist_min)
                    left_min_1 = sum(np.array(pdist_min)*np.array(bootstrap_sample_min_list))
                    left_min = left_min_1
                else:
                    left_min = np.mean(bootstrap_sample_min_list)
                # left_min = np.mean(bootstrap_sample_min_list)#-np.std(bootstrap_sample_min_list)
                # left_min = st.mode(np.array(bootstrap_sample_min_list))[0][0]
            else:
                left_min = min(data_set)
            if estimated_mean < input_mean:
                # left_min = min(data_set)
                final_estimated_std = left_min - different_input_mean_bootstrap_mean
    #            final_estimated_std = left_min - estimated_std_of_diff_dist_std
                print("\n>>>>> Bootstrap in left end: estimated_mean < input_mean")
                print("left min:", min(data_set), " final_estimated_std:", \
                      final_estimated_std)
                
            if input_mean < estimated_mean:
                
                # left_min = min(data_set)
    #            final_estimated_std = left_min - estimated_std_of_std
                final_estimated_std = left_min - different_input_std_bootstrap_std
                print("\n>>>>> Bootstrap in left end: input_mean < estimated_mean")
                print("left min:", min(data_set), " final_estimated_std:", \
                      final_estimated_std)
        
        if end_side == "right":
            if minmax_boost is True:
                if prob:
                    dist_max = abs(bootstrap_sample_max_list-np.mean(bootstrap_sample_max_list))
                    idist_max = 1/dist_max
                    pdist_max = idist_max/np.sum(idist_max)
                    right_max_1 = sum(np.array(pdist_max)*np.array(bootstrap_sample_max_list))
                    right_max = right_max_1
                else:
                    right_max = np.mean(bootstrap_sample_max_list)
                # right_max = np.mean(bootstrap_sample_max_list)#+np.std(bootstrap_sample_max_list)
                # right_max = st.mode(np.array(bootstrap_sample_max_list))[0][0]
            else:
                right_max = max(data_set)
            if estimated_mean < input_mean:
                # right_max = max(data_set)
    #            final_estimated_std = right_max + estimated_std_of_std
                final_estimated_std = right_max + different_input_std_bootstrap_std
                print("\n>>>>> Bootstrap in right end: estimated_mean < input_mean")
                print("right max:", max(data_set), " final_estimated_std:", \
                      final_estimated_std)
                                      
            if input_mean < estimated_mean:
                # right_max = max(data_set)
                final_estimated_std = right_max + different_input_mean_bootstrap_mean            
                print("\n>>>>> Bootstrap in right end: input_mean < estimated_mean")
                print("right max:", max(data_set), " final_estimated_std:", \
                      final_estimated_std)
    
        print(" ")
        print("final_estimated_std:", final_estimated_std)
    
        return final_estimated_std    
    #============================================================================
    def estimate_width_1D(input_data: list, max_value: float, z_score:float) -> float:
        sample = copy.deepcopy(input_data)
        sample = np.array(sample)
        size = len(sample)
        estimate_width = abs(((max_value)/(1-z_score*(math.sqrt(2/(size-1)))) + \
                                (max_value)/(1+z_score*(math.sqrt(2/(size-1)))) )/2)
    
        return estimate_width
    #-------------------------------------------------------------------
    def mean(data: list):
        list_size = len(data)
        avg = 0
        for i in range(list_size):
           avg = avg + data[i]
        avg = avg/list_size
        return avg
    #-------------------------------------------------------------------
    def stdev(data: list,avg = None) -> float:
        list_size = len(data)
        if avg is None:
            avg = 0
            for i in range(list_size):
                avg = avg + data[i]
            avg = avg/list_size
        variance = [(i-avg)**2 for i in data]
        std = math.sqrt(sum(variance)/(list_size-1))
        # std = 0
        # for i in range(list_size):
        #     std = std + (data[i] - avg)**2
        # std = math.sqrt(std/(list_size-1))
        return std
            
    #-------------------------------------------------------------------
    def recursive_std(prev_size, present_size, prev_std, present_data):
        total_size = prev_size + present_size
        present_avg = 0
        for i in range(present_size):
           present_avg = present_avg + present_data[i]
        present_avg = present_avg/present_size
        present_var = 0
        for i in range(present_size):
            present_var = present_var + (present_data[i] - present_avg)**2
        present_var = present_var*(present_size/total_size)
        prev_var = (prev_std*prev_std)/(prev_size/total_size)
        new_var = math.sqrt(prev_var + present_var)
    
        return new_var
    
    #---------------------------------------------------------------------
    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha = 'center')
    
    #---------------------------------------------------------------------------
    def plot_histogram(slstd4,slstd3,slstd2,slstd1,srstd1,srstd2,srstd3, srstd4, \
                       lstd4,lstd3,lstd2,lstd1,rstd1,rstd2,rstd3, rstd4, \
                              min_expand, max_expand, chunk_no = 1, \
                             expand = 0):
        
        data_interval_name =['lstd4','lstd3', 'lstd2', 'lstd1', 'rstd1', 'rstd2', \
                             'rstd3', 'rstd4']
        standard_interval_name =['slstd4','slstd3', 'slstd2', 'slstd1', 'srstd1', 'srstd2', \
                             'srstd3', 'srstd4']
    
        width = (max_expand - min_expand)/8
        # print("width:", width)
        a = int(min_expand + 0*width)
        b = int(min_expand + 1*width)
        interval1 = "[" + str(a) + " - " + str(b) + "]"
    
        a = int(min_expand + 1*width)
        b = int(min_expand + 2*width)
        interval2 = "[" + str(a) + " - " + str(b) + "]"
    
        a = int(min_expand + 2*width)
        b = int(min_expand + 3*width)
        interval3 = "[" + str(a) + " - " + str(b) + "]"
    
        a = int(min_expand + 3*width)
        b = int(min_expand + 4*width)
        interval4 = "[" + str(a) + " - " + str(b) + "]"
    
        a = int(min_expand + 4*width)
        b = int(min_expand + 5*width)
        interval5 = "[" + str(a) + " - " + str(b) + "]"
    
        a = int(min_expand + 5*width)
        b = int(min_expand + 6*width)
        interval6 = "[" + str(a) + " - " + str(b) + "]"
    
        a = int(min_expand + 6*width)
        b = int(min_expand + 7*width)
        interval7 = "[" + str(a) + " - " + str(b) + "]"
    
        a = int(min_expand + 7*width)
        b = int(min_expand + 8*width)
        interval8 = "[" + str(a) + " - " + str(b) + "]"
        
        interval_value = [interval1, interval2, interval3, interval4, interval5, interval6, \
                interval7, interval8]
        
        std_hist = []
        std_hist.append(slstd4)
        std_hist.append(slstd3)
        std_hist.append(slstd2)
        std_hist.append(slstd1)
        std_hist.append(srstd1)
        std_hist.append(srstd2)
        std_hist.append(srstd3)
        std_hist.append(srstd4)
    
        std_histogram = []
        std_histogram.append(lstd4)
        std_histogram.append(lstd3)
        std_histogram.append(lstd2)
        std_histogram.append(lstd1)
        std_histogram.append(rstd1)
        std_histogram.append(rstd2)
        std_histogram.append(rstd3)
        std_histogram.append(rstd4)
    
        plt.text(0, lstd4, data_interval_name[0], ha = 'center', fontsize = 7, \
                 color = 'blue')
        plt.text(7, rstd4, data_interval_name[7], ha = 'center', fontsize = 7, \
                 color = 'blue')
    
        plt.text(0, slstd4, standard_interval_name[0], ha = 'center', fontsize = 7, \
                 color = 'blue')
        plt.text(7, srstd4, standard_interval_name[7], ha = 'center', fontsize = 7, \
                 color = 'blue')
        
        plt.xticks(rotation = 25,fontsize = 7, color = 'blue')
        title = "chunk: " + str(chunk_no) + " expand: " + \
                str(expand)
        plt.title(title)
        plt.bar(interval_value, std_hist, alpha = 0.1, color = 'red')    
        plt.bar(interval_value, std_histogram, alpha = 0.07, color = 'blue')
        plt.show()
    
    #---------------------------------------------------------------------------
    def distribution_percent(distribution_name: str) -> float:
        if distribution_name == "chi2":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = stat_dist.chi2_percent_area_in_each_std()
    
        if distribution_name == "powerlaw":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
                                                stat_dist.powerlaw_percent_area_in_each_std()
    
        if distribution_name == "expon":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = stat_dist.expon_percent_area_in_each_std() 
    
        if distribution_name == "uniform":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
                                                stat_dist.uniform_percent_area_in_each_std()
    
        if distribution_name == "cauchy":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = stat_dist.cauchy_percent_area_in_each_std()
    
        if distribution_name == "lognorm":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
                                                  stat_dist.lognorm_percent_area_in_each_std()
    
        if distribution_name == "rayleigh":   
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
                                                 stat_dist.rayleigh_percent_area_in_each_std()
    
        if distribution_name == "exponpow":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
                                                  stat_dist.exponpow_percent_area_in_each_std()
    
        if distribution_name == "norm":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = stat_dist.norm_percent_area_in_each_std()
    
        if distribution_name == "wald":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = stat_dist.wald_percent_area_in_each_std()
    
        if distribution_name == "gamma":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = stat_dist.gamma_percent_area_in_each_std()
    
        if distribution_name == "exponweib":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
                                                 stat_dist.exponweib_percent_area_in_each_std()
    
        if distribution_name == "weibull_min":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
                                               stat_dist.weibull_min_percent_area_in_each_std()
    
        if distribution_name == "weibull_max":
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
                                               stat_dist.weibull_max_percent_area_in_each_std()
            
        return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4
    
    #---------------------------------------------------------------------------
    def get_percent_std_data_from_best_distribution(total_size: int,left_interval_in_data_list: list,\
                                            right_interval_in_data_list: list , distribution_list: list) -> list:
    # def get_percent_std_data_from_best_distribution(total_size,distribution_list):

        # left_data_set = copy.deepcopy(left_interval_in_data_list)

        # right_data_set = copy.deepcopy(right_interval_in_data_list)
    
        # end_height_each_distrubtion = []
        difference_std_height_left_right_end = []
        
        for distribution_name in distribution_list:
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = boostrap_v1.distribution_percent(\
                                                     distribution_name)
            
            slstd4 = math.ceil(percent_data_lstd4*total_size/100.0)
            srstd4 = math.ceil(percent_data_rstd4*total_size/100.0)
            
            left_height = len(left_interval_in_data_list)
            left_height_difference = abs(left_height - slstd4)
            
            right_height = len(right_interval_in_data_list)
            right_height_difference = abs(right_height - srstd4)
            difference_std_height_left_right_end.append([left_height_difference, \
                                                right_height_difference,               
                                                distribution_name])
    
        size = len(difference_std_height_left_right_end)
        left_min = 9999
        right_min = 9999
        
        right_name = ''
        left_name =''
        for i in range(size):
            left_height = difference_std_height_left_right_end[i][0]
            right_height = difference_std_height_left_right_end[i][1]
            if left_min > left_height:
                left_min = left_height
                left_name = difference_std_height_left_right_end[i][2]
    
            if right_min > right_height:
                right_min = right_height
                right_name = difference_std_height_left_right_end[i][2]
        
        if right_name == left_name:
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = boostrap_v1.distribution_percent(\
                                                     right_name)
        if right_name != left_name:
            if left_min < right_min:
                percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
                percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
                percent_data_rstd3, percent_data_rstd4 = boostrap_v1.distribution_percent(\
                                                     right_name)
            if right_min < left_min:
                percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
                percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
                percent_data_rstd3, percent_data_rstd4 = boostrap_v1.distribution_percent(\
                                                     left_name)
    
        print("---- best distribution: left std", left_name, " right std:", right_name)
        # return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        #         percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        #         percent_data_rstd3, percent_data_rstd4
        return [percent_data_lstd4, percent_data_lstd3, percent_data_lstd2,percent_data_lstd1,\
                percent_data_rstd1, percent_data_rstd2,percent_data_rstd3, percent_data_rstd4] 
    
    #---------------------------------------------------------------------------
    def bootstrap_whole_data(input_data: list, number_bootstrap_iteration: int = None,\
                             minmax_boost: bool = False, prob: bool = None) -> bool:
        data_set = copy.deepcopy(input_data)
        if minmax_boost is None:
           minmax_boost = False
        if prob is None:
            prob is False
        if number_bootstrap_iteration is None:
            number_bootstrap_iteration = 600
    
        bootstrap_means = np.zeros(number_bootstrap_iteration)
        bootstrap_std = []
        bootstrap_max_diff_dist_std = []
    
        # Compute prob. of data points.    
        size_data_set = len(data_set)
        if size_data_set < 30:
            size_boost = size_data_set
        else:
            size_boost = 30
        if prob:    
            dist = abs(data_set-np.mean(data_set))
            idist = 1/dist
            pdist = idist/np.sum(idist)    
            input_mean = sum(np.array(pdist)*np.array(data_set))   
        else:
            input_mean  = np.mean(data_set)
        
        previous_bootstrap_mean = 0
        
        bootstrap_sample_list = []
        if minmax_boost:
            bootstrap_sample_min_list = []
            bootstrap_sample_max_list = []
        # Perform bootstrap sampling
        iteration = number_bootstrap_iteration + number_bootstrap_iteration
    
        print("\n---- compute mean from each bootstrap sampling sets for estimated mean\n")
        
        for i in range(number_bootstrap_iteration):
            # with_replacement = True 
            # bootstrap_sample = list(np.random.choice(data_set, size_data_set, \
                                                     # replace = with_replacement))
            if prob:
                bootstrap_sample = [ np.random.choice(data_set, 3,replace=True,p=pdist)[2] for i in range(size_boost)]
            else:
                bootstrap_sample = [ np.random.choice(data_set, 3,replace=True)[2] for i in range(size_boost)]
           
            bootstrap_sample_list.append(bootstrap_sample)
            present_bootstrap_mean = (np.mean(bootstrap_sample) +
                                      previous_bootstrap_mean)/2
            previous_bootstrap_mean = present_bootstrap_mean
            if minmax_boost:
                bootstrap_sample_min_list.append(min(bootstrap_sample))
                bootstrap_sample_max_list.append(max(bootstrap_sample))
            
            bootstrap_means[i] = present_bootstrap_mean
            print("#", end = "")
    
        print("\n")
        estimated_mean = np.mean(bootstrap_means)
        estimated_std_of_mean = boostrap_v1.stdev(bootstrap_means,estimated_mean)
        different_input_mean_bootstrap_mean = abs(boostrap_v1.mean(input_data) - estimated_mean)
        # different_input_mean_bootstrap_mean = abs(boostrap_v1.mean(input_data) - estimated_mean)
    
        print("\n---- compute std from each bootstrap sampling set for estimated std\n")
        for i in range(number_bootstrap_iteration):
            data = bootstrap_sample_list[i]
            variance = [(k-estimated_mean)**2 for k in data]
            std = math.sqrt(sum(variance)/(size_boost-1))
            # std = 0
            # for j in range(size_boost):
            #     std = std + (data[j] - estimated_mean)**2
            # std = math.sqrt(std/(size_boost-1))
            # # diff_dist_right_std = max(data) - (std + estimated_mean)
            # # diff_dist_left_std = (estimated_mean - std) - min(data)
            # # max_std = (diff_dist_right_std + diff_dist_left_std)/2
            bootstrap_std.append(std)
            # bootstrap_max_diff_dist_std.append(max_std)
            print("#", end = "")
    
        print("\n")
        estimated_std = np.mean(bootstrap_std)
        estimated_std_of_std = boostrap_v1.stdev(bootstrap_std,estimated_std)
        different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data,input_mean) - estimated_std)
        # different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data) - estimated_std)
    
        estimated_diff_dist_std = np.mean(bootstrap_max_diff_dist_std)
        estimated_std_of_diff_dist_std = np.std(bootstrap_max_diff_dist_std)
    
        # input_mean = np.mean(input_data)
        input_std = boostrap_v1.stdev(input_data,input_mean)
    
        print("=====================================================================")
        print(">>>>> in Bootstrap")
        
        print("left min:", min(data_set))
        print("right max:", max(data_set))
        print("\ninput mean:", input_mean)
        print("estimated_mean:", estimated_mean)
        print("estimated_std_of_mean:", estimated_std_of_mean)
        print(" ")
        print("input std:", input_std)
        print("estimated_std:", estimated_std)
        print("estimated_std_of_std:", estimated_std_of_std)
        print(" ")
        print("estimated_diff_dist_std:", estimated_diff_dist_std)
        print("estimated_std_of_diff_dist_std:", estimated_std_of_diff_dist_std)
    
        # input_mean = np.mean(input_data)
        if minmax_boost is True:
           if prob:
                dist_min = abs(bootstrap_sample_min_list-np.mean(bootstrap_sample_min_list))
                idist_min = 1/dist_min
                pdist_min = idist_min/np.sum(idist_min)
                left_min_1 = sum(np.array(pdist_min)*np.array(bootstrap_sample_min_list))
                left_min = left_min_1
                
                dist_max = abs(bootstrap_sample_max_list-np.mean(bootstrap_sample_max_list))
                idist_max = 1/dist_max
                pdist_max = idist_max/np.sum(idist_max)
                right_max_1 = sum(np.array(pdist_max)*np.array(bootstrap_sample_max_list))
                right_max = right_max_1
           else:
               left_min = np.mean(bootstrap_sample_min_list)
               right_max = np.mean(bootstrap_sample_max_list)
        else:
            
           left_min = min(data_set)
           right_max = max(data_set)
    
        if estimated_mean < input_mean:
            # left_min = min(data_set)
            left_final_estimated_std = left_min - different_input_mean_bootstrap_mean
    
            print("\n>>>>> Bootstrap in left end: estimated_mean < input_mean")
            print("left min:", min(data_set), " final_estimated_std:", \
                      left_final_estimated_std)
                
        
        if input_mean < estimated_mean:
    #         left_min = min(data_set)
    # #        left_final_estimated_std = left_min - estimated_std_of_std
            left_final_estimated_std = left_min - different_input_std_bootstrap_std
            print("\n>>>>> Bootstrap in left end: input_mean < estimated_mean")
            print("left min:", min(data_set), " final_estimated_std:", \
                    left_final_estimated_std)
        
        if estimated_mean < input_mean:
    #         right_max = max(data_set)
    # #        right_final_estimated_std = right_max + estimated_std_of_std
            right_final_estimated_std = right_max + different_input_std_bootstrap_std
            print("\n>>>>> Bootstrap in right end: estimated_mean < input_mean")
            print("right max:", max(data_set), " final_estimated_std:", \
                      right_final_estimated_std)
                                      
        if input_mean < estimated_mean:
            # right_max = max(data_set)
            right_final_estimated_std = right_max + different_input_mean_bootstrap_mean            
            print("\n>>>>> Bootstrap in right end: input_mean < estimated_mean")
            print("right max:", max(data_set), " final_estimated_std:", \
                      right_final_estimated_std)
    
        print(" ")
        print("left_final_estimated_std:", left_final_estimated_std)
        print("right_final_estimated_std:", right_final_estimated_std)
    
        return left_final_estimated_std, right_final_estimated_std

class booststream:
    def __init__(self,online: bool = None,minmax_boost: bool = None ,\
                 addnoise = None ,prob: bool = None):
        
        if minmax_boost is  None:
            self.minmax_boost = False
        else:
            self.minmax_boost = True
        
        if prob is None:
            self.prob = False
        else:
            self.prob = True
            
        if online is None:
            self.online = True
        else:
            self.online = False
        
        if addnoise is None:
            addnoise = False
        else:
            addnoise = True
        # self.online = pd_series['online']
        # self.prob = pd_series['prob']
        # self.minmax_boost = pd_series['minmax_boost']
        self.numbin = 8
        self.number_bt_iter = 600
        self.nboost = 3
        self.filename = ''
        self.filewd = ''
        self.pop_max = 0
        self.pop_min = 0
        self.ch_size = 0
        self.feed_size = 0  
        self.dist_list = ['exponweib', 'wald', 'gamma', 'norm',\
                         'expon', 'powerlaw', 'lognorm', 'chi2', 'weibull_min',\
                         'weibull_max']
        
        self.expandR = [] 
        self.expandL = []
        self.range = 0
        self.endL = []
        self.endR = []
        self.endLn = []
        self.endRn = []
        self.min_chs = -1
        self.max_chs = -1
        self.total_size = 0
        self.avg = []
        self.std = []
        self.re_samp = [] # right error from sample set ()
        self.le_samp = [] # left error from sample set ()
        self.re_pop = [] # right error from population set ()
        self.le_pop = [] # left error from population set ()
        self.expandch = [] # chunk number with expanded left or right end
    
        
    # def apply_mm(self):
    #     self.minmax_boost = True
    
    # def apply_prob(self):
    #     self.prob = True
    def report_console(self) -> None:
        print(f'Dataset: {self.filename}')
        dict1 = {'Online':self.online,'MinmaxBoost':[self.minmax_boost],\
                 'Prob.':[self.prob]}
        df1 = pd.DataFrame(dict1)        
        print(tabulate(df1, headers = 'keys', tablefmt = 'psql')) 
        method = ['Pop.','Samp','Expand','Er pop','Er samp','Rang pop.','Range']
        dict = {'Min':[self.pop_min,self.min_chs,self.expandL,self.le_pop[-1],\
                       self.le_samp[-1],(self.pop_max-self.pop_min),(self.expandR - self.expandL)],\
                'Max':[self.pop_max,self.max_chs,self.expandR,self.re_pop[-1],\
                       self.re_samp[-1],(self.pop_max-self.pop_min),(self.expandR - self.expandL)] }
        df = pd.DataFrame(dict,index = method)    
        print(tabulate(df, headers = 'keys', tablefmt = 'psql'))  
        dict2 = {'Expand ch':self.expandch}
        df2 = df1 = pd.DataFrame(dict2)
        print(tabulate(df2, headers = 'keys', tablefmt = 'psql'))
    
    def report_xcel(self,filename,append = None) -> None:
        sheetname1 = 'O'+str(int(self.online))+'-mm'+\
            str(int(self.minmax_boost))+'-p'+str(int(self.prob))
        method = ['Pop.','Samp','Expand','Er pop','Er samp','Rang pop.','Range']
        dict = {'Min':[self.pop_min,self.min_chs,self.expandL,self.le_pop[-1],\
                       self.le_samp[-1],(self.pop_max-self.pop_min),(self.expandR - self.expandL)],\
                'Max':[self.pop_max,self.max_chs,self.expandR,self.re_pop[-1],\
                       self.re_samp[-1],(self.pop_max-self.pop_min),(self.expandR - self.expandL)] }
        df = pd.DataFrame(dict,index = method)        
        if append is None:
            append = False
            with pd.ExcelWriter(filename+".xlsx") as writer:
                df.to_excel(writer,sheet_name = sheetname1,index=True)
        else:
            append = True
            with pd.ExcelWriter(filename+".xlsx", mode = 'a') as writer:
                df.to_excel(writer,sheet_name = sheetname1,index=True)  
    
    def expand_whole(self,input_data: list) -> None:
        data_set = copy.deepcopy(input_data)
        bootstrap_means = np.zeros(self.number_bt_iter)
        bootstrap_std = []
        size_data_set = len(data_set)
        previous_bootstrap_mean = 0
        bootstrap_sample_list = []
        if self.minmax_boost:
            bootstrap_sample_min_list = []
            bootstrap_sample_max_list = []
            
        # Setting boostrap size
        size_boost = size_data_set
        # if size_data_set < 30:
        #     size_boost = size_data_set
        # else:
        #     size_boost = 30
        
        
        
        # Perform bootstrap sampling
        data_set_mean = np.mean(data_set)
        if self.prob:    
            dist = abs(data_set-data_set_mean)
            idist = 1/dist
            pdist = idist/np.sum(idist)    
            input_mean = sum(np.array(pdist)*np.array(data_set))   
        else:
            input_mean  = data_set_mean
        
        previous_bootstrap_mean = 0
        
        bootstrap_sample_list = [] 
        for i in range(self.number_bt_iter):
            if self.prob:
                #bootstrap_sample = [ np.random.choice(data_set, 3,replace=True,p=pdist)[2] for i in range(size_boost)]
                bootstrap_sample = list(np.random.choice(data_set, size_boost,replace=True,p=pdist))
            else:
                #bootstrap_sample = [ np.random.choice(data_set, 3,replace=True)[2] for i in range(size_boost)]
                bootstrap_sample = list(np.random.choice(data_set, size_boost,replace=True))
            
            bootstrap_sample_list.append(bootstrap_sample)
            present_bootstrap_mean = (np.mean(bootstrap_sample) +
                                      previous_bootstrap_mean)/2
            previous_bootstrap_mean = present_bootstrap_mean
            if self.minmax_boost:
                bootstrap_sample_min_list.append(min(bootstrap_sample))
                bootstrap_sample_max_list.append(max(bootstrap_sample))
            
            bootstrap_means[i] = present_bootstrap_mean
        estimated_mean = np.mean(bootstrap_means)
        # estimated_std_of_mean = boostrap_v1.stdev(bootstrap_means,estimated_mean)
        different_input_mean_bootstrap_mean = abs(boostrap_v1.mean(input_data) - estimated_mean)
        
        for i in range(self.number_bt_iter):
            data = bootstrap_sample_list[i]
            variance = [(k-estimated_mean)**2 for k in data]
            std = math.sqrt(sum(variance)/(size_boost-1))
            
            bootstrap_std.append(std)
        estimated_std = np.mean(bootstrap_std)
        # estimated_std_of_std = boostrap_v1.stdev(bootstrap_std,estimated_std)
        different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data,input_mean) - estimated_std)
        # different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data) - estimated_std)
    
        # estimated_diff_dist_std = np.mean(bootstrap_max_diff_dist_std)
        # estimated_std_of_diff_dist_std = np.std(bootstrap_max_diff_dist_std)
    
        # input_mean = np.mean(input_data)
        # input_std = boostrap_v1.stdev(input_data,input_mean)
        if self.minmax_boost is True:
           if self.prob:
                dist_min = abs(bootstrap_sample_min_list-np.mean(bootstrap_sample_min_list))
                idist_min = 1/dist_min
                pdist_min = idist_min/np.sum(idist_min)
                left_min_1 = sum(np.array(pdist_min)*np.array(bootstrap_sample_min_list))
                left_min = left_min_1
                
                dist_max = abs(bootstrap_sample_max_list-np.mean(bootstrap_sample_max_list))
                idist_max = 1/dist_max
                pdist_max = idist_max/np.sum(idist_max)
                right_max_1 = sum(np.array(pdist_max)*np.array(bootstrap_sample_max_list))
                right_max = right_max_1
           else:
               left_min = np.mean(bootstrap_sample_min_list)
               right_max = np.mean(bootstrap_sample_max_list)
        else:
            
           left_min = min(data_set)
           right_max = max(data_set)
           
        if estimated_mean < input_mean:
            # left_min = min(data_set)
            left_final_estimated_std = left_min - different_input_mean_bootstrap_mean
    
                
        
        if input_mean < estimated_mean:
    #         left_min = min(data_set)
    # #        left_final_estimated_std = left_min - estimated_std_of_std
            left_final_estimated_std = left_min - different_input_std_bootstrap_std
            
        
        if estimated_mean < input_mean:
    #         right_max = max(data_set)
    # #        right_final_estimated_std = right_max + estimated_std_of_std
            right_final_estimated_std = right_max + different_input_std_bootstrap_std
            
                                      
        if input_mean < estimated_mean:
            # right_max = max(data_set)
            right_final_estimated_std = right_max + different_input_mean_bootstrap_mean            
        self.expandL = left_final_estimated_std
        self.expandR = right_final_estimated_std
        self.min_chs = min(input_data)
        self.max_chs = max(input_data)
        self.total_size = len(input_data)
        self.range = self.expandR - self.expandL
        self.le_samp.append(self.min_chs - self.expandL)
        self.re_samp.append(self.max_chs - self.expandR)
        self.le_pop.append(self.pop_min - self.expandL)
        self.re_pop.append(self.pop_max - self.expandR)

    def expand(self,new_data_chunk: list) -> bool:
        self.total_size += len(new_data_chunk)
        new_data_chunk_min = min(new_data_chunk)
        new_data_chunk_max = max(new_data_chunk)
        if self.max_chs < new_data_chunk_max:
            self.max_chs = new_data_chunk_max
            
        if self.min_chs > new_data_chunk_min:
            self.min_chs = new_data_chunk_min    
        expand_max = False
        expand_min = False
        expand = False
        expansion = False
        if new_data_chunk_min < self.expandL:
            # expand_min = True
            if len(self.endL) >= self.nboost and (self.minmax_boost is True):
                # end_bin_left_tmp = copy.deepcopy(end_bin_left)
                # end_bin_left_tmp.append(new_data_chunk_min)
                self.endL.append(new_data_chunk_min)
                expand_min = True
                # expansion = True
                adjust_left_std = boostrap_v1.bootstrap_online(self.endL, "left",\
                                                               number_bootstrap_iteration = self.number_bt_iter, \
                                                                   minmax_boost = self.minmax_boost,\
                                                                       prob = self.prob)
                if self.expandL >= adjust_left_std:
                    self.expandL = adjust_left_std
                    # end_bin_left.append(adjust_left_std)
                # else:
                #     self.expandL = (self.expandL+adjust_left_std)/2
                    
            else:
                expand_min = True
                # expansion = True
                self.expandL = new_data_chunk_min
        
        if new_data_chunk_max > self.expandR:
            # expand_max = True
            if (len(self.endR) >= self.nboost) and (self.minmax_boost is True):
                
                self.endR.append(new_data_chunk_max)
                adjust_right_std = boostrap_v1.bootstrap_online(self.endR, "right",\
                                                                number_bootstrap_iteration = self.number_bt_iter, \
                                                                    minmax_boost = self.minmax_boost,\
                                                                        prob = self.prob)
                
                expand_max = True
                # expansion = True
                if self.expandR <= adjust_right_std:
                    self.expandR = adjust_right_std
                    # end_bin_left.append(adjust_left_std)
                # else:
                #     self.expandR = (self.expandR+adjust_right_std)/2
            else:
                expand_max = True
                # expansion = True
                self.expandR = new_data_chunk_max
                
        if expand_min is True or expand_max is True:
            self.update_center_range(self.expandL,self.expandR)
            avg = self.avg[-1]
            std = self.std[-1]
            end_bin_left = []
            end_bin_right = []
            new_data_chunk = new_data_chunk + self.endL + self.endR
            self.endL = [k for k in new_data_chunk if (avg - 4*std <= k <= avg - 3*std)]
            self.endR = [k for k in new_data_chunk if (avg + 3*std <= k <= avg + 4*std)]
            hist_data = [0] * int(self.numbin)    
            hist_theo = [0] * int(self.numbin)
            hist_data[0] = len(end_bin_left) # left most bin
            hist_data[-1] = len(end_bin_right) # right most bin
            hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
            hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
            percent_data = boostrap_v1.get_percent_std_data_from_best_distribution(\
                                               self.total_size, self.endL, self.endR, \
                                               self.dist_list)
            hist_theo = [math.ceil(i*self.total_size/100.0) for i in percent_data]
            self.endLn.append(len(self.endL))
            self.endRn.append(len(self.endR))
            expand = False
            expansion = False
            difference_max = hist_data[-1] - hist_theo[-1]
            difference_min = hist_data[0] - hist_theo[0]
            # while (difference_max > 0 or difference_min > 0):
            if (difference_max > 0 or difference_min > 0):
                dif_expand = True    
            else:
                dif_expand = False    
            while dif_expand is True:    
                # difference_max_tmp = difference_max
                # difference_min_tmp = difference_min
                expandL = self.expandL
                expandR = self.expandR
                if difference_max > 0:
            
                    # if use_bootstrap is True:
                    if hist_data[-1] >= self.nboost:
                        if self.expandR <= max(self.endR):
                            self.expandR = boostrap_v1.bootstrap_online(self.endR, "right",\
                                                                        number_bootstrap_iteration = self.number_bt_iter, \
                                                                          minmax_boost = self.minmax_boost)
                            # self.expandR = max_right_end
                            expand = True
                            expansion = True
                        
                if difference_min > 0:
                    if hist_data[-1] >= self.nboost:
                        if self.expandL >= min(self.endL):
                            self.expandL = boostrap_v1.bootstrap_online(self.endL, "left",\
                                                                        number_bootstrap_iteration = self.number_bt_iter, \
                                                                            minmax_boost = self.minmax_boost)
                            # self.expandL = min_left_end
                            expand = True
                            expansion = True
                if expand is True:
                    self.update_center_range(self.expandL,self.expandR)
                    avg = self.avg[-1]
                    std = self.std[-1]
                    expand = False
                    hist_data = [0] * int(self.numbin)
                    end_bin_left = [] # add
                    end_bin_right = [] # add
                    end_bin_left = [i for i in new_data_chunk if (avg - 4*std <= i <= avg - 3*std)]
                    end_bin_right = [i for i in new_data_chunk if (avg + 3*std <= i <= avg + 4*std)]
                    # end_bin_num_left.append(len(end_bin_left)) 
                    # end_bin_num_right.append(len(end_bin_right))
                    hist_data[0] = len(end_bin_left)
                    hist_data[-1] = len(end_bin_right)
                    hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
                    hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
                    # count += 1
                    # boostrap_v1.plot_histogram(hist_theo[0],hist_theo[1], hist_theo[2],hist_theo[3],\
                    #                    hist_theo[4],hist_theo[5], hist_theo[6],hist_theo[7],\
                    #                        hist_data[0],hist_data[1], hist_data[2],hist_data[3],\
                    #                            hist_data[4],hist_data[5], hist_data[6],hist_data[7],\
                    #                                min_expand, max_expand, 1, expand)
                    # plt.show()
                    # difference_max = hist_data[-1] - hist_theo[-1]
                    # difference_min = hist_data[0] - hist_theo[0]
                    self.endL = end_bin_left
                    self.endR = end_bin_right
                    self.endLn.append(len(end_bin_left))
                    self.endRn.append(len(end_bin_right))
                    
                    
                    # if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
                    if expandL == self.expandL and expandR == self.expandR:     
                        # if self.expandL < new_data_chunk_min and new_data_chunk_max < self.expandR:
                        #     # tmp_leftright = False
                        #     break
                        dif_expand = False
                
                        
                # else:
                    
                #     if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
                #         # tmp_chunk_end  = end_bin_left + end_bin_right
                #         # tmp_leftright = True
                #         break
        if (expansion is True) or (expand_max is True) or (expand_min is True):
           self.range = self.expandR - self.expandL
           self.le_samp.append(self.min_chs - self.expandL)
           self.re_samp.append(self.max_chs - self.expandR)
           self.le_pop.append(self.pop_min - self.expandL)
           self.re_pop.append(self.pop_max - self.expandR)
           expansion = True
        return expansion        
            
        
    def expand_init(self,new_data_chunk: list) -> bool:
        self.total_size += len(new_data_chunk)
        # min_lef_end = min(new_data_chunk)
        # max_right_end = max(new_data_chunk)
        
        min_new_input = min(new_data_chunk)
        max_new_input = max(new_data_chunk)
        self.max_chs= max_new_input
        self.min_chs = min_new_input
        self.expandL = min_new_input
        self.expandR = max_new_input
        self.update_center_range(self.expandL,self.expandR)
        avg = self.avg[-1]
        std = self.std[-1]
        # end_bin_left = []
        # end_bin_right = []
        self.endL = [i for i in new_data_chunk if (avg - 4*std <= i <= avg - 3*std)]
        self.endR = [i for i in new_data_chunk if (avg + 3*std <= i <= avg + 4*std)]
        # end_bin_num_left.append(len(end_bin_left)) 
        # end_bin_num_right.append(len(end_bin_right))  
        hist_data = [0] * int(self.numbin)    
        hist_theo = [0] * int(self.numbin)
        hist_data[0] = len(self.endL) # left most bin
        hist_data[-1] = len(self.endR) # right most bin
        hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
        hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
        percent_data = boostrap_v1.get_percent_std_data_from_best_distribution(\
                                           self.total_size, self.endL, self.endR, \
                                           self.dist_list)
        hist_theo = [math.ceil(i*self.total_size/100.0) for i in percent_data]
        
        self.endLn.append(len(self.endL))
        self.endRn.append(len(self.endR))
        expand = False
        expansion = False
        
        difference_max = hist_data[-1] - hist_theo[-1]
        difference_min = hist_data[0] - hist_theo[0]
        while (difference_max > 0 or difference_min > 0):
            difference_max_tmp = difference_max
            difference_min_tmp = difference_min
            if difference_max > 0:
                # if use_bootstrap is True:
                if hist_data[-1] >= self.nboost:
                    if self.expandR <= max(self.endR):
                        self.expandR = boostrap_v1.bootstrap_online(self.endR, "right",\
                                                                      number_bootstrap_iteration = self.number_bt_iter, \
                                                                          minmax_boost = self.minmax_boost)
                        # self.expandR = max_right_end
                        expand = True
                        expansion = True
                    
            if difference_min > 0:
                if hist_data[-1] >= self.nboost:
                    if self.expandL >= min(self.endL):
                        self.expandL = boostrap_v1.bootstrap_online(self.endL, "left",\
                                                                    number_bootstrap_iteration = self.number_bt_iter, \
                                                                        minmax_boost = self.minmax_boost)
                        # self.expandL = min_left_end
                        expand = True
                        expansion = True
            if expand is True:
                self.update_center_range(self.expandL,self.expandR)
                avg = self.avg[-1]
                std = self.std[-1]
                expand = False
                hist_data = [0] * int(self.numbin)
                end_bin_left = [] # add
                end_bin_right = [] # add
                end_bin_left = [i for i in new_data_chunk if (avg - 4*std <= i <= avg - 3*std)]
                end_bin_right = [i for i in new_data_chunk if (avg + 3*std <= i <= avg + 4*std)]
                # end_bin_num_left.append(len(end_bin_left)) 
                # end_bin_num_right.append(len(end_bin_right))
                hist_data[0] = len(end_bin_left)
                hist_data[-1] = len(end_bin_right)
                hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
                hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
                # count += 1
                # boostrap_v1.plot_histogram(hist_theo[0],hist_theo[1], hist_theo[2],hist_theo[3],\
                #                    hist_theo[4],hist_theo[5], hist_theo[6],hist_theo[7],\
                #                        hist_data[0],hist_data[1], hist_data[2],hist_data[3],\
                #                            hist_data[4],hist_data[5], hist_data[6],hist_data[7],\
                #                                min_expand, max_expand, 1, expand)
                # plt.show()
                difference_max = hist_data[-1] - hist_theo[-1]
                difference_min = hist_data[0] - hist_theo[0]
                self.endL = end_bin_left
                self.endR = end_bin_right
                self.endLn.append(len(end_bin_left))
                self.endRn.append(len(end_bin_right))
            
                if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
                    if self.expandL < min_new_input and max_new_input < self.expandR:
                        # tmp_leftright = False
                        break
            else:
                
                if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
                    # tmp_chunk_end  = end_bin_left + end_bin_right
                    # tmp_leftright = True
                    break
        if expansion:
           self.range = self.expandR - self.expandL
           self.le_samp.append(self.min_chs - self.expandL)
           self.re_samp.append(self.max_chs - self.expandR)
           self.le_pop.append(self.pop_min - self.expandL)
           self.re_pop.append(self.pop_max - self.expandR)
               
        return expansion 
        
            
    def update_samp_er(self) -> None:
        self.le_samp.append(self.min_chs -self.expandL)
        self.re_samp.append(self.max_chs-self.expandR)
            
    def update_pop_er(self) -> None:
        self.le_pop.append(self.pop_min-self.expandL)
        self.re_pop.append(self.pop_max-self.expandR)
    
    def update_center_range(self,leftmost: float,rightmost: float)-> None:
        self.avg.append((rightmost+leftmost)/2)
        self.std.append((rightmost-leftmost)/8)
        # std = (max_new_input - min_new_input)/8
        # avg = (max_new_input + min_new_input)/2
    
    def update_range(self) -> None:
        self.range = self.expandR -self.expandL 
    
    # def update_cls_att(self, filewd: str,filename: str,\
    #                    pop_min: float, pop_max: float,ch_size: int,\
    #                    feed_size: int, addnoise: bool = None) -> None:
    def update_cls_att(self, filename: str,\
                       pop_min: float, pop_max: float,ch_size: int,\
                       feed_size: int, addnoise: bool = None) -> None:    
        # self.filewd = filewd
        self.filename = filename
        self.pop_min = pop_min
        self.pop_max = pop_max
        self.ch_size = ch_size
        self.feed_size = feed_size
        if addnoise is not None:
            self.addnoise = False
    
    def update_cls_addnoise(self):
        self.addnoise = True
