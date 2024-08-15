# File is created by premjunsawang 
# This is a sample Python script.
'''
Follow the file of AjChidV4
'''

import numpy as np
import argparse
import pandas as pd
import math
import matplotlib.pyplot as plt
import array as arr
import sys
import copy
import csv
import scipy.stats as stats
from scipy.stats import kurtosis
from scipy.stats import skew
from matplotlib import pyplot
import os.path
import random
from matplotlib import pyplot

from fitter import Fitter
from fitter import get_distributions
from fitter import get_common_distributions
# import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import exponweib
from scipy.stats import wald
from scipy.stats import exponpow
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import expon
from scipy.stats import powerlaw
from scipy.stats import lognorm
from scipy.stats import cauchy
from scipy.stats import chi2
from scipy.stats import uniform
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import weibull_min
from scipy.stats import weibull_max


def gamma_percent_area_in_each_std():
    a = 1.99
    mean, var, skew, kurt = gamma.stats(a, moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = gamma.cdf(mean, a, loc_value, scale_value)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# ----------------------------------------------------------------------------
def norm_percent_area_in_each_std():
    mean, var, skew, kurt = norm.stats(moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = norm.cdf(mean, loc_value, scale_value)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# ----------------------------------------------------------------------------
def exponpow_percent_area_in_each_std():
    b = 2.7
    mean, var, skew, kurt = exponpow.stats(b, moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = exponpow.cdf(mean, b, loc_value, scale_value)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# ----------------------------------------------------------------------------
def wald_percent_area_in_each_std():
    mean, var, skew, kurt = wald.stats(moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = wald.cdf(mean, loc_value, scale_value)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# ----------------------------------------------------------------------------
def exponweib_percent_area_in_each_std():
    a, c = 2.89, 1.95
    mean, var, skew, kurt = exponweib.stats(a, c, moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# ------------------------------------------------------------------------------
def rayleigh_percent_area_in_each_std():
    mean, var, skew, kurt = rayleigh.stats(moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = rayleigh.cdf(mean, loc_value, scale_value)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# ---------------------------------------------------------------------------
def powerlaw_percent_area_in_each_std():
    a = 0.659
    mean, var, skew, kurt = powerlaw.stats(a, moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# -------------------------------------------------------------------------
def expon_percent_area_in_each_std():
    mean, var, skew, kurt = expon.stats(moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = expon.cdf(mean, loc_value, scale_value)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# -------------------------------------------------------------------------
def uniform_percent_area_in_each_std():
    mean, var, skew, kurt = uniform.stats(moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = uniform.cdf(mean, loc_value, scale_value)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# --------------------------------------------------------------------------
def lognorm_percent_area_in_each_std():
    s = 0.954
    mean, var, skew, kurt = lognorm.stats(s, moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = lognorm.cdf(mean, s, loc_value, scale_value)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# --------------------------------------------------------------------
def chi2_percent_area_in_each_std():
    df = 55
    mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = chi2.cdf(mean, df, 0, 1)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# -----------------------------------------------------------------------
def weibull_min_percent_area_in_each_std():
    c = 1.79
    mean, var, skew, kurt = weibull_min.stats(c, moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = weibull_min.cdf(mean, c, 0, 1)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# ----------------------------------------------------------------------
def weibull_max_percent_area_in_each_std():
    c = 2.87
    mean, var, skew, kurt = weibull_max.stats(c, moments='mvsk')
    std = math.sqrt(var)

    loc_value = mean
    scale_value = std

    lstd1 = mean - std
    lstd2 = mean - 2 * std
    lstd3 = mean - 3 * std
    lstd4 = mean - 4 * std
    rstd1 = mean + std
    rstd2 = mean + 2 * std
    rstd3 = mean + 3 * std
    rstd4 = mean + 4 * std

    cdf_area_at_mean = weibull_max.cdf(mean, c, 0, 1)
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
    # --------------------------------------------------------------------------

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
                 area_between_lstd2_lstd1 + area_between_lstd1_mean + \
                 area_between_mean_rstd1 + area_between_rstd1_rstd2 + \
                 area_between_rstd2_rstd3 + area_between_rstd3_rstd4

    percent_data_rstd1 = 100 * (area_between_mean_rstd1 / total_area)
    percent_data_rstd2 = 100 * (area_between_rstd1_rstd2 / total_area)
    percent_data_rstd3 = 100 * (area_between_rstd2_rstd3 / total_area)
    percent_data_rstd4 = 100 * (area_between_rstd3_rstd4 / total_area)

    percent_data_lstd1 = 100 * (area_between_lstd1_mean / total_area)
    percent_data_lstd2 = 100 * (area_between_lstd2_lstd1 / total_area)
    percent_data_lstd3 = 100 * (area_between_lstd3_lstd2 / total_area)
    percent_data_lstd4 = 100 * (area_between_lstd4_lstd3 / total_area)

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4
# ==========================================================================
def get_percent_std_data_from_best_distribution(left_interval_in_data_list, \
                                                right_interval_in_data_list, total_size, \
                                                distribution_list):
    left_data_set = copy.deepcopy(left_interval_in_data_list)
    right_data_set = copy.deepcopy(right_interval_in_data_list)

    end_height_each_distrubtion = []
    difference_std_height_left_right_end = []

    for distribution_name in distribution_list:
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = distribution_percent( \
            distribution_name)

        slstd4 = math.ceil(percent_data_lstd4 * total_size / 100.0)
        srstd4 = math.ceil(percent_data_rstd4 * total_size / 100.0)

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

    # -------------- here -----------------------

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
            percent_data_rstd3, percent_data_rstd4 = distribution_percent( \
            right_name)
    if right_name != left_name:
        if left_min < right_min:
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
                percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
                percent_data_rstd3, percent_data_rstd4 = distribution_percent( \
                right_name)
        if right_min < left_min:
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
                percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
                percent_data_rstd3, percent_data_rstd4 = distribution_percent( \
                left_name)

    print("---- best distribution: left std", left_name, " right std:", right_name)
    # return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
    #     percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
    #     percent_data_rstd3, percent_data_rstd4
    return [percent_data_lstd4, percent_data_lstd3, percent_data_lstd2, percent_data_lstd1,
         percent_data_rstd1, percent_data_rstd2,
        percent_data_rstd3, percent_data_rstd4]
def distribution_percent(distribution_name):
    if distribution_name == "chi2":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = chi2_percent_area_in_each_std()

    if distribution_name == "powerlaw":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            powerlaw_percent_area_in_each_std()

    if distribution_name == "expon":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = expon_percent_area_in_each_std()

    if distribution_name == "uniform":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = uniform_percent_area_in_each_std()
    if distribution_name == "lognorm":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            lognorm_percent_area_in_each_std()

    if distribution_name == "rayleigh":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            rayleigh_percent_area_in_each_std()

    if distribution_name == "exponpow":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            exponpow_percent_area_in_each_std()

    if distribution_name == "norm":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = norm_percent_area_in_each_std()

    if distribution_name == "wald":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = wald_percent_area_in_each_std()

    if distribution_name == "gamma":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = gamma_percent_area_in_each_std()

    if distribution_name == "exponweib":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            exponweib_percent_area_in_each_std()

    if distribution_name == "weibull_min":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            weibull_min_percent_area_in_each_std()

    if distribution_name == "weibull_max":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            weibull_max_percent_area_in_each_std()
    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4
# ===========================================================================
def bootstrap(input_data, end_side,min_left_end,max_right_end,number_bootstrap_iteration = None): # updated AjChidV4
    """

    :param input_data: data list for bootstraping (In this case, the left or right most sides are used)
    :param number_bootstrap_iteration:
    :return: estimated standard deviation
    """
    if number_bootstrap_iteration is None:
        number_bootstrap_iteration = 800
    if len(input_data) > 0:
        data_set = copy.deepcopy(input_data)

        bootstrap_means = np.zeros(number_bootstrap_iteration)
        bootstrap_std = []
        bootstrap_max_diff_dist_std = []

        size_data_set = len(data_set)
        previous_bootstrap_mean = 0

        left_min = min(data_set) # AjChidV4
        right_max = max(data_set) # AjChidV4

        bootstrap_sample_list = []
        # Perform bootstrap sampling
        for i in range(number_bootstrap_iteration):
            bootstrap_sample = list(np.random.choice(data_set, size_data_set, replace=True))
            bootstrap_sample_list.append(bootstrap_sample)
            present_bootstrap_mean = (np.mean(bootstrap_sample) + previous_bootstrap_mean) / 2
            previous_bootstrap_mean = present_bootstrap_mean
            bootstrap_means[i] = present_bootstrap_mean

        estimated_mean = np.mean(bootstrap_means)
        estimated_std_of_mean = np.std(bootstrap_means)

        different_input_mean_bootstrap_mean = abs(mean(input_data) - estimated_mean)

        for i in range(number_bootstrap_iteration):
            data = bootstrap_sample_list[i]
            std = 0
            for j in range(size_data_set):
                std = std + (data[j] - estimated_mean) ** 2
            std = math.sqrt(std / size_data_set)
            diff_dist_right_std = max(data) - (std + estimated_mean)
            diff_dist_left_std = (estimated_mean - std) - min(data)
            #        max_std = min(diff_dist_right_std, diff_dist_left_std)
            max_std = (diff_dist_right_std + diff_dist_left_std) / 2
            bootstrap_std.append(std)
            bootstrap_max_diff_dist_std.append(max_std)

        estimated_std = np.mean(bootstrap_std)
        estimated_std_of_std = np.std(bootstrap_std)

        different_input_std_bootstrap_std = abs(stdev(input_data) - 1.02 * estimated_std)

        estimated_diff_dist_std = np.mean(bootstrap_max_diff_dist_std)
        estimated_std_of_diff_dist_std = np.std(bootstrap_max_diff_dist_std)

        print("estimated_mean:", estimated_mean)
        print("estimated_std_of_mean:", estimated_std_of_mean)
        print(" ")
        print("estimated_std:", estimated_std)
        print("estimated_std_of_std:", estimated_std_of_std)
        print(" ")
        print("estimated_diff_dist_std:", estimated_diff_dist_std)
        print("estimated_std_of_diff_dist_std:", estimated_std_of_diff_dist_std)

        #    final_estimated_std = 2*(estimated_std + estimated_std_of_std)+ \
        #                            estimated_diff_dist_std
        # final_estimated_std = different_input_mean_bootstrap_mean - different_input_std_bootstrap_std

        input_mean = np.mean(input_data)
        if end_side == "right":
            if input_mean > estimated_mean:
                final_estimated_std = right_max + 1.04 * estimated_std_of_std
            if estimated_mean >= input_mean:
                final_estimated_std = right_max + 1.02 * estimated_std_of_std + \
                                      estimated_std_of_mean
        if end_side == "left":
            if input_mean > estimated_mean:
                final_estimated_std = left_min - 0.9 * estimated_std_of_std - \
                                      0.91 * estimated_std_of_mean
            if estimated_mean >= input_mean:
                final_estimated_std = left_min - 1.03 * estimated_std_of_std
        print(" ")
        print("final_estimated_std:", final_estimated_std)
    else:
        if end_side == "right":
            final_estimated_std = max_right_end
        else:
            final_estimated_std = min_left_end
    return final_estimated_std
def estimate_width_1D(input_data, max_value, z_score):
    #sample = copy.deepcopy(input_data)
    # sample = np.array(input_data)
    # size = len(sample)
    size = len(np.array(input_data))
    estimate_width = abs(((max_value) / (1 - z_score * (math.sqrt(2 / (size - 1)))) +
                          (max_value) / (1 + z_score * (math.sqrt(2 / (size - 1))))) / 2)

    return estimate_width
def mean(data):
    avg = sum(data)/len(data)
    # list_size = len(data)
    # avg = 0
    # for i in range(list_size):
    #    avg = avg + data[i]
    # avg = avg/list_size
    return avg
#-------------------------------------------------------------------

def stdev(data):
    # list_size = len(data)
    # avg = 0
    # for i in range(list_size):
    #    avg = avg + data[i]
    # avg = avg/list_size
    ndat = len(data)
    avg = sum(data) / ndat
    std = [(x-avg)**2 for x in data]
    std = math.sqrt(sum(std)/ndat)
    # for i in range(list_size):
    #     std = std + (data[i] - avg)**2
    # std = math.sqrt(std/list_size)
    return std

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
def distribution_percent(distribution_name):
    if distribution_name == "chi2":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = chi2_percent_area_in_each_std()

    if distribution_name == "powerlaw":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            powerlaw_percent_area_in_each_std()

    if distribution_name == "expon":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = expon_percent_area_in_each_std()

    if distribution_name == "uniform":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            uniform_percent_area_in_each_std()

    # if distribution_name == "cauchy":
    #     percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
    #         percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
    #         percent_data_rstd3, percent_data_rstd4 = cauchy_percent_area_in_each_std()

    if distribution_name == "lognorm":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            lognorm_percent_area_in_each_std()

    if distribution_name == "rayleigh":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            rayleigh_percent_area_in_each_std()

    if distribution_name == "exponpow":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            exponpow_percent_area_in_each_std()

    if distribution_name == "norm":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = norm_percent_area_in_each_std()

    if distribution_name == "wald":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = wald_percent_area_in_each_std()

    if distribution_name == "gamma":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = gamma_percent_area_in_each_std()

    if distribution_name == "exponweib":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            exponweib_percent_area_in_each_std()

    if distribution_name == "weibull_min":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            weibull_min_percent_area_in_each_std()

    if distribution_name == "weibull_max":
        percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = \
            weibull_max_percent_area_in_each_std()

    return percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
        percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
        percent_data_rstd3, percent_data_rstd4

# def std_hist(prev_size,current_size,left_end_bin,right_end_bin, best_dist = None, distribution_list=None):
def std_hist(prev_size, current_size, left_end_bin, right_end_bin,distribution_list=None):
    """
    :param nsampl: int
    :param port: slist
    :return: list of standard freqencies based on the sample size
    """

    # if best_dist is None:
    #     best_dist = False
    if distribution_list is None:
        distribution_list = ['exponweib', 'wald', 'gamma', 'norm',
                             'expon', 'powerlaw', 'lognorm', 'chi2', 'weibull_min',
                             'weibull_max']
    # if best_dist is False:
    #     prop_percent = [0.1, 2.1, 13.6, 34.1, 34.1, 13.6, 2.1, 0.1]
    total_size = prev_size + current_size
    prop_percent = get_percent_std_data_from_best_distribution(left_end_bin, right_end_bin, total_size, distribution_list)
    listname = ['lstd4','lstd3', 'lstd2', 'lstd1', 'rstd1', 'rstd2', 'rstd3', 'rstd4']
    theo_list = [math.ceil((i*total_size)/100.0) for i in prop_percent]
    # theo_list = list(map(lambda v, nsampl1: math.ceil(v * nsampl1 / 100.0),prop_percent,exp_hist.values()))

    # if prop_percent is None:
    #     prop_percent = [0.1, 2.1, 13.6, 34.1, 34.1, 13.6, 2.1, 0.1]
    # listprop = [nsampl] * len(prop_percent)
    # port_num = lambda v, nsampl1: math.ceil(v * nsampl1 / 100.0)
    # listprop = list(map(port_num, prop_percent, listprop))
    theo_hist = dict(zip(listname,theo_list))
    return theo_hist
def exp_hist(a, avg, std, numbin = None):
    """
    Count the number of data following 8 bin
    :param a:
    :param avg:
    :param std:
    :param numbin:
    :return:
    """
    left_end_bin = []
    right_end_bin = []
    if numbin is None:
        numbin = 8
    listname = ['lstd4', 'lstd3', 'lstd2', 'lstd1', 'rstd1', 'rstd2', 'rstd3', 'rstd4']
    data_hist = [0] * int(numbin)
    data_hist = dict(zip(listname, data_hist))
    # if avg - 4 * std <= data_hist['lstd4'] < avg - 3 * std:
    #     data_hist['lstd4'] +=1
    # if avg + 3 * std <= data_hist['rstd4'] < avg + 4 * std:
    #     data_hist['rstd4'] += 1

    for i in a:
        # if avg - std <= i < avg:
        #     data_hist1[3] += 1
        # if avg - 2 * std <= i < avg - std:
        #     data_hist1[2] += 1
        # if avg - 3 * std <= i < avg - 2 * std:
        #     data_hist1[1] += 1
        if avg - 4 * std <= i < avg - 3 * std:
            left_end_bin.append(i)
            data_hist['lstd4'] +=1

        # if avg <= i < avg + std:
        #     data_hist1[4] += 1
        # if avg + std <= i < avg + 2 * std:
        #     data_hist1[5] += 1
        # if avg + 2 * std <= i < avg + 3 * std:
        #     data_hist1[6] += 1
        if avg + 3 * std <= i < avg + 4 * std:
            right_end_bin.append(i)
            data_hist['rstd4'] += 1

    return data_hist, left_end_bin, right_end_bin

def parse_args(known=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='sim_data/normal_3000.csv')
    # parser.add_argument('--std', action='store_true', default=True)
    # parser.add_argument('--acons', type=float, default=0.01)
    # parser.add_argument('--iter', type=int, default=600)
    # parser.add_argument('--npop', type=int, default=1000)
    # parser.add_argument('--npop', dest='# of generated population', type=int, default=6000)
    # parser.add_argument('--acons', dest='adjusted value of expanding constant', type=float, default=0.01)
    # parser.add_argument('--stapt', type=float, default=1)
    # parser.add_argument('--endpt',type=float, default=10000)
    parser.add_argument('--chsize', type=int, default=100)
    parser.add_argument('--iter', type=int, default=800)
    # parser.add_argument('--std', dest='Use std or not', action = 'store_true')
    # parser.add_argument('--data', dest='data file name', type=str, default='1drandom.csv')
    #     parser.add_argument('--y', type=int, default=2)
    return parser.parse_known_args()[0] if known else parser.parse_args()
def main(opt):
    """
    :param opt:
    :return:

    Pseudocode:
    1. Create population and create the data chunk with predefine the size.
    2. Define standard histogram named Shist.
    3. Define adjust constant parameter.
    4. Compute data histogram named Dhist.
    5. Compute avg, max-right-end, max-left-end, std(range)
    6. While two right most bins of Dhist > these of Shist or two left most bins of Dhist > those of Shist do
    7.  Compute the different max and different min and then add these values to the current data chunk.
    8.  Compute the new avg and std based on the new current data chunk.
    9.  Compute the Dhist based on the new avg and std.
    10. Compute the Shist base on the original data chunk
    """
    randomlist = pd.read_csv(opt.data)
    randomlist = pd.DataFrame.to_numpy(randomlist)
    randomlist = randomlist.flatten()
    amount_population = len(randomlist)
    '''
    amount_population = opt.npop  # ---- must be <= start-end value
    start_interval_random_value = opt.stapt
    end_interval_random_value = opt.endpt
    randomlist = np.random.randint(start_interval_random_value,
                                   end_interval_random_value, amount_population)
    '''


    # =====>  1. Generate a population list (from discrete uniform distribution)


    # # Generate from normal distribution
    # rng1 = np.random.default_rng()
    # randomlist = rng1.normal(size = amount_population)
    # pop_size = len(randomlist)
    input_size = opt.chsize
    # distribution_list = ['exponweib', 'wald', 'gamma', 'norm',
    #                      'expon', 'powerlaw', 'lognorm', 'chi2', 'weibull_min',
    #                      'weibull_max']

    # ----------------------- first chunk ---------------------------------------
    print("------ first chunk ------")
    data = randomlist.tolist()
    a = randomlist[0:input_size].tolist()
    usestd = True

    if usestd is False:
        std = stdev(a)
        avg = mean(data)
        # print("std_interval", std, "average", avg)
        print(f"std_interval is {std:.2f} average is {avg:.2f}")

    if usestd is True:
        max_a = max(a)
        min_a = min(a)
        std = (max_a - min_a) / 8
        avg = (max_a + min_a) / 2
        # print("max-min std", std)
        # print("std_interval", std, "average", avg, "max", max_a, "min", min_a)
        print(f"std_interval is {std:.2f} average is {avg:.2f} max is {max_a:.2f} min is {min_a:.2f}")

    print("-------------------")
    chunk_size = len(a)
    # print("first list size", total_size)
    data_hist, left_end_bin, right_end_bin = exp_hist(a, avg, std)
    prev_size = 0
    current_size = chunk_size
    theo_hist = std_hist(prev_size, current_size, left_end_bin, right_end_bin)
    # ----------------------------------------------------------------------------
    # check if violate standard histogram
    # new_hist_width is an interval for building a histogram. b is adjustted
    # to fit standard histogram
    # -----------------------------------------------------------------------------
    hist_data = []
    for i in a:
        hist_data.append(i)

    max_right_end = max(hist_data)
    min_left_end = min(hist_data)
    avg = (max_right_end + min_left_end) / 2
    std = (max_right_end - min_left_end) / 8
    # count = 1
    # expansion = 0
    expand_count = 0
    adjust_constant = 0.01

    use_bootstrap = True
    while (data_hist['rstd4'] - theo_hist['rstd4']) > 1 or (data_hist['lstd4'] - theo_hist['lstd4']) > 1:
        print(">>>>> expand the 1st chunk.")
        expand_count += 1
        max_right_end = max(hist_data)
        min_left_end = min(hist_data)
        if use_bootstrap is False:
            adjust_left_std = adjust_constant * std
            adjust_right_std = adjust_constant * std
            print("adjust_constant:", adjust_constant)
            adjust_constant = adjust_constant + 0.001
            print("adjust left std:", adjust_left_std)
            print("adjust right std:", adjust_right_std)

        if use_bootstrap is True:
            number_bootstrap_iteration = opt.iter
            adjust_left_std = bootstrap(left_end_bin, "left", number_bootstrap_iteration)
            adjust_right_std = bootstrap(right_end_bin, 'right', number_bootstrap_iteration)
            print("adjust_left_std bootstrap:", adjust_left_std)
            print("adjust_right_std bootstrap:", adjust_right_std)
            print(" ")
        difference_max = data_hist['rstd4']- theo_hist['rstd4']
        if difference_max > 0:
            # max_right_end = max_right_end + adjust_right_std
            max_right_end = adjust_right_std
            max_expand = max_right_end
            # hist_data.append(max_right_end) # prem correct

        difference_min = data_hist['lstd4']- theo_hist['lstd4']
        if difference_min > 0:
            # min_left_end = min_left_end - adjust_right_std
            min_left_end = adjust_left_std
            min_expand = min_left_end # prem correct
            # hist_data.append(min_left_end)
        # max_right_end = max(hist_data)
        # min_left_end = min(hist_data)
        # avg = (max_right_end + min_left_end) / 2 # Aj Chid
        print("expand: min left end", min_left_end, "max right end",
              max_right_end)

        # Aj Chid
        max_right_end = max(hist_data)
        min_left_end = min(hist_data)
        avg = (max_right_end + min_left_end) / 2

        std = stdev(hist_data)

        '''
        avg = mean(hist_data)
        std = stdev(hist_data)
        '''

        data_hist, left_end_bin, right_end_bin = exp_hist(a, avg, std)
        prev_size = 0
        current_size = chunk_size
        theo_hist = std_hist(prev_size, current_size, left_end_bin, right_end_bin)
    print(f'====== The 1st chunk summary =========')
    print(f'# of expansion: {expand_count}')
    print(f'Experiment histogram: {data_hist.values()}')
    print(f'Theory histogram: {theo_hist.values()}')

    print("--------- get next chunk")
    number_data_chunk = int(amount_population // input_size)
    print("number data chunk:", number_data_chunk)
    prev_size = chunk_size
    prev_avg = avg
    prev_std = std
    for i in range(1, number_data_chunk):
        print("\n\n------------------- new chunk", i + 1, "----------------------")
        k = i * input_size
        new_data_chunk = randomlist[k:(k + input_size)].tolist()
        current_size = len(new_data_chunk)
        total_size = prev_size + current_size
        a = a + new_data_chunk
        print(f"from prev # left most hist: {data_hist['lstd4']}, #right most hist: {data_hist['rstd4']}")
        # left_end_bin = []
        # right_end_bin = []
        data_hist, left_end_bin, right_end_bin = exp_hist(new_data_chunk, prev_avg, prev_std)
        theo_hist = std_hist(prev_size, current_size, left_end_bin, right_end_bin)
        expand = False
        while (data_hist['rstd4'] - theo_hist['rstd4']) > 1 or (data_hist['lstd4'] - theo_hist['lstd4']) > 1:
            expand = True
            print(f">>>>> expand the {i} chunk.")
            expand_count += 1
            difference_right_end = data_hist['rstd4'] - theo_hist['rstd4']
            difference_left_end = data_hist['lstd4'] - theo_hist['lstd4']
            # max_right_end = max(hist_data)
            # min_left_end = min(hist_data)
            if use_bootstrap is False:
                adjust_left_std = adjust_constant * std
                adjust_right_std = adjust_constant * std
                print("adjust_constant:", adjust_constant)
                adjust_constant = adjust_constant + 0.001
                print("adjust left std:", adjust_left_std)
                print("adjust right std:", adjust_right_std)

            if use_bootstrap is True:
                number_bootstrap_iteration = opt.iter
                adjust_left_std = bootstrap(left_end_bin, 'left', number_bootstrap_iteration)
                adjust_right_std = bootstrap(right_end_bin, 'right', number_bootstrap_iteration)
                print("adjust_left_std bootstrap:", adjust_left_std)
                print("adjust_right_std bootstrap:", adjust_right_std)
                print(" ")
            if difference_right_end > 0 and use_bootstrap is False:
                max_right_end = max_right_end + adjust_right_std
                max_expand = max_right_end
                new_data_chunk.append(max_right_end)
                #            hist_data.append(max_right_end)
                total_size += 1

            if difference_left_end > 0 and use_bootstrap is False:
                min_left_end = min_left_end - adjust_left_std
                min_expand = min_left_end
                new_data_chunk.append(min_left_end)
                #            hist_data.append(min_left_end)
                total_size += 1

            if difference_right_end > 0 and use_bootstrap is True:
                # max_right_end = prev_avg + 3 * prev_std + adjust_right_std
                max_right_end = adjust_right_std
                max_expand = max_right_end
                new_data_chunk.append(max_right_end)
                #            hist_data.append(max_right_end)
                total_size += 1

            if difference_left_end > 0 and use_bootstrap is True:
                # min_left_end = prev_avg - 3 * prev_std - adjust_left_std
                min_left_end = adjust_left_std
                new_data_chunk.append(min_left_end)
                min_expand = min_left_end
                #            hist_data.append(min_left_end)
                total_size += 1
            print("expand: min left end", min_left_end, "max right end",
                  max_right_end)
            current_list_size = len(new_data_chunk)
            avg = (prev_size / total_size) * prev_avg + \
                  (current_list_size / total_size) * mean(new_data_chunk)
            prev_avg = avg
            std = (max_right_end - min_left_end) / 8
            prev_size = total_size
            prev_std = std

            new_data_list_size = len(new_data_chunk)

            # # total_size = prev_size + current_size
            # # avg = (prev_size / total_size) * prev_avg + (current_size / total_size) * mean(new_data_chunk)
            # # prev_avg = avg
            # std = (max_right_end - min_left_end) / 8
            # prev_std = std

            data_hist, left_end_bin, right_end_bin = exp_hist(new_data_chunk, prev_avg, prev_std)
            theo_hist = std_hist(prev_size, current_size, left_end_bin, right_end_bin)
        if expand is False:
            total_size = prev_size + current_size
            avg = (prev_size / total_size) * prev_avg + (current_size / total_size) * mean(new_data_chunk)
            prev_avg = avg
            prev_size = total_size
    print('======================')
    print(f'# population is: {amount_population}')
    print(f'# expansion is: {expand_count}')
    print(f'pop avg: {mean(data):.2f} ')
    print(f'pop std: {stdev(data):.2f}')
    print(f'boostrap avg: {prev_avg:.2f}')
    print(f'boostrap std: {prev_std:.2f}')
    print(f'=================')
    # =====>  1. Generate population
    # randomlist = pd.read_csv(opt.data)
    # randomlist = pd.DataFrame.to_numpy(randomlist)
    # randomlist = randomlist.flatten()
    # input_size = 300
    # amount_population = len(randomlist)
    # print(f' size of population is: {amount_population}')
    # # start_interval_random_value = opt.stapt
    # # end_interval_random_value = opt.endpt
    # # input_size = opt.chsize
    # # number_bootstrap_iteration = opt.iter
    # # amount_population = opt.npop  # ---- must be <= start-end value
    # # randomlist = np.random.randint(start_interval_random_value, end_interval_random_value, amount_population)
    # # print("random list population:")
    # # print(f'Des. stat => pop. mean: {mean(randomlist):.2f}/ pop. sd: {stdev(randomlist):.2f} / pop. size: {len(randomlist)}.')
    # #




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    opt = parse_args()
    main(opt)
