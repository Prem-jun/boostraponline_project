
'''
Statistical distribution

'''
import math
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
