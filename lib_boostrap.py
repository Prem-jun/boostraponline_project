'''
   This the library for boostrap method 
'''

from dataclasses import dataclass, field
import statistics
import math
from typing import List
import pickle
import pandas as pd
import copy 
import numpy as np



class boostrap_v1:
    def bootstrap_online(input_data: list, end_side: str,
                         number_bootstrap_iteration: int = None,
                         minmax_boost: bool = None, prob: bool = None) -> float:
        # input_data: data for conduct boostrap
        # end_side: 'left' or 'right'
        # number_bootstrap_iteration: # of boostrap samples
        # minmax_boost: True or False for minmax boostraping
        # prob: True or False for samples selection with probability.
        
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
        
        
        
        bootstrap_sample_list = []
        
        if minmax_boost:
            bootstrap_sample_min_list = []
            bootstrap_sample_max_list = []
        
        # Setting boostrap size
        size_boost = size_data_set
            
        # Perform bootstrap sampling
        if prob:
            dist = abs(data_set-np.mean(data_set))
            idist = 1/dist
            pdist = idist/np.sum(idist)
            bootstrap_sample_list = [ list(np.random.choice(data_set, size_boost,replace=True,p=pdist))
                                            for i in range(number_bootstrap_iteration)
                                ]  
        else:
                
            # create list of boostrap samples.
            bootstrap_sample_list = [ list(np.random.choice(data_set, size_boost,replace=True))
                                            for i in range(number_bootstrap_iteration)
        previous_bootstrap_mean = 0                        ]  
        for idx,samples in enumerate(bootstrap_sample_list):
            present_bootstrap_mean = (np.mean(bootstrap_sample) +
                                      previous_bootstrap_mean)/2
            previous_bootstrap_mean = present_bootstrap_mean
            bootstrap_means[i] = present_bootstrap_mean
            if minmax_boost:
                bootstrap_sample_min_list.append(np.min(samples))
                bootstrap_sample_max_list.append(np.max(samples))
        # compute the mean of boostrap sample by        
        estimated_mean = np.mean(bootstrap_means)
        
        if prob:
          input_mean = sum(np.array(pdist)*np.array(input_data))
        else:
            input_mean = np.mean(input_data) # Aj Chid mean    
            
        # compute difference between the mean of current data sample and boostrap mean
        different_input_mean_bootstrap_mean = abs(input_mean - estimated_mean)
        
        # compute the boostrap std value.
        for data in bootstrap_sample_list:
            variance = [(k-estimated_mean)**2 for k in data]
            std = math.sqrt(sum(variance)/(size_boost-1))
            bootstrap_std.append(std)
            
        estimated_std = np.mean(bootstrap_std)
        different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data,input_mean) - estimated_std)
        
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
    
                # print("\n>>>>> Bootstrap in left end: estimated_mean < input_mean")
                # print("left min:", min(data_set), " final_estimated_std:", \
                #       final_estimated_std)
                
            if input_mean < estimated_mean:
                
                # left_min = min(data_set)
    #            final_estimated_std = left_min - estimated_std_of_std
                final_estimated_std = left_min - different_input_std_bootstrap_std
                # print("\n>>>>> Bootstrap in left end: input_mean < estimated_mean")
                # print("left min:", min(data_set), " final_estimated_std:", \
                #       final_estimated_std)
        
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
                # print("\n>>>>> Bootstrap in right end: estimated_mean < input_mean")
                # print("right max:", max(data_set), " final_estimated_std:", \
                #       final_estimated_std)
                                      
            if input_mean < estimated_mean:
                # right_max = max(data_set)
                final_estimated_std = right_max + different_input_mean_bootstrap_mean            
                # print("\n>>>>> Bootstrap in right end: input_mean < estimated_mean")
                # print("right max:", max(data_set), " final_estimated_std:", \
                #       final_estimated_std)
    
        # print(" ")
        # print("final_estimated_std:", final_estimated_std)
    
        return final_estimated_std
        
        
        
        
        
        
        
        # for i in range(number_bootstrap_iteration):
     
        #     if prob:
                
        #         # bootstrap_sample = [ np.random.choice(data_set, 1,replace=True,p=pdist) for i in range(size_boost)]
        #         bootstrap_sample = list(np.random.choice(data_set, size_boost,replace=True,p=pdist))
        #     else:
        #         # bootstrap_sample = [ np.random.choice(data_set, 1,replace=True) for i in range(size_boost)]
        #         bootstrap_sample = list(np.random.choice(data_set, size_boost,replace=True))
            
            
        #     # bootstrap_sample = list(np.random.choice(data_set, size_data_set, \
        #     #                                           replace = with_replacement))
        #     # bootstrap_sample = list(random.sample)
        #     bootstrap_sample_list.append(bootstrap_sample)
        #     if minmax_boost:
        #         bootstrap_sample_min_list.append(min(bootstrap_sample))
        #         bootstrap_sample_max_list.append(max(bootstrap_sample))
            
        #     present_bootstrap_mean = (np.mean(bootstrap_sample) +
        #                               previous_bootstrap_mean)/2
        #     previous_bootstrap_mean = present_bootstrap_mean
        #     bootstrap_means[i] = present_bootstrap_mean
    
        # estimated_mean = np.mean(bootstrap_means)
        # # estimated_std_of_mean = boostrap_v1.stdev(bootstrap_means,estimated_mean)
        # if prob:
        #   input_mean = sum(np.array(pdist)*np.array(input_data))
        # else:
        #     input_mean = np.mean(input_data) # Aj Chid mean
        
        # # different_input_mean_bootstrap_mean = abs(boostrap_v1.mean(input_data) - estimated_mean)
        # different_input_mean_bootstrap_mean = abs(input_mean - estimated_mean)
        
        # for i in range(number_bootstrap_iteration):
        #     data = bootstrap_sample_list[i]
        #     variance = [(k-estimated_mean)**2 for k in data]
        #     std = math.sqrt(sum(variance)/(size_boost-1))
        #     # std = 0
        #     # for j in range(size_boost):
        #     #     std = std + (data[j] - estimated_mean)**2
        #     # std = math.sqrt(std/(size_boost-1))
        #     # # diff_dist_right_std = max(data) - (std + estimated_mean)
        #     # # diff_dist_left_std = (estimated_mean - std) - min(data)
        #     # # max_std = (diff_dist_right_std + diff_dist_left_std)/2
        #     bootstrap_std.append(std)
        #     # bootstrap_max_diff_dist_std.append(max_std)
    
        # estimated_std = np.mean(bootstrap_std)
        # # estimated_std_of_std = boostrap_v1.stdev(bootstrap_std,estimated_std)
        # different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data,input_mean) - estimated_std)
        # # different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data,) - estimated_std)
    
        # # estimated_diff_dist_std = np.mean(bootstrap_max_diff_dist_std)
        # # estimated_std_of_diff_dist_std = np.std(bootstrap_max_diff_dist_std)
    
        
        # # input_std = boostrap_v1.stdev(input_data) No use in the below lines.
    
        # # print("=====================================================================")
        # # print(">>>>> in Bootstrap")
        # # print("\n----- ", end_side)
        # # print("left min:", min(data_set))
        # # print("right max:", max(data_set))
        # # print("\ninput mean:", input_mean)
        # # print("estimated_mean:", estimated_mean)
        # # print("estimated_std_of_mean:", estimated_std_of_mean)
        # # print(" ")
        # # print("input std:", input_std)
        # # print("estimated_std:", estimated_std)
        # # print("estimated_std_of_std:", estimated_std_of_std)
        # # print(" ")
        # # print("estimated_diff_dist_std:", estimated_diff_dist_std)
        # # print("estimated_std_of_diff_dist_std:", estimated_std_of_diff_dist_std)
    
        # # input_mean = np.mean(input_data)
    
    #     if end_side == "left":
    #         if minmax_boost is True:
    #             if prob:
    #                 dist_min = abs(bootstrap_sample_min_list-np.mean(bootstrap_sample_min_list))
    #                 idist_min = 1/dist_min
    #                 pdist_min = idist_min/np.sum(idist_min)
    #                 left_min_1 = sum(np.array(pdist_min)*np.array(bootstrap_sample_min_list))
    #                 left_min = left_min_1
    #             else:
    #                 left_min = np.mean(bootstrap_sample_min_list)
    #             # left_min = np.mean(bootstrap_sample_min_list)#-np.std(bootstrap_sample_min_list)
    #             # left_min = st.mode(np.array(bootstrap_sample_min_list))[0][0]
    #         else:
    #             left_min = min(data_set)
    #         if estimated_mean < input_mean:
    #             # left_min = min(data_set)
    #             final_estimated_std = left_min - different_input_mean_bootstrap_mean
    # #            final_estimated_std = left_min - estimated_std_of_diff_dist_std
    #             print("\n>>>>> Bootstrap in left end: estimated_mean < input_mean")
    #             print("left min:", min(data_set), " final_estimated_std:", \
    #                   final_estimated_std)
                
    #         if input_mean < estimated_mean:
                
    #             # left_min = min(data_set)
    # #            final_estimated_std = left_min - estimated_std_of_std
    #             final_estimated_std = left_min - different_input_std_bootstrap_std
    #             print("\n>>>>> Bootstrap in left end: input_mean < estimated_mean")
    #             print("left min:", min(data_set), " final_estimated_std:", \
    #                   final_estimated_std)
        
    #     if end_side == "right":
    #         if minmax_boost is True:
    #             if prob:
    #                 dist_max = abs(bootstrap_sample_max_list-np.mean(bootstrap_sample_max_list))
    #                 idist_max = 1/dist_max
    #                 pdist_max = idist_max/np.sum(idist_max)
    #                 right_max_1 = sum(np.array(pdist_max)*np.array(bootstrap_sample_max_list))
    #                 right_max = right_max_1
    #             else:
    #                 right_max = np.mean(bootstrap_sample_max_list)
    #             # right_max = np.mean(bootstrap_sample_max_list)#+np.std(bootstrap_sample_max_list)
    #             # right_max = st.mode(np.array(bootstrap_sample_max_list))[0][0]
    #         else:
    #             right_max = max(data_set)
    #         if estimated_mean < input_mean:
    #             # right_max = max(data_set)
    # #            final_estimated_std = right_max + estimated_std_of_std
    #             final_estimated_std = right_max + different_input_std_bootstrap_std
    #             print("\n>>>>> Bootstrap in right end: estimated_mean < input_mean")
    #             print("right max:", max(data_set), " final_estimated_std:", \
    #                   final_estimated_std)
                                      
    #         if input_mean < estimated_mean:
    #             # right_max = max(data_set)
    #             final_estimated_std = right_max + different_input_mean_bootstrap_mean            
    #             print("\n>>>>> Bootstrap in right end: input_mean < estimated_mean")
    #             print("right max:", max(data_set), " final_estimated_std:", \
    #                   final_estimated_std)
    
    #     print(" ")
    #     print("final_estimated_std:", final_estimated_std)
    
    #     return final_estimated_std    
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


@dataclass
class booststream:
    online:bool = False # Online boostrap exploited.
    minmax_boost:bool = False # Minmax boostrap applied. 
    # addnoise:bool = False # Addnoise in simulate data
    # prob:bool = False # prob-based sample selection in boostrap applied.
    filesampl:str = ''
    numbin:int = 0
    number_bt_iter:int = 0
    nboost:int = 0 # minimum number of conducting boostrap.
    dist_list:List[str] = field(default_factory=list)
    total_size:int = -1
    chunk_size:float = 0.0
    max_chs:float = 0.0
    min_chs:float = 0.0
    min_list:List[str] = field(default_factory=list) 
    max_list:List[str] = field(default_factory=list)
    avg = List[str] = field(default_factory=list)
    std = List[str] = field(default_factory=list)
    exp_l:float = 0.0
    exp_r:float = 0.0
    range:float = 0.0
    
    # filewd: dict = ''
    # pop_max: float = 0.0
    # pop_min: float = 0.0
    # ch_size: int = 0 # # data points per chunk.
    # feed_percent: int = 0 # portion of data used as samples from entire dataset. 
    
    
    def set_online(self,minmax_flag:bool=False):
        dist_list = ['exponweib', 'wald', 'gamma', 'norm',\
                         'expon', 'powerlaw', 'lognorm', 'chi2', 'weibull_min',\
                         'weibull_max']
        self.online = True
        self.minmax_boost = minmax_flag
        self.numbin = 8
        self.number_bt_iter = 600
        self.dist_list = dist_list
        self.nboost = 3
        self.total_size  = 0
        self.max_chs = - 9999.99
        self.min_chs = 9999.99
        self.exp_l = 9999.99
        self.exp_r = -9999.99
    
    def update_center_range(self,leftmost: float,rightmost: float)-> None:
        self.avg.append((rightmost+leftmost)/2)
        self.std.append((rightmost-leftmost)/8)
        
    def expand_bt_online(self,new_data_chunk:list) -> None:
        try: 
            if self.online is False:
                raise ValueError("The network in traditional mode. Can not perform online mode.")
            
        except ValueError as e:
            return print(f"Error: {e}")
        
        self.total_size += len(new_data_chunk)
        new_data_chunk_min = min(new_data_chunk)
        new_data_chunk_max = max(new_data_chunk)
        if new_data_chunk_min < self.exp_l:
            expand_min = True
            if len(self.min_list) >= self.nboost and (self.minmax_boost is True):
                self.min_list.append(new_data_chunk_min)
                # expand_min = True
                adjust_left_std = boostrap_v1.bootstrap_online(self.min_list, "left",\
                                                               number_bootstrap_iteration = self.number_bt_iter, \
                                                                   minmax_boost = self.minmax_boost,\
                                                                       prob = False)
                if self.exp_l >= adjust_left_std:
                    self.exp_l = adjust_left_std      
            else:
                # expand_min = True
                self.exp_l = new_data_chunk_min
                
        if new_data_chunk_max > self.exp_r:
            expand_max = True
            if (len(self.max_list) >= self.nboost) and (self.minmax_boost is True):
                self.max_list.append(new_data_chunk_max)
                adjust_right_std = boostrap_v1.bootstrap_online(self.max_list, "right",\
                                                                number_bootstrap_iteration = self.number_bt_iter, \
                                                                    minmax_boost = self.minmax_boost,\
                                                                        prob = False)
                
        #         expand_max = True
        #         # expansion = True
                if self.exp_r <= adjust_right_std:
                    self.exp_r = adjust_right_std
            else:
        #         expand_max = True
        #         # expansion = True
                self.exp_r = new_data_chunk_max
                
        if expand_min is True or expand_max is True:
            self.update_center_range(self.exp_l,self.exp_r)
            avg = self.avg[-1]
            std = self.std[-1]
            end_bin_left = []
            end_bin_right = []
            new_data_chunk = new_data_chunk + self.min_list + self.max_list
            self.min_list = [k for k in new_data_chunk if (avg - 4*std <= k <= avg - 3*std)]
            self.max_list = [k for k in new_data_chunk if (avg + 3*std <= k <= avg + 4*std)]
            hist_data = [0] * int(self.numbin) # initialize experimental histogram    
            hist_theo = [0] * int(self.numbin) # initialize theoritical histogram
        #     hist_data[0] = len(end_bin_left) # left most bin
        
        #     hist_data[-1] = len(end_bin_right) # right most bin
        #     hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
        #     hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
        #     percent_data = boostrap_v1.get_percent_std_data_from_best_distribution(\
        #                                        self.total_size, self.endL, self.endR, \
        #                                        self.dist_list)
        #     hist_theo = [math.ceil(i*self.total_size/100.0) for i in percent_data]
        #     self.endLn.append(len(self.endL))
        #     self.endRn.append(len(self.endR))
        #     expand = False
        #     expansion = False
        #     difference_max = hist_data[-1] - hist_theo[-1]
        #     difference_min = hist_data[0] - hist_theo[0]
        #     # while (difference_max > 0 or difference_min > 0):
        #     if (difference_max > 0 or difference_min > 0):
        #         dif_expand = True    
        #     else:
        #         dif_expand = False    
        #     while dif_expand is True:    
        #         # difference_max_tmp = difference_max
        #         # difference_min_tmp = difference_min
        #         expandL = self.expandL
        #         expandR = self.expandR
        #         if difference_max > 0:
            
        #             # if use_bootstrap is True:
        #             if hist_data[-1] >= self.nboost:
        #                 if self.expandR <= max(self.endR):
        #                     self.expandR = boostrap_v1.bootstrap_online(self.endR, "right",\
        #                                                                 number_bootstrap_iteration = self.number_bt_iter, \
        #                                                                   minmax_boost = self.minmax_boost)
        #                     # self.expandR = max_right_end
        #                     expand = True
        #                     expansion = True
                        
        #         if difference_min > 0:
        #             if hist_data[-1] >= self.nboost:
        #                 if self.expandL >= min(self.endL):
        #                     self.expandL = boostrap_v1.bootstrap_online(self.endL, "left",\
        #                                                                 number_bootstrap_iteration = self.number_bt_iter, \
        #                                                                     minmax_boost = self.minmax_boost)
        #                     # self.expandL = min_left_end
        #                     expand = True
        #                     expansion = True
        #         if expand is True:
        #             self.update_center_range(self.expandL,self.expandR)
        #             avg = self.avg[-1]
        #             std = self.std[-1]
        #             expand = False
        #             hist_data = [0] * int(self.numbin)
        #             end_bin_left = [] # add
        #             end_bin_right = [] # add
        #             end_bin_left = [i for i in new_data_chunk if (avg - 4*std <= i <= avg - 3*std)]
        #             end_bin_right = [i for i in new_data_chunk if (avg + 3*std <= i <= avg + 4*std)]
        #             # end_bin_num_left.append(len(end_bin_left)) 
        #             # end_bin_num_right.append(len(end_bin_right))
        #             hist_data[0] = len(end_bin_left)
        #             hist_data[-1] = len(end_bin_right)
        #             hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
        #             hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
        #             # count += 1
        #             # boostrap_v1.plot_histogram(hist_theo[0],hist_theo[1], hist_theo[2],hist_theo[3],\
        #             #                    hist_theo[4],hist_theo[5], hist_theo[6],hist_theo[7],\
        #             #                        hist_data[0],hist_data[1], hist_data[2],hist_data[3],\
        #             #                            hist_data[4],hist_data[5], hist_data[6],hist_data[7],\
        #             #                                min_expand, max_expand, 1, expand)
        #             # plt.show()
        #             # difference_max = hist_data[-1] - hist_theo[-1]
        #             # difference_min = hist_data[0] - hist_theo[0]
        #             self.endL = end_bin_left
        #             self.endR = end_bin_right
        #             self.endLn.append(len(end_bin_left))
        #             self.endRn.append(len(end_bin_right))
                    
                    
        #             # if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
        #             if expandL == self.expandL and expandR == self.expandR:     
        #                 # if self.expandL < new_data_chunk_min and new_data_chunk_max < self.expandR:
        #                 #     # tmp_leftright = False
        #                 #     break
        #                 dif_expand = False
                
                        
        #         # else:
                    
        #         #     if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
        #         #         # tmp_chunk_end  = end_bin_left + end_bin_right
        #         #         # tmp_leftright = True
        #         #         break
        # if (expansion is True) or (expand_max is True) or (expand_min is True):
        #    self.range = self.expandR - self.expandL
        #    self.le_samp.append(self.min_chs - self.expandL)
        #    self.re_samp.append(self.max_chs - self.expandR)
        #    self.le_pop.append(self.pop_min - self.expandL)
        #    self.re_pop.append(self.pop_max - self.expandR)
        #    expansion = True
        # return expansion          
        
    def expand_bt_trad(self,input_data:list) -> None:
        # Traditional boostrap method
        try: 
            if self.online is True:
                raise ValueError("The network in online mode. Can not perform whole mode.")
            
        except ValueError as e:
            return print(f"Error: {e}")
        data_set = copy.deepcopy(input_data)
        bootstrap_min = []
        bootstrap_max = []
        bootstrap_std = []
        size_boost = len(data_set)
        input_mean  = np.mean(data_set)  
        # create list of boostrap samples.
        bootstrap_sample_list = [ list(np.random.choice(data_set, size_boost,replace=True))
                                        for i in range(self.number_bt_iter)
                            ]  
        for idx,samples in enumerate(bootstrap_sample_list):
            bootstrap_min.append(np.min(samples))
            bootstrap_max.append(np.max(samples))
        
        self.chunk_size = size_boost
        self.max_chs = np.max(data_set)
        self.min_chs = np.min(data_set)
        self.exp_l = np.mean(bootstrap_min)
        self.exp_r = np.mean(bootstrap_max)
        self.range = self.exp_r - self.exp_l    
            # if idx == 0:
            #     bootstrap_min.append(np.min(samples))
            #     previous_bootstrap_mean = bootstrap_means[0]
            # else:
            #     bootstrap_means.append(
            #     0.5*(np.mean(samples) + previous_bootstrap_mean)
            #     )
            #     previous_bootstrap_mean = bootstrap_means[-1]
    
    def expand_whole(self,input_data:list) -> None:
        try: 
            if self.online is True:
                raise ValueError("The network in online mode. Can not perform whole mode.")
            
        except ValueError as e:
            return print(f"Error: {e}")
        data_set = copy.deepcopy(input_data)
        bootstrap_means = []
        bootstrap_std = []
        size_boost = len(data_set)
        input_mean  = np.mean(data_set)
        
        # create list of boostrap samples.
        bootstrap_sample_list = [ list(np.random.choice(data_set, size_boost,replace=True))
                                        for i in range(self.number_bt_iter)
                            ]
        
        for idx,samples in enumerate(bootstrap_sample_list):
            if idx == 0:
                bootstrap_means.append(np.mean(samples))
                previous_bootstrap_mean = bootstrap_means[0]
            else:
                bootstrap_means.append(
                0.5*(np.mean(samples) + previous_bootstrap_mean)
                )
                previous_bootstrap_mean = bootstrap_means[-1]
        
        # compute the boostrap mean estimation
        estimated_mean = np.mean(bootstrap_means)
        # compute the difference between the input mean and the boostrap mean estimation
        different_input_mean_bootstrap_mean = abs(input_mean - estimated_mean)
        
        est_mean_list = list(estimated_mean)*size_boost
        for data in bootstrap_sample_list:
            variance = list(map(lambda a,b: (a-b)**2,data,est_mean_list))
            std = math.sqrt(sum(variance)/(size_boost-1))
            bootstrap_std.append(std)
        estimated_std = np.mean(bootstrap_std)
        different_input_std_bootstrap_std = abs(statistics.stdev(data_set) - estimated_std)
    
    
    
    # exp_r: float = 0.0 # upper bound
    # exp_l: float = 0.0 # lower boun
    # exp_range: float = 0.0  # range 
    # expandR: List[float] = field(default_factory=list) # list of right expand values  
    # expandL: List[float] = field(default_factory=list) # list of right expand values  
    # total_size: int = 0 # number of learned ldata
    # # range: float = 0.0 # max-min
    # # endL: list = [] # list of data outside the left bound.
    # # endR: list = [] # list of data outside the right bound.
    # # endLn: list = [] # list of number of data outside the left bound.
    # # endRn: list = [] # list of number of data outside the right bound.
    # # min_chs: int = -1 # chunk number containing the minimum value.
    # # max_chs: int = -1
    
    # # avg: list = [] # list of total average values.
    # # std: list = [] # list of total std values.
    # # re_samp: list = [] # right error from sample set ()
    # # le_samp: list = [] # left error from sample set ()
    # # re_pop: list = [] # right error from population set ()
    # # le_pop: list = [] # left error from population set ()
    # # expandch: list = [] # chunk number with expanded left or right end

# https://www.theknowledgeacademy.com/blog/python-dataclass/

@dataclass
class samp1d():
    file_config:str # .yaml config file.
    #pop_sim: List[float] = field(default_factory=list) # list of poppulation values.     
    nsim:int = 0
    #shape: float = 0.0
    name:str = None
    chunk_size: int = 0
    percent_feed: int = 30
    samp_chuck: List[float] = field(default_factory=list) # list of samples values.     
    
    def split2chunk(self,pop_sim:list,chunk_size:int = None):
        if chunk_size is None:
            chunk_size  = 50
        self.chunk_size = chunk_size
        pop_n = len(pop_sim)
        samp_num = math.ceil((self.percent_feed/100)*pop_n)
        num_ch = math.floor(samp_num/self.chunk_size)
        list_chunk = [pop_sim[i:i + self.chunk_size] for i in range(0,(self.chunk_size*(num_ch-1)),self.chunk_size)]
        list_chunk.append(pop_sim[(self.chunk_size*(num_ch-1)):(self.chunk_size*(num_ch-1)) + self.chunk_size])
        self.samp_chuck = list_chunk

net1 = booststream()
net1.set_online()
net1.expand_whole()
# print(net1) 
