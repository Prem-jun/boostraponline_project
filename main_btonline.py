#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:38:24 2024

@author: premjunsawang
"""
import numpy as np
import pandas as pd
# import boostraponline
from boostraponline import stat_dist, managefile, boostrap_v1 
import math
import matplotlib.pyplot as plt
import os
import random, xlsxwriter, pickle

def main(filewd = None,filename = None,nboost = None,boostkeep = None):
    if filewd is None:
        filewd = "./sim_data/"
    if filename is None:
        filename = 'wiebullshape5n20000-20'
    if nboost == None:
        nboost = 3
    if boostkeep == None:
        boostkeep = False
    randomlist,feed_ch,pop_max,pop_min,samp_max,samp_min = pickle.load(open(filewd+filename+".p","rb"))    
    numbin = 8
    usestd = True
    # if sheet_name is None:
    #     sheet_name = 0
        
    # if input_size is None:
    #     input_size = 300
    # if nboost == None:
    #     nboost = 3
    # if boostkeep == None:
    #     boostkeep = True
    
    
    # numbin = 8
    # usestd = True
    # name,extension = os.path.splitext(filename)
    # if extension == '.xlsx':
    #    df = pd.read_excel(filewd+filename,sheet_name = sheet_name) 
    # if extension == '.csv':
    #    df = pd.read_csv(filewd+filename) 
    
    # randomlist = pd.DataFrame.to_numpy(df['V1'])
    # randomlist = randomlist.flatten().tolist()   
    # amount_population = len(randomlist)
    
    # number_data_chunk = int(amount_population//input_size)
    # print("number data chunk:", number_data_chunk)    
    # chunk_percent = 35    
    # num_exp_chunk = math.ceil((chunk_percent/100)*number_data_chunk)
    # df = pd.DataFrame()
    # list_exp_chunk=[]
    # # list_ch_num = []

    # for k in range(num_exp_chunk):
    #     # list_ch_num = list_ch_num+[k]*input_size
    #     list_exp_chunk.append(randomlist[k*input_size:((k+1)*input_size)])
        
    # list_ch_all = list_exp_chunk+[randomlist[(k*num_exp_chunk+1):amount_population]]
    # # fig, ax = plt.subplots(1, 1)
    # # fig.figure(figsize =(len(list_ch_all), len(list_ch_all)+2))
    # # fig.add_axes([0, 0, 1, 1])
    # # ax.boxplot(list_exp_chunk)
    # list_ch_all = feed_ch+randomlist 
    _fig = plt.figure(figsize =(len(feed_ch), len(feed_ch)+2))
    _ax = _fig.add_axes([0, 0, 1, 1])
    _bp = _ax.boxplot(feed_ch)
    plt.show()
    
    
    # # df = pd.DataFrame({'value':randomlist,'Ch-no':list_ch_num})
    # # df.boxplot(by ='Ch-no', column =['value'], grid = False) 
    # if boostkeep  is True:
    #     file_stat_name = name+'_'+sheet_name+'_Keep'
    # else:
    #     file_stat_name = name+'_'+sheet_name
        
    # #listname = ['lstd4', 'lstd3', 'lstd2', 'lstd1', 'rstd1', 'rstd2', 'rstd3', 'rstd4']
    hist_data = [0] * int(numbin)
    # data_hist = dict(zip(listname, data_hist)) 
    #listname = ['slstd4', 'slstd3', 'slstd2', 'slstd1', 'srstd1', 'srstd2', 'srstd3', 'srstd4']
    hist_theo = [0] * int(numbin)
    # theo_hist = dict(zip(listname, theo_hist)) 
      
    list_size = len(randomlist)
    # all_min = min(randomlist)
    # all_max = max(randomlist)
    all_min = pop_min
    all_max = pop_max
    expand_chunk_list = []
    end_bin_num_left = []
    end_bin_num_right = []
    
    distribution_list = ['exponweib', 'wald', 'gamma', 'norm',\
                     'expon', 'powerlaw', 'lognorm', 'chi2', 'weibull_min',\
                     'weibull_max']
    #----------------------- first chunk ---------------------------------------
    print("------ The 1 st chunk-------------")
    
    # new_input = randomlist[0:(input_size)]
    new_input = feed_ch[0]
    if usestd is True:
        max_new_input = max(new_input)
        min_new_input = min(new_input)
        std = (max_new_input - min_new_input)/8
        avg = (max_new_input + min_new_input)/2
        print("std_interval ", std, "average ", avg, "min ", min_new_input, "max", \
              max_new_input)
    else:
        print('no std used.')
        return 0
    total_size = len(new_input)
    print("current size", total_size)
    # ------------
    end_bin_left = [i for i in new_input if (avg - 4*std <= i <= avg - 3*std)]
    end_bin_right = [i for i in new_input if (avg + 3*std <= i <= avg + 4*std)]
    end_bin_num_left.append(len(end_bin_left)) 
    end_bin_num_right.append(len(end_bin_right))     
    hist_data[0] = len(end_bin_left)
    hist_data[-1] = len(end_bin_right)
    hist_data[1] = len([i for i in new_input if (avg - 3*std <= i <= avg - 2*std)])
    hist_data[-2] = len([i for i in new_input if (avg + 2*std <= i <= avg + 3*std)])
    print(f"# end_bin_left and right: {hist_data[0]} and {hist_data[-1]}")
    '''
    left_end_bin = []
    right_end_bin = []
    
    for l in range(total_size):
        if avg - 4*std <= new_input[l] <= avg - 3*std:
            lstd4 = lstd4 + 1
            left_end_bin.append(new_input[l])
    
        if avg + 3*std <= new_input[l] <= avg + 4*std:
            rstd4 = rstd4 + 1
            right_end_bin.append(new_input[l])
    '''        
    # ---------------
    percent_data = boostrap_v1.get_percent_std_data_from_best_distribution(\
                                           total_size, end_bin_left, end_bin_right, \
                                           distribution_list)
    # ----------------
    hist_theo = [math.ceil(i*total_size/100.0) for i in percent_data] 
    '''
    slstd4 = math.ceil(percent_data_lstd4*total_size/100.0)
    slstd3 = math.ceil(percent_data_lstd3*total_size/100.0)
    slstd2 = math.ceil(percent_data_lstd2*total_size/100.0)
    slstd1 = math.ceil(percent_data_lstd1*total_size/100.0)
    srstd1 = math.ceil(percent_data_rstd1*total_size/100.0)
    srstd2 = math.ceil(percent_data_rstd2*total_size/100.0)
    srstd3 = math.ceil(percent_data_rstd3*total_size/100.0)
    srstd4 = math.ceil(percent_data_rstd4*total_size/100.0)    
    '''
    #------------------
    boostrap_v1.plot_histogram(hist_theo[0],hist_theo[1], hist_theo[2],hist_theo[3],\
                               hist_theo[4],hist_theo[5], hist_theo[6],hist_theo[7],\
                                   hist_data[0],hist_data[1], hist_data[2],hist_data[3],\
                                       hist_data[4],hist_data[5], hist_data[6],hist_data[7],\
                                           min_new_input, max_new_input, 1, 0)
    
    #----------------------------------------------------------------------------
    # check if violate standard histogram 
    # new_hist_width is an interval for building a histogram. b is adjustted
    # to fit standard histogram
    #-----------------------------------------------------------------------------

    max_right_end = max(new_input)
    min_left_end = min(new_input)
    
    print(f"----- first chunk min left end: {min_left_end:.2f} max_right_end: {max_right_end:.2f}")
    
    avg =  (max_new_input + min_new_input)/2
    std = (max_right_end - min_left_end)/8
    count = 1
    expand = False
    expansion = False
    
    use_bootstrap = True
    min_expand = 0
    max_expand = 0
    
    adjust_left_std = avg
    adjust_right_std = 0
    
    max_right_end = 0
    min_left_end = max(new_input)    
    difference_max = hist_data[-1] - hist_theo[-1]
    difference_min = hist_data[0] - hist_theo[0]
    while (difference_max > 0 or difference_min > 0):
        difference_max_tmp = difference_max
        difference_min_tmp = difference_min
        if difference_max > 0:
            if use_bootstrap is True:
                if len(end_bin_right) >= nboost:
                    if max_right_end <= max(end_bin_right):
                        adjust_right_std = boostrap_v1.bootstrap_online(end_bin_right, "right")
                        max_right_end = adjust_right_std
                        max_expand = max_right_end
                        print(f">>> go to Bootstrap right end bin: {max_right_end:.2f}")
                        expand = True
                        expansion = True
                
        if difference_min > 0:
            if use_bootstrap is True:
                if len(end_bin_left) >= nboost:
                    if min_left_end >= min(end_bin_left):
                        adjust_left_std = boostrap_v1.bootstrap_online(end_bin_left, "left")
                        min_left_end = adjust_left_std
                        min_expand = min_left_end
                        print("adjust_left_std bootstrap:", adjust_left_std)
                        expand = True
                        expansion = True
        
        
    
        if expand is True:
            print(f"expand: min left end {min_left_end:.2f} max right end {max_right_end:.2f}")
            avg = (max_right_end + min_left_end)/2
            std = (max_right_end - min_left_end)/8
            expand = False
            hist_data = [0] * int(numbin)
            #-------- modify : delete duplicate in left right end bin -------------------
            #--------------- recount lstd4 rstd4 ----------------------------------------
            end_bin_left = [] # add
            end_bin_right = [] # add
            end_bin_left = [i for i in new_input if (avg - 4*std <= i <= avg - 3*std)]
            end_bin_right = [i for i in new_input if (avg + 3*std <= i <= avg + 4*std)]
            end_bin_num_left.append(len(end_bin_left)) 
            end_bin_num_right.append(len(end_bin_right))
            hist_data[0] = len(end_bin_left)
            hist_data[-1] = len(end_bin_right)
            hist_data[1] = len([i for i in new_input if (avg - 3*std <= i <= avg - 2*std)])
            hist_data[-2] = len([i for i in new_input if (avg + 2*std <= i <= avg + 3*std)])
            count += 1
            boostrap_v1.plot_histogram(hist_theo[0],hist_theo[1], hist_theo[2],hist_theo[3],\
                               hist_theo[4],hist_theo[5], hist_theo[6],hist_theo[7],\
                                   hist_data[0],hist_data[1], hist_data[2],hist_data[3],\
                                       hist_data[4],hist_data[5], hist_data[6],hist_data[7],\
                                           min_expand, max_expand, 1, expand)
            plt.show()
            difference_max = hist_data[-1] - hist_theo[-1]
            difference_min = hist_data[0] - hist_theo[0]
            if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
                if min_left_end < min_new_input and max_new_input < max_right_end:
                    tmp_leftright = False
                    break
            # if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
            #     end_bin_left.append(min_expand)
            #     end_bin_right.append(max_expand)
            # else: # add 
            #     if min_left_end < min_new_input and max_new_input < max_right_end:
            #         tmp_leftright = False
            #         break
        else:
            if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
                tmp_chunk_end  = end_bin_left + end_bin_right
                tmp_leftright = True
                break
    if expansion is True:
        expand_chunk_list.append(1)
    else:
        print('No expansion in the 1 st chunk')
        
    '''
    ----------------------------------------------------------------------------
        need max_right_end,  min_left_end, hist_avg, hist_size
        left_end_slot_data, right_end_slot_data
        new data chunk
    ----------------------------------------------------------------------------
    '''
    print("=============================================================")
    print("\n--------- get next chunk")
    num_exp_chunk = len(feed_ch)    
    for i in range(1,num_exp_chunk): 
    # for i in feed_ch:    
        print("\n\n------------------- new chunk", i+1, "----------------------")
        # k = i*input_size
        new_data_chunk = []
        # # end_bin_left = []
        # # end_bin_right = []
        # new_data_chunk = randomlist[k:(k+input_size)]
        new_data_chunk = feed_ch[i]
        if boostkeep is True:
            if tmp_leftright is True:
                new_data_chunk = new_data_chunk + tmp_chunk_end
                tmp_chunk_end = []
        
        # # end_bin_left_tmp = [k for k in new_data_chunk if (avg - 4*std <= k <= avg - 3*std)]
        # # end_bin_right_tmp = [k for k in new_data_chunk if (avg + 3*std <= k <= avg + 4*std)]
        
        # # new_data_chunk_min = random.choice(end_bin_left_tmp)
        # # new_data_chunk_max = random.choice(end_bin_right_tmp)
        
        # new_data_chunk_min = min(new_data_chunk)
        # new_data_chunk_max = max(new_data_chunk)
        new_data_chunk_min = samp_min[i]
        new_data_chunk_max = samp_max[i]
        print(f"present min and max data in new_data_chunk: {new_data_chunk_min:.2f} and {new_data_chunk_max:.2f}")
        expand_max = False
        expand_min = False
        if new_data_chunk_min < min_left_end:
            # expand_min = True
            if len(end_bin_left) >= nboost:
                end_bin_left.append(new_data_chunk_min)
                expand_min = True
                adjust_left_std = boostrap_v1.bootstrap_online(end_bin_left, "left")
                if min_left_end >= adjust_left_std:
                    min_left_end = adjust_left_std
                else:
                    min_left_end = min(new_data_chunk)
            else:
                min_left_end = min(new_data_chunk)
                # expand_min = True
            
        if new_data_chunk_max > max_right_end:
            # expand_max = True
            if len(end_bin_right) >= nboost:
                end_bin_right.append(new_data_chunk_max)
                adjust_right_std = boostrap_v1.bootstrap_online(end_bin_right, "right")
                expand_max = True
                if max_right_end <= adjust_right_std:
                    max_right_end = adjust_right_std
                else:
                    max_right_end = max(new_data_chunk)
            else:
                max_right_end = max(new_data_chunk)
                # expand_min = True
            
        if expand_min is True or expand_max is True:
            avg = (max_right_end + min_left_end)/2
            std = (max_right_end - min_left_end)/8
        
        total_size = total_size + len(new_data_chunk)
        end_bin_left = []
        end_bin_right = []
        end_bin_left = [k for k in new_data_chunk if (avg - 4*std <= k <= avg - 3*std)]
        end_bin_right = [k for k in new_data_chunk if (avg + 3*std <= k <= avg + 4*std)]
        end_bin_num_left.append(len(end_bin_left)) 
        end_bin_num_right.append(len(end_bin_right))     
        hist_data[0] = len(end_bin_left)
        hist_data[-1] = len(end_bin_right)
        hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
        hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
        percent_data = boostrap_v1.get_percent_std_data_from_best_distribution(\
                                           total_size, end_bin_left, end_bin_right, \
                                           distribution_list)
        hist_theo = [math.ceil(k*total_size/100.0) for k in percent_data] 
        
        expand = False
        expansion = False
        difference_max = hist_data[-1] - hist_theo[-1]
        difference_min = hist_data[0] - hist_theo[0]
        if (difference_max > 0 or difference_min > 0):
            while (difference_max > 0 or difference_min > 0):
                difference_max_tmp = difference_max
                difference_min_tmp = difference_min
                if difference_max > 0:
                    if use_bootstrap is True:
                        if len(end_bin_right) >= nboost:
                            if max_right_end <= max(end_bin_right):
                                adjust_right_std = boostrap_v1.bootstrap_online(end_bin_right, "right")
                                max_right_end = adjust_right_std
                                max_expand = max_right_end
                                print(f">>> go to Bootstrap right end bin: {max_right_end:.2f}")
                                expand = True
                                expansion = True
    
                    
                        
                if difference_min > 0:
                    if use_bootstrap is True:
                        if len(end_bin_left) >= nboost:
                            if min_left_end >= min(end_bin_left):
                                adjust_left_std = boostrap_v1.bootstrap_online(end_bin_left, "left")
                                min_left_end = adjust_left_std
                                min_expand = min_left_end
                                print("adjust_left_std bootstrap:", adjust_left_std)
                                expand = True
                                expansion = True
                if expand is True:
                    print(f"expand: min left end {min_left_end:.2f} max right end {max_right_end:.2f}")
                    avg = (max_right_end + min_left_end)/2
                    std = (max_right_end - min_left_end)/8
                    expand = False
                    hist_data = [0] * int(numbin)
                    #-------- modify : delete duplicate in left right end bin -------------------
                    #--------------- recount lstd4 rstd4 ----------------------------------------
                    end_bin_left = [] # add
                    end_bin_right = [] # add
                    end_bin_left = [i for i in new_data_chunk if (avg - 4*std <= i <= avg - 3*std)]
                    end_bin_right = [i for i in new_data_chunk if (avg + 3*std <= i <= avg + 4*std)]
                    end_bin_num_left.append(len(end_bin_left)) 
                    end_bin_num_right.append(len(end_bin_right))
                    hist_data[0] = len(end_bin_left)
                    hist_data[-1] = len(end_bin_right)
                    count += 1
                    boostrap_v1.plot_histogram(hist_theo[0],hist_theo[1], hist_theo[2],hist_theo[3],\
                                       hist_theo[4],hist_theo[5], hist_theo[6],hist_theo[7],\
                                           hist_data[0],hist_data[1], hist_data[2],hist_data[3],\
                                               hist_data[4],hist_data[5], hist_data[6],hist_data[7],\
                                                   min_expand, max_expand, 1, 1)
                    plt.show()
                    difference_max = hist_data[-1] - hist_theo[-1]
                    difference_min = hist_data[0] - hist_theo[0]
                    if min_left_end < min_new_input and max_new_input < max_right_end:
                        tmp_leftright = False
                        break
                else:
                    if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
                        # tmp_chunk_end  = end_bin_left + end_bin_right
                        # tmp_leftright = True
                        break
                print("after adjust chunk", i+1)
                print("interval std", std)        
                print("expand: min left end {min_left_end:.2f} max right end {max_right_end:.2f}")
        # else:
            # if expand_max is True:
            #     if use_bootstrap is True:
            #         if len(end_bin_right) >= nboost:
            #             if max_right_end <= max(end_bin_right):
            #                 adjust_right_std = boostrap_v1.bootstrap_online(end_bin_right, "right")
            #                 max_right_end = adjust_right_std
            #                 max_expand = max_right_end
            #                 print(f">>> go to Bootstrap right end bin: {max_right_end:.2f}")
            #                 expand = True
            #                 expansion = True
                    
            # if expand_min is True:
            #     if use_bootstrap is True:
            #         if len(end_bin_left) >= nboost:
            #             if min_left_end >= min(end_bin_left):
            #                 adjust_left_std = boostrap_v1.bootstrap_online(end_bin_left, "left")
            #                 min_left_end = adjust_left_std
            #                 min_expand = min_left_end
            #                 print("adjust_left_std bootstrap:", adjust_left_std)
            #                 expand = True
            #                 expansion = True
            # if (expand_min is True) or (expand_max is True):
            #     expand_chunk_list.append(i+1)
        
        boostrap_v1.plot_histogram(hist_theo[0],hist_theo[1], hist_theo[2],hist_theo[3],\
                               hist_theo[4],hist_theo[5], hist_theo[6],hist_theo[7],\
                                   hist_data[0],hist_data[1], hist_data[2],hist_data[3],\
                                       hist_data[4],hist_data[5], hist_data[6],hist_data[7],\
                                           min_expand, max_expand, i+1, expand)  
            
        if expansion is True or expand_min is True or expand_max is True:
            expand_chunk_list.append(i+1)
    print("\n====================================================\n")
    print("amount_population:", amount_population)
    print(f"min population: {all_min:.2f}")
    print(f"max population: {all_max:.2f}")
    print("input_size:", input_size)
    print(f'Keep end right and left data: {boostkeep}') 
    print(f"# experiment vs total chunks: {num_exp_chunk} vs. {number_data_chunk}", )
    print(f'# of expansion: {len(expand_chunk_list)}')
    print(f'expand_min_left_end and right end: [{min_left_end:.2f},{max_right_end:.2f}]')
    print(f'present min and max data: [{min(new_data_chunk):.2f},{max(new_data_chunk):.2f}]')
    print(f'min and max population: [{all_min:.2f},{all_max:.2f}]')
    
    print(">>>>> bootstrap whole data >>>>>>>")

    expand_left_min, expand_right_max = boostrap_v1.bootstrap_whole_data(randomlist[0:num_exp_chunk*input_size])
    print("\n====================================================\n")
    print("-----bootstrap whole data")
    print("\namount_population:", amount_population)
    
    print("\nexpand_min_left_end:", expand_left_min, " expand_max_right_end:", \
          expand_right_max)
    
    print("all_min:", all_min, " all_max:", all_max)
    
    diff_left_end_whole_data = (expand_left_min - all_min)
    diff_right_end_whole_data = (expand_right_max - all_max)
    print("\ndiff_left_end_whole_data:", diff_left_end_whole_data)
    print("diff_right_end_whole_data:", diff_right_end_whole_data)
    
    
    file_stat = open((file_stat_name+'.txt'), 'w')
    
    print("==================================================",file = file_stat)
    print("CONCLUSION:",file = file_stat)
    print(f'filename and sheet name: {filename} and {sheet_name}',file = file_stat)  
    print('\n>>>>> Population >>>> ',file = file_stat)
    print(f'# of population: {amount_population}', file = file_stat) 
    print(f"Min and Max: [{all_min:.2f},{all_max:.2f}]",file = file_stat)
    
    print('\n>>>>> Samples >>>> ',file = file_stat)
    print(f'Input size: {input_size}',file = file_stat)
    amount_samp = len(randomlist[0:num_exp_chunk*input_size])
    print(f'# of samples: { amount_samp}', file = file_stat) 
    all_min_samp = min(randomlist[0:num_exp_chunk*input_size])
    all_max_samp = max(randomlist[0:num_exp_chunk*input_size])
    print(f"Min and Max: [{all_min_samp:.2f},{all_max_samp:.2f}]",file = file_stat)
    print(f'Number of eperimental chunks vs. total chunk: {num_exp_chunk} vs. {number_data_chunk}',file = file_stat) 
    
    print("\n>>>> EXPAND online left right ends <<<<",file = file_stat)
    print(f'Keep boostrap: {boostkeep}',file = file_stat) 
    print(f'# expansions: {len(expand_chunk_list)}',file = file_stat) 
    print('expansion chunks:' ,expand_chunk_list,file = file_stat)
    print(f"Expand online boostrap: [{min_left_end:.2f},{max_right_end:.2f}]",file = file_stat)

    diff_left_end_online = (min_left_end - all_min)
    diff_right_end_online = (max_right_end - all_max)
    print(f'Expand error on left and right: {diff_left_end_online:.2f} and {diff_right_end_online:.2f}',file = file_stat)
    # print(f"diff_min_left_end_and_all_min:", diff_left_end_online)
    # print("diff_max_right_end_and_all_max:", diff_right_end_online)

    print("\n>>>> EXPAND whole data left right ends <<<<",file = file_stat)
    print(f"Expand online boostrap: [{expand_left_min:.2f},{expand_right_max:.2f}]",file = file_stat)

    diff_left_end_whole_data = (expand_left_min - all_min)
    diff_right_end_whole_data = (expand_right_max - all_max)
    print(f'Expand error left and right: {diff_left_end_whole_data:.2f} and {diff_right_end_whole_data:.2f}',file = file_stat)
    # print("all_min:", all_min, " all_max:", all_max)
    # print("expand_min_left_end:", expand_left_min, " expand_max_right_end:", \
    #       expand_right_max)
    
    
    # print("\ndiff_left_end_whole_data:", diff_left_end_whole_data)
    # print("diff_right_end_whole_data:", diff_right_end_whole_data)
    
    print("\n====================================================")
    
    # print("\n>>>> Difference from actual end: whole data")
    # print("\ndiff_left_end_whole_data:", diff_left_end_whole_data )
    # print("diff_right_end_whole_data:", diff_right_end_whole_data)
    
    # print("\n>>>> Difference from actual end: online")
    # print("\ndiff_min_left_end_and_all_min:", diff_left_end_online)
    # print("diff_max_right_end_and_all_max:", diff_right_end_online)

    list_error = [diff_left_end_online, diff_right_end_online, diff_left_end_whole_data, diff_right_end_whole_data]
    list_expand =[min_left_end,max_right_end,expand_left_min,expand_right_max] 
    list_bin_mean = [boostrap_v1.mean(end_bin_num_left),boostrap_v1.mean(end_bin_num_right)]
    pop_stat = {'N':amount_population,'min.val':all_min,'max.val':all_max}
    samp_stat = {'N':amount_samp,'min.val':all_min_samp,'max.val':all_max_samp,\
                 'input size':input_size,'no. boostrap expand': expand_chunk_list}
    return pop_stat, samp_stat,list_expand, list_error, expand_chunk_list, list_bin_mean

if __name__=='__main__':
    data_no = 5
    nround = 1
    if data_no == 1:
        filename = 'unifs0e500n10000-2'
        sheet_name = 'unifs0e500n10000-2' #random-data_minin_maxin_noise
        # sheet_name = 'random-data_minout_maxout' #random-data_minout_maxout random-data_minin_maxin
        input_size = 200
    if data_no == 2:
        filename = 'normal_1000_2'
        sheet_name = 'normal_1000_2_o5o5'  #normal_1000_2_ii, normal_1000_2_io5, normal_1000_2_o5o5 
        input_size = 50
    if data_no == 3:
        filename = 'normalm500n20000-2'
        sheet_name = 'normalm500n20000-2'  #normal_1000_2minin_maxin  normal_1000_2minin_maxout5 normal_1000_2minout5_maxout5
        input_size = 200
    if data_no == 4:
        filename = 'unif10000.xlsx'
        sheet_name = 'unif10000'  #unif10000
        input_size = 200
    if data_no == 5:
        filename = 'waldm1sd3n20000-2'
        sheet_name = 'waldm1sd3n20000-2'  #wald20000
        input_size = 200
    if data_no == 6:    
        filename = 'wiebullshape1n20000'
        sheet_name = 'wiebullshape1n20000'  #wiebullshape5n20000
        input_size = 400
    if data_no == 7:    
        filename = 'wiebullshape5n20000-2'
        sheet_name = 'wiebullshape5n20000-2'  #wiebullshape5n20000
        input_size = 400
    boostkeep = False
    # error_online_left = []
    # error_online_right = []
    # error_batch_left = []
    # error_batch_right = []
    for i in range(nround):
        pop_stat, samp_stat,list_expand, list_error, expand_chunk_list, list_bin_mean = main()
        # pop_stat, samp_stat,list_expand, list_error, expand_chunk_list, list_bin_mean = main(filename=(filename+'.xlsx'),sheet_name=sheet_name,input_size = input_size,boostkeep = boostkeep)
        # df_res = {'pop.':[pop_stat['# pop'],pop_stat['min.pop'],pop_stat['max.pop']],\
        #           'sampl':[samp_stat['# pop'],samp_stat['min.pop'],samp_stat['max.pop']]}
        workbook = xlsxwriter.Workbook((sheet_name +'-rest.xlsx'))
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'FileName:')
        worksheet.write('B1', filename)
        worksheet.write('C1', 'Sheetname:')
        worksheet.write('D1', sheet_name)
        
        worksheet.write('B2', 'Population')
        worksheet.write('C2', 'Samples')
        
        worksheet.write('A3', 'N')
        worksheet.write('B3', pop_stat['N'])
        worksheet.write('C3', samp_stat['N'])
        
        worksheet.write('A4', 'Minimum')
        worksheet.write('B4', pop_stat['min.val'])
        worksheet.write('C4', samp_stat['min.val'])
        
        worksheet.write('A5', 'Maximum')
        worksheet.write('B5', pop_stat['max.val'])
        worksheet.write('C5', samp_stat['max.val'])
        
        worksheet.write('B6', 'Online Bt')
        worksheet.write('C6', 'Whole Bt')
        
        worksheet.write('A7', 'Left expand')
        worksheet.write('B7', list_expand[0])
        worksheet.write('C7', list_expand[2])
        
        worksheet.write('A8', 'Right expand')
        worksheet.write('B8', list_expand[1])
        worksheet.write('C8', list_expand[3])
        
        worksheet.write('A9', 'Left error')
        worksheet.write('B9', list_error[0])
        worksheet.write('C9', list_error[2])
        
        worksheet.write('A10', 'Right eror')
        worksheet.write('B10', list_error[1])
        worksheet.write('C10', list_error[3])
        worksheet.write('A11', '# bt chunks:')
        worksheet.write('B11', len(expand_chunk_list))
        worksheet.write('A12', '# left end bin:')
        worksheet.write('B12', list_bin_mean[0])
        worksheet.write('A13', '# right end bin:')
        worksheet.write('B13', list_bin_mean[1])
        worksheet.write('A14', 'Input size:')
        worksheet.write('B14', input_size)
        # worksheet.write('A9', 'Chunk no. update:')
        # worksheet.write('B9', expand_chunk_list)
        workbook.close()
        # error_online_left.append(list_error[0])
        # error_online_right.append(list_error[1])
        # error_batch_left.append(list_error[2])
        # error_batch_right.append(list_error[3])
        
        
    # print(f'Status: {y}')