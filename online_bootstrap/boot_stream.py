
'''
   This the library for boostrap method 
'''

from dataclasses import dataclass, field
from typing import List
from online_bootstrap import bootstrap_v1, BatchOutlierDetection
import math, copy, statistics
import numpy as np
import pandas as pd
import statistics


@dataclass
class booststream:
    online_cum: bool = False
    online:bool = False # if Online boostrap active, it is set to True.
    minmax_boost:bool = False # if Minmax boostrap active, it is set to True. 
    # addnoise:bool = False # Addnoise in simulate data
    # prob:bool = False # prob-based sample selection in boostrap applied.
    filesampl:str = '' # file name of samples
    numbin:int = 0 # number of theoritical bin for an example 8
    number_bt_iter:int = 600 # number of boostrap iteration.
    nboost:int = 0 # minimum number of conducting boostrap.
    dist_list:List[str] = field(default_factory=list)
    total_size:int = 0 # number of total learned data.
    chunk_size:int = 0 
    max_chs:float = - 9999.99 # maximumn value of learning data from the start to current chunk.
    min_chs:float =  9999.99 # minimum value of learning data from the start to current chunk.
    min_list:List[str] = field(default_factory=list) 
    max_list:List[str] = field(default_factory=list)
    avg: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    exp_l:float = 9999.99
    exp_r:float = -9999.99
    range:float = 0.0
    flag_learning:bool = False # flag for idenfying the learning process is performed.
    nlearn_l: List[int] = field(default_factory=list)
    nlearn_r: List[int] = field(default_factory=list)
    
    def set_online(self,minmax_flag:bool = False):
        dist_list = ['exponweib', 'wald', 'gamma', 'norm',\
                         'expon', 'powerlaw', 'lognorm', 'chi2', 'weibull_min',\
                         'weibull_max']
        self.online = True
        self.minmax_boost = minmax_flag 
        self.numbin = 8
        # self.number_bt_iter = 600
        self.dist_list = dist_list
        self.nboost = 3
        # self.total_size  = 0
        # self.max_chs = - 9999.99
        # self.min_chs = 9999.99
        # self.exp_l = 9999.99
        # self.exp_r = -9999.99
        
        
    
    def compute_error(self,target_l: float,target_r: float):
        target_range = target_r - target_l
        return (target_l-self.exp_l), (target_r-self.exp_r), (target_range-self.range)
        
    def update_center_range(self,leftmost: float,rightmost: float)-> None:
        self.avg.append((rightmost+leftmost)/2)
        self.std.append((rightmost-leftmost)/8)
        
    
        
    def expand_bt_online(self,new_data_chunk:list,outlier:bool,cum:bool = False,cum_left_right:bool = False) -> None:
        '''
        1. Check if the network is online manner or not
        2. Update the number of learning samples
        3. Compute min and max values of the current data chunk
        4. If we get the new min or max values update the left-expand or right-expand
            4.1 Compute the update vales based on min-max boostrapping, or
            4.2 Compute the update vales based on min and max vaues of the current data chunk
        5. If the left and right expand values have been updated.
            5.1 Update mean and std from the left and right expand values
            5.2 Collect the list of minimum and maximum data list from data fall into \
                the leftmost bin and the rightmost bin
            5.3 Compute the data histogram and theoritical histogram    
        '''
        
        # 1. Check if the network is online manner or not
        try: 
            if self.online is False:
                raise ValueError("The network in traditional mode. Can not perform online mode.")
            
        except ValueError as e:
            return print(f"Error: {e}")
        
        # 2. Update the number of learning samples
        if not cum:  # 
            self.total_size += len(new_data_chunk)
        else:
            self.total_size = len(new_data_chunk)
        self.chunk_size = len(new_data_chunk)
        
        if outlier:
            if self.avg ==[]:    
                self.avg.append(statistics.mean(new_data_chunk))
            if self.std == []:
                self.std.append(statistics.stdev(new_data_chunk)) # sample standard deviation. 
            detector = BatchOutlierDetection.ZBatchOutlierDetector()
            detector.add_init_params(threshold=3.0,mean=self.avg[-1],sd=self.std[-1])
            new_data_chunk = detector.get_clean_data(new_data_chunk)
        
        # 3. Compute min and max values of the current data chunk
        
        
        
        new_data_chunk_min = min(new_data_chunk)
        new_data_chunk_max = max(new_data_chunk)
        if new_data_chunk_min < self.min_chs:
            self.min_chs = new_data_chunk_min
        if new_data_chunk_max > self.max_chs:
            self.max_chs = new_data_chunk_max
        
        # Start learning
        expand_min = False
        expand_max = False
        expansion = False
        # 4. If we get the new min or max values update the left-expand or right-expand
            # 4.1 Compute the update vales based on min-max boostrapping, or
            # 4.2 Compute the update vales based on min and max vaues of the current data chunk
        
        if new_data_chunk_min < self.exp_l: # the min of the chunk is less than the min of the network.
            expand_min = True
            if len(self.min_list) >= self.nboost and (self.minmax_boost is True):
                self.min_list.append(new_data_chunk_min)
                # expand_min = True
                adjust_left_std = bootstrap_v1.bootstrap_online(self.min_list, "left",\
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
                adjust_right_std = bootstrap_v1.bootstrap_online(self.max_list, "right",\
                                                                number_bootstrap_iteration = self.number_bt_iter, \
                                                                    minmax_boost = self.minmax_boost,\
                                                                        prob = False)
                
                if self.exp_r <= adjust_right_std:
                    self.exp_r = adjust_right_std
            else:
                self.exp_r = new_data_chunk_max
                
        # 5. If the left and right expand values have been updated.        
        if (expand_min is True) or (expand_max is True):
            # 5.1 Update mean and std from the left and right expand values
            self.update_center_range(self.exp_l,self.exp_r)
            avg = self.avg[-1]
            std = self.std[-1]
            end_bin_left = []
            end_bin_right = []
            
            if cum_left_right is True: # accumulated addition of the left and the right bin
                new_data_chunk = new_data_chunk + self.min_list + self.max_list
            # 5.2 Collect the list of minimum and maximum data list from data fall into \
                # the leftmost bin and the rightmost bin  
            self.min_list = [k for k in new_data_chunk if (avg - 4*std <= k <= avg - 3*std)]
            self.max_list = [k for k in new_data_chunk if (avg + 3*std <= k <= avg + 4*std)]
            hist_data = [0] * int(self.numbin) # initialize experimental histogram    
            hist_theo = [0] * int(self.numbin) # initialize theoritical histogram
            end_bin_left= self.min_list
            end_bin_right = self.max_list
            
            # 5.3 Compute the data histogram and theoritical histogram 
            hist_data[0] = len(end_bin_left) # The # data in the left most bin.
            hist_data[-1] = len(end_bin_right) # The # data in the right most bin.
            hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
            hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
            percent_data = bootstrap_v1.get_percent_std_data_from_best_distribution(\
                                               self.total_size, self.min_list, self.max_list, \
                                               self.dist_list)
            hist_theo = [math.ceil(i*self.total_size/100.0) for i in percent_data]
        #     self.endLn.append(len(self.endL))
        #     self.endRn.append(len(self.endR))
            
            expand = False
            expansion = False
            
             
            
            # 5.4 Compute the different values of the both histograms
            difference_max = hist_data[-1] - hist_theo[-1]
            difference_min = hist_data[0] - hist_theo[0]
        #     # while (difference_max > 0 or difference_min > 0):
            if (difference_max > 0 or difference_min > 0):
                dif_expand = True
                if difference_max > 0:
                    self.nlearn_r.append(hist_data[-1])
                else:
                    self.nlearn_r.append(0)    
                    
                if difference_min > 0:
                    self.nlearn_l.append(hist_data[0])
                else:
                    self.nlearn_l.append(0)
            else:
                dif_expand = False    
                
            while dif_expand is True:    
        #         # difference_max_tmp = difference_max
        #         # difference_min_tmp = difference_min
                expandL = self.exp_l
                expandR = self.exp_r
                if difference_max > 0:
                    
                    if hist_data[-1] >= self.nboost:
                        if self.flag_learning is False:
                            tmp_exp_r = bootstrap_v1.bootstrap_online(self.max_list, "right",\
                                                                number_bootstrap_iteration = self.number_bt_iter, \
                                                                    minmax_boost = self.minmax_boost,\
                                                                        prob = False) 
                            if tmp_exp_r>expandR:
                                self.exp_r = tmp_exp_r
                                expand = True
                                expansion = True
                                
                        if self.exp_r <= max(self.max_list):
                            self.exp_r = bootstrap_v1.bootstrap_online(self.max_list, "right",\
                                                                number_bootstrap_iteration = self.number_bt_iter, \
                                                                    minmax_boost = self.minmax_boost,\
                                                                        prob = False)
                            if self.exp_r<expandR:
                                self.exp_r = expandR
                                
                            expand = True
                            expansion = True
                else:
                    expand = False
                        
                if difference_min > 0:
                    if hist_data[0] >= self.nboost:
                        if self.flag_learning is False:
                            self.flag_learning = True
                            tmp_exp_l = bootstrap_v1.bootstrap_online(self.min_list, "left",\
                                                                number_bootstrap_iteration = self.number_bt_iter, \
                                                                    minmax_boost = self.minmax_boost,\
                                                                        prob = False) 
                            if tmp_exp_l<expandL:
                                self.exp_l = tmp_exp_l
                                expand = True
                                expansion = True
                        if self.exp_l >= min(self.min_list):
                            self.exp_l = bootstrap_v1.bootstrap_online(self.min_list, "left",\
                                                                number_bootstrap_iteration = self.number_bt_iter, \
                                                                    minmax_boost = self.minmax_boost,\
                                                                        prob = False)
                            if self.exp_l>expandL:
                                self.exp_l = expandL
                            expand = True
                            expansion = True
                # else:
                #     if expand is False:
                #         expand = True
                if expand is True:
                    self.update_center_range(self.exp_l,self.exp_r)
                    avg = self.avg[-1]
                    std = self.std[-1]
                    expand = False
                    hist_data = [0] * int(self.numbin)
                    end_bin_left = [] # add
                    end_bin_right = [] # add
                    end_bin_left = [i for i in new_data_chunk if (avg - 4*std <= i <= avg - 3*std)]
                    end_bin_right = [i for i in new_data_chunk if (avg + 3*std <= i <= avg + 4*std)]
                    hist_data[0] = len(end_bin_left)
                    hist_data[-1] = len(end_bin_right)
                    hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
                    hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
                    self.min_list = end_bin_left
                    self.max_list = end_bin_right
        #             self.endLn.append(len(end_bin_left))
        #             self.endRn.append(len(end_bin_right))
                    
                    
        #             # if abs(difference_max-difference_max_tmp) == 0 and abs(difference_min-difference_min_tmp) == 0:
                    if expandL == self.exp_l and expandR == self.exp_r:     
                        dif_expand = False
                    else:
                        expand = False
                        difference_max = hist_data[-1] - hist_theo[-1]
                        difference_min = hist_data[0] - hist_theo[0]
                else:
                    dif_expand = False
        if (expansion is True) or (expand_max is True) or (expand_min is True):
           self.range = self.exp_r - self.exp_l
        #    self.le_samp.append(self.min_chs - self.expandL)
        #    self.re_samp.append(self.max_chs - self.expandR)
        #    self.le_pop.append(self.pop_min - self.expandL)
        #    self.re_pop.append(self.pop_max - self.expandR)
           expansion = True
        return expansion  
    
    # def expand_bt_online_outlier(self,new_data_chunk:list,cum:bool = False,cum_left_right:bool = False) -> None: 
    #     self.outlier = True
    #     if self.avg ==[]:    
    #         self.avg.append(statistics.mean(new_data_chunk))
    #     if self.std == []:
    #         self.std.append(statistics.stdev(new_data_chunk)) # sample standard deviation. 
    #     detector = BatchOutlierDetection.ZBatchOutlierDetector()
    #     detector.add_init_params(threshold=3.0,mean=self.avg[-1],sd=self.std[-1])
    #     new_data_chunk = detector.get_clean_data(new_data_chunk)
    #     expansion = self.expand_bt_online(new_data_chunk,cum,cum_left_right)     
    #     return expansion            
        
    def expand_bt_trad(self,input_data:list) -> None:
        # Traditional boostrap method
        try: 
            if self.online is True:
                raise ValueError("The network in online mode. Can not perform whole mode.")
            
        except ValueError as e:
            return print(f"Error: {e}")
        self.number_bt_iter = 600
        data_set = copy.deepcopy(input_data)
        nsample = len(data_set)
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
        self.nlearn_l.append(nsample)
        self.nlearn_r.append(nsample) 
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