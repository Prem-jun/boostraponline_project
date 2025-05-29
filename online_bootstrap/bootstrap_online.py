# press command + I   for copilot
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import json, math
from math import inf
from online_bootstrap import bootstrap_v1, BatchOutlierDetection
import math, copy, statistics

@dataclass
class BootstrapOnline:
    # Configurable settings
    online_cum: bool = False
    online: bool = False
    outlier_detection: bool = False
    outlier_method: str = ''
    flag_learning: bool = False

    # Initialized later
    minmax_boost: bool = field(init=False)
    numbin: Optional[int] = field(init=False)
    number_bt_iter: Optional[int] = field(init=False)
    nboost: Optional[int] = field(init=False)

    # Statistics
    avg: float = field(init=False)
    std: float = field(init=False)
    exp_l: float = field(init=False)
    exp_r: float = field(init=False)
    total_size: int = 0
    num_learn: Optional[int] = field(init=False)

    # Tracking and distribution settings
    history: List[dict] = field(default_factory=list)
    dist_list: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.minmax_boost = False
        self.numbin = None
        self.number_bt_iter = None
        self.nboost = None
        self.avg = inf
        self.std = inf
        self.exp_l = inf
        self.exp_r = -inf
        self.num_learn = 0
        
        
    
    def update_center_range(self,leftmost: float,rightmost: float)-> None:
        self.avg = (rightmost+leftmost)/2
        self.std = (rightmost-leftmost)/8
    
    def pipe(self, func):
        """Apply a function to the instance and return the instance (chainable)."""
        func(self)
        return self    

    def set_trad(self):
        self.online = False
        self.online_cum = False
        self.minmax_boost = False
        self.number_bt_iter = 600
        self.history = [{'exp_l':inf,'exp_r':-inf,'ch_min':-inf, 'ch_max': inf,'min_list': [], \
            'max_list': [],'nlearn_l': 0, 'nlearn_r': 0,\
                'flag_learning': False}]
    
    def set_outlier_detection(self, method: str):
        """Set the outlier detection method."""
        self.outlier_detection = True
        self.outlier_method = method
    
    def set_online(self,online_cum :bool = False,minmax_boost:bool = False):
        """Configure the object for online mode with default parameters."""
        self.online = True
        self.online_cum = online_cum
        self.minmax_boost = minmax_boost
        self.numbin = 8
        self.number_bt_iter = 600
        self.nboost = 3
        self.dist_list = [
            'exponweib', 'wald', 'gamma', 'norm',
            'expon', 'powerlaw', 'lognorm', 'chi2',
            'weibull_min', 'weibull_max'
        ]
        self.history = [{'exp_l':inf,'exp_r':-inf,'ch_min':-inf, 'ch_max': inf,'min_list': [], \
            'max_list': [],'nlearn_l': 0, 'nlearn_r': 0,\
                'flag_learning': False}]
        
    def log_epoch(self, ch_min: float, ch_max: float, \
        min_list: Optional[List[float]] = None, \
            max_list: Optional[List[float]] = None,\
                nlearn_l: Optional[int] = None, \
                    nlearn_r: Optional[int] = None,\
                        flag_learning: Optional[bool] = None):
        if self.num_learn == 0:
            
            self.history[0]['exp_l'] = float(self.exp_l)
            self.history[0]['exp_r'] = float(self.exp_r)
            self.history[0]['ch_min'] = ch_min
            self.history[0]['ch_max'] = ch_max
            self.history[0]['min_list'] = min_list if min_list is not None else []
            self.history[0]['max_list'] = max_list if max_list is not None else []
            self.history[0]['nlearn_l'] = nlearn_l if nlearn_l is not None else 0
            self.history[0]['nlearn_r'] = nlearn_r if nlearn_r is not None else 0
            self.history[0]['flag_learning'] = flag_learning if flag_learning is not None else False
        else:
            self.history.append({'exp_l':float(self.exp_l),'exp_r':float(self.exp_r),"ch_min": ch_min, "ch_max": ch_max,\
                "min_list": min_list if min_list is not None else [],\
                    "max_list": max_list if max_list is not None else [],\
                        "nlearn_l": nlearn_l if nlearn_l is not None else 0,\
                            "nlearn_r": nlearn_r if nlearn_r is not None else 0,\
                                "flag_learning": flag_learning if flag_learning is not None else False})    
        self.num_learn += 1
    
    
    def expand_leftby_chmin(self, chunk_min: float):
        """Expand the left boundary by the minimum value."""
        if chunk_min < self.exp_l: # the min of the chunk is less than the min of the network.
            expand_min = True
            # min_list.append(chunk_min)
            if len(self.history[-1]['min_list']) >= self.nboost and (self.minmax_boost is True):
                # expand_min = True
                adjust_left_std = bootstrap_v1.bootstrap_online(self.history[-1]['min_list'], "left",\
                                                               number_bootstrap_iteration = self.number_bt_iter, \
                                                                   minmax_boost = self.minmax_boost,\
                                                                       prob = False)
                if self.exp_l >= adjust_left_std:
                    self.exp_l = adjust_left_std      
            else:
                # expand_min = True
                self.exp_l = chunk_min
            return True    
        else:
            return False
         
    def expand_rightby_chmax(self, chunk_max: float):
        """Expand the right boundary by the maximum value."""
        if chunk_max > self.exp_r:
            expand_max = True
            # self.history[-1]['max_list'].append(chunk_max)
            if (len(self.history[-1]['max_list']) >= self.nboost) and (self.minmax_boost is True):
                
                adjust_right_std = bootstrap_v1.bootstrap_online(self.history[-1]['max_list'], "right",\
                                                                number_bootstrap_iteration = self.number_bt_iter, \
                                                                    minmax_boost = self.minmax_boost,\
                                                                        prob = False)
                
                if self.exp_r <= adjust_right_std:
                    self.exp_r = adjust_right_std
            else:
                self.exp_r = chunk_max
            return True
        else:
            return False
    def compute_histogram(self, new_data_chunk: List[float]):
        # 5.1 Update mean and std from the left and right expand values
        self.update_center_range(self.exp_l,self.exp_r)
        avg = self.avg
        std = self.std
        end_bin_left = []
        end_bin_right = []  
        
        # 5.2 Collect the list of minimum and maximum data list from data fall into \
                # the leftmost bin and the rightmost bin  
        min_list = [k for k in new_data_chunk if (avg - 4*std <= k <= avg - 3*std)]
        max_list = [k for k in new_data_chunk if (avg + 3*std <= k <= avg + 4*std)]
        hist_data = [0] * int(self.numbin) # initialize experimental histogram    
        hist_theo = [0] * int(self.numbin) # initialize theoritical histogram  
        
        # 5.3 Compute the data histogram and theoritical histogram
        percent_data = bootstrap_v1.get_percent_std_data_from_best_distribution(\
                                               self.total_size, min_list, max_list, \
                                               self.dist_list)
        hist_theo = [math.ceil(i*self.total_size/100.0) for i in percent_data]
        hist_data[0] = len(min_list) # The # data in the left most bin.
        hist_data[-1] = len(max_list) # The # data in the right most bin.
        hist_data[1] = len([i for i in new_data_chunk if (avg - 3*std <= i <= avg - 2*std)])
        hist_data[-2] = len([i for i in new_data_chunk if (avg + 2*std <= i <= avg + 3*std)])
        return hist_data, hist_theo, min_list, max_list
    # def get_histogram_difference(self, hist_data: List[int], hist_theo: List[int]):
    
    def detect_outliers(self, data: List[float], method: str ='zscore') -> List[bool]:    
        if self.outlier_method == 'zscore':
            # Use Z-score method for outlier detection
            avg = self.avg
            std = self.std
            min_list = [k for k in data if (k <= avg - 3*std)]
            max_list = [k for k in data if (avg + 3*std <= k)]
            mid_list = [k for k in data if (avg - 3*std < k < avg + 3*std)]
            # min_list = [k for k in data if (avg - 4*std <= k <= avg - 3*std)]
            # max_list = [k for k in data if (avg + 3*std <= k<= avg + 4*std)]
            # mid_list = [k for k in data if (avg - 3*std < k < avg + 3*std)]
            if len(max_list)>=5:
                detector_r = BatchOutlierDetection.ZBatchOutlierDetector()
                mean_r = statistics.mean(max_list)
                std_r = statistics.stdev(max_list)
                detector_r.add_init_params(threshold=3.5,mean=mean_r,sd=std_r)   
                # outliers = detector.detect_outliers(max_list)
                clean_max_list = detector_r.get_clean_data(data = max_list)
            else:
                 clean_max_list = max_list  
                #  clean_max_list = []
            if len(min_list)>=5:
                detector_l = BatchOutlierDetection.ZBatchOutlierDetector()
                mean_l = statistics.mean(min_list)
                std_l = statistics.stdev(min_list)
                detector_l.add_init_params(threshold=3.5,mean=mean_l,sd=std_l)
                clean_min_list = detector_l.get_clean_data(data = min_list)
            else:
                clean_min_list = min_list
                # clean_min_list = []
            return clean_min_list+ mid_list + clean_max_list
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
            
    def bt_online(self,new_data_chunk:list,ndata:int):    
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
        if not self.online_cum:  # 
            self.total_size += ndata
        else:
            self.total_size = ndata
        self.chunk_size = ndata
        
        # 3. Compute min and max values of the current data chunk
        if self.outlier_detection:
            if self.num_learn == 0:
                chunk_min = min(new_data_chunk)
                chunk_max = max(new_data_chunk)
            else:
                new_data_chunk=self.detect_outliers(new_data_chunk, method='zscore')
         
        chunk_min = min(new_data_chunk)
        chunk_max = max(new_data_chunk)
        # Start learning
        expansion = False
        
        # 4. If we get the new min or max values update the left-expand or right-expand
            # 4.1 Compute the update vales based on min-max boostrapping, or
            # 4.2 Compute the update vales based on min and max vaues of the current data chunk
        
        expand_min = self.expand_leftby_chmin(chunk_min)
        expand_max = self.expand_rightby_chmax(chunk_max)
        
        # 5. If the left and right expand values have been updated.        
        if expand_min or expand_max:
            hist_data, hist_theo, min_list, max_list = self.compute_histogram(new_data_chunk)
            expand = False
            expansion = False
            
            # 5.4 Compute the differences at both ends of the histograms
            difference_max = hist_data[-1] - hist_theo[-1]
            difference_min = hist_data[0] - hist_theo[0]

            # Check if either boundary has excess data
            if difference_max > 0 or difference_min > 0:
                dif_expand = True
                
                # Determine the amount of learning needed on the right
                nlearn_r = hist_data[-1] if difference_max > 0 else 0
                # if nlearn_r > 5:
                #     if self.outlier_detection:
                #         if self.outlier_method == 'zscore':
                #                 # Use Z-score method for outlier detection
                #                 detector = BatchOutlierDetection.BatchOutlierDetection()
                #                 mean = statistics.mean(max_list)
                #                 std = statistics.stdev(max_list)
                #                 detector.add_init_params(threshold=4.0,mean=mean,std=std)   
                #                 # outliers = detector.detect_outliers(max_list)
                #                 clearn_data = detector.get_clean_data(data = max_list)
                #                 if len(clearn_data) < hist_data[-1]:
                #                     hist_data[-1] = len(clearn_data)
                #                     nlearn_r = len(clearn_data)
                #                     self.exp_r = max(clearn_data)  # Update exp_r to the new max value

                # Determine the amount of learning needed on the left
                nlearn_l = hist_data[0] if difference_min > 0 else 0
                # if nlearn_l > 5:
                #     if self.outlier_detection:
                #         if self.outlier_method == 'zscore':
                #                 # Use Z-score method for outlier detection
                #                 detector = BatchOutlierDetection.ZBatchOutlierDetector()
                #                 mean = statistics.mean(min_list)
                #                 std = statistics.stdev(min_list)
                #                 detector.add_init_params(threshold=3.5,mean=mean,sd=std)   
                #                 # outliers = detector.detect_outliers(min_list)
                #                 clearn_data = detector.get_clean_data(data = min_list)
                #                 if len(clearn_data) < hist_data[0]:
                #                     hist_data[0] = len(clearn_data)
                #                     nlearn_l = len(clearn_data)
                #                     self.exp_l = min(clearn_data)
                                
                                
                                
            else:
                dif_expand = False
                nlearn_r = 0
                nlearn_l = 0    
            
            while dif_expand:
                expandL = self.exp_l
                expandR = self.exp_r
                expand = False  # Reset at each loop
                expansion = False  # Tracks whether any expansion was done in this iteration
    
                # Check right-end expansion
                if difference_max > 0 and hist_data[-1] >= self.nboost:
                    if not self.flag_learning:
                        tmp_exp_r = bootstrap_v1.bootstrap_online(
                            max_list, "right",
                            number_bootstrap_iteration=self.number_bt_iter,
                            minmax_boost=self.minmax_boost,
                            prob=False
                        )
                        if tmp_exp_r > expandR:
                            self.exp_r = tmp_exp_r
                            expand = True
                            expansion = True
                
                    if self.exp_r <= max(max_list):
                        new_exp_r = bootstrap_v1.bootstrap_online(
                            max_list, "right",
                            number_bootstrap_iteration=self.number_bt_iter,
                            minmax_boost=self.minmax_boost,
                            prob=False
                        )
                        self.exp_r = max(new_exp_r, expandR)  # Ensure it doesn't shrink
                        expand = True
                        expansion = True
                # Check left-end expansion
                if difference_min > 0 and hist_data[0] >= self.nboost:
                    if not self.flag_learning:
                        self.flag_learning = True
                        tmp_exp_l = bootstrap_v1.bootstrap_online(
                            min_list, "left",
                            number_bootstrap_iteration=self.number_bt_iter,
                            minmax_boost=self.minmax_boost,
                            prob=False
                        )
                        if tmp_exp_l < expandL:
                            self.exp_l = tmp_exp_l
                            expand = True
                            expansion = True

                    if self.exp_l >= min(min_list):
                        new_exp_l = bootstrap_v1.bootstrap_online(
                            min_list, "left",
                            number_bootstrap_iteration=self.number_bt_iter,
                            minmax_boost=self.minmax_boost,
                            prob=False
                        )
                        self.exp_l = min(new_exp_l, expandL)
                        expand = True
                        expansion = True
                # If any expansion happened, update center range and recompute histogram edges
                if expand:
                    self.update_center_range(self.exp_l, self.exp_r)
                    avg = self.avg
                    std = self.std
                    hist_data = [0] * self.numbin

                    # Update hist_data from new_data_chunk
                    end_bin_left = [i for i in new_data_chunk if avg - 4 * std <= i <= avg - 3 * std]
                    end_bin_right = [i for i in new_data_chunk if avg + 3 * std <= i <= avg + 4 * std]
                    hist_data[0] = len(end_bin_left)
                    hist_data[-1] = len(end_bin_right)
                    hist_data[1] = len([i for i in new_data_chunk if avg - 3 * std <= i <= avg - 2 * std])
                    hist_data[-2] = len([i for i in new_data_chunk if avg + 2 * std <= i <= avg + 3 * std])

                    min_list = end_bin_left
                    max_list = end_bin_right

                    # Check for convergence: no update in expansion
                    if expandL == self.exp_l and expandR == self.exp_r:
                        dif_expand = False
                    else:
                        # Recompute differences for next iteration
                        difference_max = hist_data[-1] - hist_theo[-1]
                        difference_min = hist_data[0] - hist_theo[0]
                else:
                    dif_expand = False        
            # # Final range update if any expansion occurred
            # if expansion:
            #     self.range = self.exp_r - self.exp_l 
            self.log_epoch(ch_min = chunk_min, ch_max = chunk_min,\
                min_list = min_list, max_list = max_list,\
                    nlearn_l = nlearn_l, nlearn_r = nlearn_r,\
                        flag_learning = self.flag_learning)
        else: 
            self.flag_learning = False          
            self.log_epoch(ch_min = chunk_min, ch_max = chunk_min,\
                        flag_learning = self.flag_learning)        
        
        
    def bt_trad(self,new_data_chunk:list, ndata:int) -> None:
        # Traditional boostrap method
        try: 
            if self.online is True:
                raise ValueError("The network in online mode. Can not perform whole mode.")
            
        except ValueError as e:
            return print(f"Error: {e}")
        
        data_set = copy.deepcopy(new_data_chunk)
        bootstrap_min = []
        bootstrap_max = []
    
        size_boost = len(data_set)  
        # create list of boostrap samples.
        bootstrap_sample_list = [ list(np.random.choice(data_set, size_boost,replace=True))
                                        for i in range(self.number_bt_iter)
                            ]  
        for idx,samples in enumerate(bootstrap_sample_list):
            bootstrap_min.append(np.min(samples))
            bootstrap_max.append(np.max(samples))
        
        self.chunk_size = ndata
        self.exp_l = float(np.mean(bootstrap_min))
        self.exp_r = float(np.mean(bootstrap_max))
        self.update_center_range(self.exp_l, self.exp_r)
        chunk_max = float(np.max(data_set))
        chunk_min = float(np.min(data_set))  
        self.log_epoch(ch_min = chunk_min, ch_max = chunk_max,\
            min_list = [], max_list = [],\
                nlearn_l = 0, nlearn_r = 0,\
                    flag_learning = False)  
             

