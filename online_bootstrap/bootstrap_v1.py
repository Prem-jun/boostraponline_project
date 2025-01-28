
import copy
import numpy as np
import math
from online_bootstrap import stat_dist
import matplotlib.pyplot as plt

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
                            ]
    previous_bootstrap_mean = 0      
    for idx,samples in enumerate(bootstrap_sample_list):
        present_bootstrap_mean = (np.mean(samples) +
                                    previous_bootstrap_mean)/2
        previous_bootstrap_mean = present_bootstrap_mean
        bootstrap_means[idx] = present_bootstrap_mean
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
    different_input_std_bootstrap_std = abs(stdev(input_data,input_mean) - estimated_std)
    
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
        
#============================================================================
def estimate_width_1D(input_data: list, max_value: float, z_score:float) -> float:
    sample = copy.deepcopy(input_data)
    sample = np.array(sample)
    size = len(sample)
    estimate_width = abs(((max_value)/(1-z_score*(math.sqrt(2/(size-1)))) + \
                            (max_value)/(1+z_score*(math.sqrt(2/(size-1)))) )/2)

    return estimate_width
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
        percent_data_rstd3, percent_data_rstd4 = distribution_percent(\
                                                    right_name)
    if right_name != left_name:
        if left_min < right_min:
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = distribution_percent(\
                                                    right_name)
        if right_min < left_min:
            percent_data_lstd1, percent_data_lstd2, percent_data_lstd3, \
            percent_data_lstd4, percent_data_rstd1, percent_data_rstd2, \
            percent_data_rstd3, percent_data_rstd4 = distribution_percent(\
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
    estimated_std_of_mean = stdev(bootstrap_means,estimated_mean)
    different_input_mean_bootstrap_mean = abs(mean(input_data) - estimated_mean)
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
    estimated_std_of_std = stdev(bootstrap_std,estimated_std)
    different_input_std_bootstrap_std = abs(stdev(input_data,input_mean) - estimated_std)
    # different_input_std_bootstrap_std = abs(boostrap_v1.stdev(input_data) - estimated_std)

    estimated_diff_dist_std = np.mean(bootstrap_max_diff_dist_std)
    estimated_std_of_diff_dist_std = np.std(bootstrap_max_diff_dist_std)

    # input_mean = np.mean(input_data)
    input_std = stdev(input_data,input_mean)

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

