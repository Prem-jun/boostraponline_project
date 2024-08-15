# This is a sample Python script.
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import chisquare
def update_mean(center, ni, xvec):
    """
    :param center: float
    :param ni: int
    :param xvec: array
    :return: centern: float, new center
            ni: int, upated number
    """
    centern = (ni * center + xvec) / (ni + 1)
    ni = ni + 1
    return centern, ni


def update_std(xvec, center_new, ss, ni):
    nVar = center_new.shape[1]
    if center_new.ndim == 1:
        center_new = np.array([center_new])
    if ss.ndim == 1:
        ss = np.array([ss])
    if xvec.ndim == 1:
        xvec = np.array([xvec])
    s_new = np.array([[0.0] * nVar] * 1)
    for idx, x in np.ndenumerate(center_new):
        tmp_val = math.sqrt(((ni - 1) * (ss[0, idx[1]] ** 2) + ((ni + 1) / ni) * (x - xvec[0, idx[1]]) ** 2) / ni)
        s_new[0, idx[1]] = tmp_val
    return s_new


def batch_stdev(data):
    list_size = len(data)
    avg = 0
    for i in range(list_size):
        avg = avg + data[i]
    avg = avg / list_size
    std = 0
    for i in range(list_size):
        std = std + (data[i] - avg) ** 2
    std = math.sqrt(std / list_size)
    return std


def batch_mean(data):
    """
    Compute a mean value of batch dataset
    :param data: array
    :return: float: batch mean value
    """
    list_size = len(data)
    avg = 0
    for i in range(list_size):
        avg = avg + data[i]
    avg = avg / list_size
    return avg


def std_hist(nsampl, port=None):
    """
    :param nsampl: int
    :param port: slist
    :return: list of standard freqencies based on the sample size
    """
    if port is None:
        port = [0.1, 2.1, 13.6, 34.1, 34.1, 13.6, 2.1, 0.1]
    nsampl_list = [nsampl] * len(port)
    port_num = lambda v, nsampl1: math.ceil(v * nsampl1 / 100.0)
    sstd_hist = list(map(port_num, port, nsampl_list))
    return sstd_hist

def data_hist(a, avg, std, numbin = 8):
    """
    Count the number of data following 8 bin
    :param a:
    :param avg:
    :param std:
    :param numbin:
    :return:
    """
    data_hist1 = [0] * int(numbin)
    for i in a:
        if avg - std <= i < avg:
            data_hist1[3] += 1
        if avg - 2 * std <= i < avg - std:
            data_hist1[2] += 1
        if avg - 3 * std <= i < avg - 2 * std:
            data_hist1[1] += 1
        if avg - 4 * std <= i < avg - 3 * std:
            data_hist1[0] += 1

        if avg <= i < avg + std:
            data_hist1[4] += 1
        if avg + std <= i < avg + 2 * std:
            data_hist1[5] += 1
        if avg + 2 * std <= i < avg + 3 * std:
            data_hist1[6] += 1
        if avg + 3 * std <= i < avg + 4 * std:
            data_hist1[7] += 1
    return data_hist1


def myplot_histogram(sstd_hist, data_hist2, chunk_no = None, expand_count = None):
    if chunk_no is None:
        chunk_no = 0
    if expand_count is None:
        expand_count = 0
    name = ['lstd4', 'lstd3', 'lstd2', 'lstd1', 'rstd1', 'rstd2', 'rstd3', 'rstd4']
    title = "chunk: " + str(chunk_no) + " expand count: " + str(expand_count)
    plt.title(title)
    plt.bar(name, sstd_hist, alpha=0.1, color='red')
    plt.bar(name, data_hist2, alpha=0.07, color='blue')
    plt.show()


def gen_pop1d(start_pt,end_pt,amount,seednum = None):
    if seednum == None:
        seednum = 0
    np.random.seed(seednum)
    return np.random.randint(start_pt, end_pt, amount)

def data_hist_expand_1st_chunk(data_chunk, usestd = True,numbin=None,adjust_constant=None):
    avg = batch_mean(data_chunk)
    if numbin == None:
        numbin = 8
    if usestd:
        std = batch_stdev(data_chunk)  # Compute sample standard deviation.
        print(f'samp. std_interval is {std:.2f}')
        # print(f''"std_interval", std, "average of pop.", avg)
    else:
        max_val = max(data_chunk)
        min_val = min(data_chunk)
        std = (max_val - min_val) / numbin
        print("max-min std", std)
    if adjust_constant is None:
        adjust_constant = 0.02

    # list_size = len(hist_data)
    chunk_size = len(data_chunk)
    hist_data = data_chunk.copy()
    # hist_size = list_size
    # a = hist_data
    # avg = batch_mean(hist_data)
    # max_right_end = max(hist_data)
    # min_left_end = min(hist_data)

    avg = batch_mean(hist_data)
    max_right_end = max(hist_data)
    min_left_end = min(hist_data)
    std = (max_right_end - min_left_end) / 8
    expand = False
    sstd_hist = std_hist(chunk_size)
    exp_hist = data_hist(hist_data, avg, std)
    print("first chunk before adjust width")
    print(sstd_hist)
    print(exp_hist)
    print('-----------------')
    while (exp_hist[-2] - sstd_hist[-2]) > 4 or (exp_hist[-1] - sstd_hist[-1]) > 2 \
            or (exp_hist[1] - sstd_hist[1]) > 4 or \
            (exp_hist[0] - sstd_hist[0]) > 2:

        exp_fre = [exp_hist[-2], exp_hist[-1], exp_hist[1], exp_hist[0]]
        sstd_fre = [sstd_hist[-2], sstd_hist[-1], sstd_hist[1], sstd_hist[0]]
        exp_porp = [a/sum(exp_fre)*sum(sstd_fre) for a in exp_fre]
        statistic, p_value = chisquare(exp_porp, sstd_fre)
        # print(f'p-value is : {p_value:.2f}')
        # print(">>>>> expand first chunk 0")
        # print(sstd_hist)
        # print(exp_hist)
        if p_value < 0.05:
            expand = True
            # max_right_end = max(hist_data)
            # min_left_end = min(hist_data)
            difference_max = exp_hist[-1] - sstd_hist[-1]  # The rightmost bin
            if difference_max > 0:
                max_right_end = max_right_end + adjust_constant * std
                hist_data.append(max_right_end)

            difference_min = exp_hist[0] - sstd_hist[0]  # The leftmost bin
            if difference_min > 0:
                min_left_end = min_left_end - adjust_constant * std
                hist_data.append(min_left_end)

        # print("expand: min left end", min_left_end, "max right end", \
        # max_right_end)

            avg = batch_mean(hist_data)
            max_right_end = max(hist_data)
            min_left_end = min(hist_data)
            std = batch_stdev(hist_data)
            exp_hist = data_hist(data_chunk, avg, std)  # compute the new hist based on the new avg and std.
            prev_size = len(hist_data)

            print('-----------')
            print(exp_hist)
            print(sstd_hist)
            print('-----------')
    # if expand_count > 0:
    #     hist_size = len(hist_data)
    if expand is False:
        prev_size = chunk_size
    return expand, exp_hist,prev_size, avg, std, max_right_end, min_left_end

def data_hist_expand_kth_chunk(new_data_chunk, prev_size, exp_hist, avg, std, max_right_end, min_left_end,
                               adjust_constant = None):
    current_list_size = len(new_data_chunk)
    total_size = prev_size + current_list_size
    hist_data = new_data_chunk.copy()
    if adjust_constant is None:
        adjust_constant = 0.02
    for t in new_data_chunk:
        if avg - 4 * std <= t < avg - 3 * std:
            exp_hist[0] += 1
        if avg + 3 * std <= t < avg + 4 * std:
            exp_hist[-1] += 1
    # Create standard histogram
    sstd_hist = std_hist(total_size)
    expand = False
    prev_avg = avg


    while (exp_hist[-2] - sstd_hist[-2]) > 4 or (exp_hist[-1] - sstd_hist[-1]) > 2 \
            or (exp_hist[1] - sstd_hist[1]) > 4 or \
            (exp_hist[0] - sstd_hist[0]) > 2:
        exp_fre = [exp_hist[-2], exp_hist[-1], exp_hist[1], exp_hist[0]]
        sstd_fre = [sstd_hist[-2], sstd_hist[-1], sstd_hist[1], sstd_hist[0]]
        exp_porp = [a / sum(exp_fre) * sum(sstd_fre) for a in exp_fre]
        statistic, p_value = chisquare(exp_porp, sstd_fre)
        if p_value<0.05: # Reject H0
            # print(">>>>> expand other chunk", i)
            expand = True
            difference_max = exp_hist[-1] - sstd_hist[-1]
            # print("diff max", difference_max)
            if difference_max > 0:
                max_right_end += (adjust_constant * std)
                hist_data.append(max_right_end)

                total_size += + 1
            # print("expand: min left end", min_left_end, "max right end", \
            #         max_right_end)

            difference_min = exp_hist[0] - sstd_hist[0]
            # print("diff min", difference_min)
            if difference_min > 0:
                min_left_end -= (adjust_constant * std)
                hist_data.append(min_left_end)

                total_size += 1

            print("expand: min left end", min_left_end, "max right end", max_right_end)

            current_size = len(hist_data)
            avg = (prev_size / total_size) * prev_avg + (current_size / total_size) * batch_mean(hist_data)
            prev_avg = avg
            std = (max_right_end - min_left_end) / 8
            prev_size = total_size

            # new_data_list_size = len(new_data_chunk)
            exp_hist[0] = 0
            exp_hist[-1] = 0

            for t in new_data_chunk:
                if avg - 4 * std <= t < avg - 3 * std:
                    exp_hist[0] += 1
                if avg + 3 * std <= t < avg + 4 * std:
                    exp_hist[-1] += 1
        else:
            prev_size = total_size
    if expand is False:
        prev_size = total_size

    return expand, exp_hist, prev_size, avg, std,max_right_end, min_left_end

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
    #=====>  1. Generate population
    # amount_population = opt.npop  # ---- must be <= start-end value

    # start_interval_random_value = opt.stapt
    # end_interval_random_value = opt.endpt
    expand_track = []
    randomlist = gen_pop1d(opt.stapt,opt.endpt,opt.npop)
    adjust_constant = opt.acons
    # Create the first data chunk a
    # expansion = 0
    chunk_size = 300
    a = list(randomlist[range(0, chunk_size)])
    # count = 1
    expand, exp_hist, prev_size, avg, std, max_right_end, min_left_end = data_hist_expand_1st_chunk(a)
    expand_track.append(expand)
    # prev_avg = avg
    print("\nfirst chunk prev size", prev_size)
    prev_std = std

    for i in range(1, 11):
        print("\n\n------------------- new chunk", i, "----------------------")
        k = i * chunk_size
        new_data_chunk = []
        for j in randomlist[k:k + chunk_size]:
            new_data_chunk.append(j)
            # a.append(j)
        expand, exp_hist, prev_size, avg, std, max_right_end, min_left_end = data_hist_expand_kth_chunk(new_data_chunk,
                                                                                    prev_size, exp_hist, avg, std, max_right_end, min_left_end)
        expand_track.append(expand)
        print(f'# of pop: {len(randomlist[0:int(i * chunk_size + chunk_size)])}')
        print(f'mean of pop: {batch_mean(randomlist[0:int(i * chunk_size + chunk_size)]):.2f}')
        print(f'mean of est: {avg:.2f}')
        print(f'std of pop: {batch_stdev(randomlist[0:int(i * chunk_size + chunk_size)]):.2f}')
        print(f'std of est: {std:.2f}')
        print(expand_track)
        # print("from prev lstd4", exp_hist[0], "rstd4", exp_hist[-1])
        # current_list_size = len(new_data_chunk)
        # total_size = prev_size + len(new_data_chunk)
        # print("\ntotal size: prev chunk and new chunk", total_size)
        # expand_count, exp_hist, avg, std,max_right_end, min_left_end, prev_size = data_hist_expand_kth_chunk(exp_hist, new_data_chunk, avg, std, max_right_end, min_left_end, prev_size)
        # if expand_count>0:
        #     sstd_hist = std_hist(total_size)
        #     print("\nadjust width new chunk")
        #     print(sstd_hist)
        #     print(exp_hist)
        #     myplot_histogram(sstd_hist, exp_hist, i, expand_count)


def parse_args(known=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--npop', dest='# of generated population', type=int, default=6000)
    parser.add_argument('--acons', dest='adjusted value of expanding constant', type=float, default=0.02)
    parser.add_argument('--stapt', dest='minimum value of 1 d data frame', type=float, default=1)
    parser.add_argument('--endpt', dest='maximum value of 1 d data frame',type=float, default=3000)
    parser.add_argument('--chsize', dest='size of a data chunk', type=int, default=300)
    #     parser.add_argument('--y', type=int, default=2)
    return parser.parse_known_args()[0] if known else parser.parse_args()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    opt = parse_args()
    main(opt)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
