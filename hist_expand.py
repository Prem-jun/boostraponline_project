# File is created by premjunsawang 
import numpy as np
import copy
class boostrap_exp:
    def bootstrap(input_data, number_bootstrap_iteration=600):
        data_set = copy.deepcopy(input_data)
        bootstrap_means = np.zeros(number_bootstrap_iteration)
        bootstrap_std = []
        bootstrap_max_diff_dist_std = []

        size_data_set = len(data_set)
        previous_bootstrap_mean = 0

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

        final_estimated_std = 2 * (estimated_std + estimated_std_of_std) + estimated_diff_dist_std

        print(" ")
        print("final_estimated_std:", final_estimated_std)

        return final_estimated_std