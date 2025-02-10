# boostraponline_project
1. `sim_data_pop.py` or `sim_data_pop_v2.py` => simulate the population based on the predefined values in the `.yaml` file. For examples: `config_wald.yaml` and `config_wiebull.yaml`. The simulated data was saved in `.pkl` file.

2. `sim_data_samp_chunk.py` => create the samples data chunks from the population file simulated from `sim_data_pop.py` and save the results into `.json` file.

3. `lib_boostrap.py` => library file relating to the online boostrap functions.
 
4. `main_boostrap.py` => the main program for executing boostrap online algorithms.

5. `main_result_analysis.py` => the main program for analysing the results from `main_boostrap.py` 

## main_boostrap.py

    Step 1: Read population data file saved as list (json file).
    Step 2: For each data:
        Step 3:  
### expand_bt_online
    step 1: Compute the learned samples.
    step 2: Compute the min and max values (min_c and max_c) of the current chunk c.
    step 3: If min_c < v_min, then v_min = min_c.
    step 4: If max_c > v_max, then v_max = max_c.
    step 5: Compute average (avg) and standard deviation (sd) based on v_min and v_max.
    step 6: Construct theoritical distribution of 8 bins based on avg and std.
    step 7: Compute theoritical number of elements in the left bin (h_l) and the right bin (h_r).
    step 8: Find the number of elements falls into the left bin (n_l) and and the right bin (n_r).
    step 9: If n_l> h_l, perform bootstrap on the left elements falling in the leftmost bin to get 
            the update v_min.
    step 10: If n_r > h_r, perform boostrap on the right elements falling in the rightmost bin to get
            the update v_max.
    step 11: If v_max or v_min changed, go to step 5.
    step 12: Else stop.
