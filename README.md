# boostraponline_project
1. `sim_data_pop.py` => simulate the population based on the predefined values in the `.yaml` file. For examples: `config_wald.yaml` and `config_wiebull.yaml`. The simulated data was saved in `.pkl` file.

2. `sim_data_samp_chunk.py` => create the samples data chunks from the population file simulated from `sim_data_pop.py` and save the results into `.json` file.

3. `lib_boostrap.py` => library file relating to the online boostrap functions.
 
4. `main_boostrap.py` => the main program for executing boostrap online algorithms.

5. `main_result_analysis.py` => the main program for analysing the results from `main_boostrap.py` 