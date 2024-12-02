# Welcome to Boostraping project created by MkDocs.

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Main program
* `sim_data_pop.py` => simulate the population based on the predefined values in the `.yaml` file. For examples: `config_wald.yaml` and `config_wiebull.yaml`. The simulated data was saved in `.pkl` file.

* `sim_data_samp_chunk.py` => create the samples data chunks from the population file simulated from `sim_data_pop.py` and save the results into `.json` file.

* `lib_boostrap.py` => library file relating to the online boostrap functions.
 
* `main_boostrap.py` => the main program for executing boostrap online algorithms.

* `main_result_analysis.py` => the main program for analysing the results from `main_boostrap.py` 

## Project Results layout

    Python files `.py`    # The configuration file.
    Config_sim_data/
        `config_wald.yaml` - config file for simuling population data.
        wald/  # The documentation homepage.
            `.pkl` files - simulated population samples.
            `.json` files - streaming samples chunks
            `results_all_wald.xlsx` - online boostraping results
        ...       # Other markdown pages, images and other files.
