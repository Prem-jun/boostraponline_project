import os
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import List, Dict
import pandas as pd
import numpy as np
# from utils.stat_test.check_assump import CheckAssumption
from utils.stat_test.stat_test import StatTest
import logging
# Setup logging
logging.basicConfig(level=getattr(logging, 'INFO'),
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )
logger = logging.getLogger(__name__)

@dataclass
class StatTestPipeline():
    # check_assumption = CheckAssumption()
    results_info: Dict = field(default_factory=dict)
    name_analys: List = field(default_factory=list)
    stat_res: List = field(default_factory=list)
    
    def insert_results(self,file_path: str, df: pd.DataFrame):
        self.results_info['file_path'] = file_path
        self.results_info['results_df'] = df
        # if self.stat_res is None:
        #     self.stat_res = []
        
    def run_analysis(self, col_group = ['config_file','chunk_size']):
        df_grouped = self.results_info['results_df'].groupby(col_group)
        logger.info(f"\nFound {len(df_grouped)} config_file-chunk_size combinations:")
        # for name, group in df_grouped:
        #     self.name_analys.append(name)
        for (config_file,chunk_size), grouped in df_grouped:
            logger.info(f"\n--- Analyzing {config_file}, chunk_size={chunk_size} ---")
            self.name_analys.append(f"{config_file}_chunk_size_{chunk_size}")
            stat_test = StatTest()
            stat_test.analysis(grouped)
            self.stat_res.append(stat_test)
    def print_results(self,file_results: str):
        logger.info(f"+++++++++ Results for {file_results} ++++++++++++\n")
        for idx, res in enumerate(self.stat_res):
            logger.info(f"\n-------- {self.name_analys[idx]} --------")
            print(f"base-method: {res.results['base']}/ sample mean: {res.results['base_mean']:.4f}")
            formatted_means = ', '.join(f"{v:.4f}" for v in res.results['other_mean'])
            print(f"compared-method: {res.results['other']}/ sample means: {formatted_means}")
            # print(f"compared-method: {res.results['other']}/ sample means: {res.results['other_mean']}")
            # print(f'H0: {res.results["H0"]}')
            print(f"paired-statistics: {res.results['statistic']}")
            formatted_p = ', '.join(f"{v:.4f}" for v in res.results['p_val'])
            print(f"p-value: {formatted_p}")
            formatted_stat = ', '.join(f"{v:.4f}" for v in res.results['stat_val'])
            print(f"statistic value: {formatted_stat}")
            formatted_rej = ' '.join(str(bool(x)).lower() for x in res.results['rejected'])
            print(f"rejected H0: {formatted_rej}")
            print("+"*15)
                    
    def write_results_txt(self, file_save:str ,output_dir: str = './results'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_path = os.path.join(output_dir, (file_save+'.txt'))
        with open(file_path, 'w') as f:
            f.write(f"+++++++++ Results for {file_save} ++++++++++++\n")
            # for idx,file_comnination in enumerate(self.name_analys):
            #     f.write(f"\n-------- {file_comnination} --------\n")
            for idx, res in enumerate(self.stat_res):
                f.write(f"\n-------- {self.name_analys[idx]} --------\n")
                header = [h for h in res.results.keys() if h != "H0"]
                f.write('\t'.join(header) + '\n')
                rows = zip_longest(*[
                        res.results[h] if isinstance(res.results[h], (list, tuple, np.ndarray)) else [res.results[h]]
                        for h in header
                    ], fillvalue='')
                for row in rows:
                    
                    formatted = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
                    f.write('\t'.join(formatted) + '\n')
                    
                f.write("\n\nSummary of all results:\n")
                
            # for res in self.stat_res:
            #     f.write('\t'.join(map(str, res.to_dict().values())) + '\n')
        # file_path = os.path.join(output_dir, (file_save+'.csv'))
        # results_df = pd.DataFrame([res.to_dict() for res in self.stat_res])
        # results_df.to_csv(file_path, index=False)
        # logger.info(f"Results saved to {file_path}")        

def main(results_dict,dist_key):
    pipelines = {}
    for result_file in results_dict[dist_key]:
        print(f"Processing file: {result_file}")
        file_name = os.path.splitext(os.path.basename(result_file))[0]
        if not os.path.exists(result_file):
            logger.error(f"File {result_file} does not exist.")
            continue
        
        # Load the data
        df = pd.read_csv(result_file)

        # Step 1: Group by config_file and chunk_size
        pipeline = StatTestPipeline()
        pipeline.insert_results(file_path=result_file, df=df)
        pipeline.run_analysis()
        # pipeline.write_results_txt(file_save=f"{dist_key}_{file_name}", output_dir='./results')
        pipeline.print_results(file_results=file_name)
        pipelines[file_name] = pipeline
    

if __name__ == "__main__":
    results_dict = {'fdist':['./config_sim_data/fdist/performance_summary.csv',
                            './config_sim_data/fdist/performance_summary_outlier.csv'],
            'wald':['./config_sim_data/wald/performance_summary.csv',
                    './config_sim_data/wald/performance_summary_outlier.csv'],
            'wiebull':['./config_sim_data/wiebull/performance_summary.csv',
                    './config_sim_data/wiebull/performance_summary_outlier.csv'],
             'chi2':['./config_sim_data/chi2/performance_summary.csv',
                    './config_sim_data/chi2/performance_summary_outlier.csv'],
             'normal':['./config_sim_data/normal/performance_summary.csv',
                    './config_sim_data/normal/performance_summary_outlier.csv'],
              'realworld':['./config_sim_data/realworld/performance_summary.csv']
             }
    dist_key = 'wiebull'
    main(results_dict,dist_key)