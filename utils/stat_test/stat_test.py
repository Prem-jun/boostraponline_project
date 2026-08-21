from dataclasses import dataclass, field
from typing import List, Dict, Union
import os
# from typing import Tuple
import numpy as np
import pandas as pd
from scipy import stats
from utils.stat_test.check_assump import CheckAssumption
import logging
# Setup logging
logging.basicConfig(level=getattr(logging, 'INFO'),
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )
logger = logging.getLogger(__name__)

@dataclass
class ParametricTest:
    """
    Class to perform parametric statistical tests.
    
    Attributes:
        dir (str): Directory where results are stored.
        file (str): Configuration file name.
        outlier (bool): Flag to indicate if outlier detection is enabled.
    """
    H0: str = None
    # nsamples: list = field(default_factory=list)
    nsamples: int = None
    statistics: str = None
    p_value: float = None
    stat_value: float = None
    rejected: bool = None
    
    def paired_t_test(self, data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05):
        """
        Perform a paired t-test on two sets of data.
        
        Parameters:
            data1 (np.ndarray): First set of data.
            data2 (np.ndarray): Second set of data.
            alpha (float): Significance level for hypothesis testing.
        
        Returns:
            None
        """
        self.H0 = "means of the differenc of paired samples are not less than zero"
        self.nsamples: list = len(data1)
        if self.nsamples < 2:
            self.statistics = "Insufficient data"
            self.p_value = np.nan
            self.stat_value = np.nan
            self.rejected = False
            return None
        
        statistic, p_value = stats.ttest_rel(data1, data2,alternative='less')
        self.statistics = 'Paired t-test'
        self.p_value = p_value
        self.stat_value = statistic
        self.rejected = (p_value < alpha)
        
@dataclass
class NonParametricTest:
    """
    Class to perform non-parametric statistical tests.
    
    Attributes:
        dir (str): Directory where results are stored.
        file (str): Configuration file name.
        outlier (bool): Flag to indicate if outlier detection is enabled.
    """
    H0: str = None
    nsamples: int = None
    statistics: str = None
    p_value: float = None
    stat_value: float = None
    rejected: bool = None
    
    def wilcoxon_test(self, data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05):
        """
        Perform a Wilcoxon signed-rank test on two sets of data.
        
        Parameters:
            data1 (np.ndarray): First set of data.
            data2 (np.ndarray): Second set of data.
            alpha (float): Significance level for hypothesis testing.
        
        Returns:
            None
        """
        self.H0 = "medians of two paired samples are equal"
        self.nsamples = len(data1)
        
        if self.nsamples < 2:
            self.statistics = "Insufficient data"
            self.p_value = np.nan
            self.stat_value = np.nan
            self.rejected = False
            return None
        
        statistic, p_value = stats.wilcoxon(data1, data2,alternative='less')
        self.statistics = 'Wilcoxon signed-rank test'
        self.p_value = p_value
        self.stat_value = statistic
        self.rejected = (p_value < alpha)        

@dataclass
class StatTest:
    """
    Class to check assumptions for statistical tests.
    
    Attributes:
        H0 (str): Null hypothesis.
        nsamples (int): Number of samples.
        statistics (str): Type of statistical test used.
        p_value (float): P-value from the statistical test.
        stat_value (float): Test statistic value.
        accepted (bool): Whether the null hypothesis is accepted.
    """
    
    assumption_check: List = field(default_factory=list) #CheckAssumption()
    para_test: List = field(default_factory=list)
    stat_diff_test: List = field(default_factory=list)
    # nonpara_tes = NonParametricTest()
    alpha = float = 0.05
    results: Dict = field(default_factory=dict)
    
    def stat_report_identical_data(self):
        self.results['H0'].append('Cannot reject H0, data are identical')
        self.results['stat_val'].append(np.nan)
        self.results['p_val'].append(np.nan)
        self.results['statistic'].append('N/A')
        self.results['rejected'].append(False)
        
    def stat_report(self, test: Union[ParametricTest, NonParametricTest]):
        self.results['H0'].append(test.H0)
        self.results['stat_val'].append(test.stat_value)
        self.results['p_val'].append(test.p_value)
        self.results['statistic'].append(test.statistics)
        self.results['rejected'].append(test.rejected)
        # logger.info(f"Wilcoxon test result for {other} vs bt_est_on: "
        #             f"statistic={self.nonpara_tes.stat_value}, p_value={self.nonpara_tes.p_value}, "
        #             f"accepted={self.nonpara_tes.accepted}")    
        
    def analysis(self,grouped: pd.DataFrame,name: str = None):
        """
        Perform statistical analysis on the provided data.
        
        Parameters:
            data (np.ndarray): Input data for analysis.
            alpha (float): Significance level for hypothesis testing.
        
        Returns:
            None
        """
        base_estimator = 'bt_est_on_out'   
        col_measure = 'poperr_range' 
        estimator_groups = grouped.groupby('estimator')
        
        base_data = estimator_groups.get_group(base_estimator)[col_measure].values
        # list of the other estimators
        other_estimators = [est for est in estimator_groups.groups.keys() if est != base_estimator]
        self.results['base'] = base_estimator
        self.results['other'] = other_estimators
        self.results['base_mean'] = np.mean(base_data)
        self.results['other_mean'] = []
        self.results['H0'] = []
        self.results['stat_val'] = []
        self.results['p_val'] = []
        self.results['statistic'] = []
        self.results['rejected'] = []
        for other in other_estimators:
            # Calculate differences
            other_data = estimator_groups.get_group(other)[col_measure].values
            self.results['other_mean'].append(np.mean(other_data))
            if np.array_equal(base_data,other_data):
                self.stat_report_identical_data()
            else:
                differences = base_data - other_data
                self.assumption_check.append(CheckAssumption())
                self.assumption_check[-1].check_normality(differences, self.alpha)
                if self.assumption_check[-1].accepted: # Normality assumption accepted
                    self.para_test.append(True)
                    self.stat_diff_test.append(ParametricTest())
                    self.stat_diff_test[-1].paired_t_test(base_data, other_data, self.alpha)
                else: # Normality assumption not accepted
                    self.para_test.append(False)
                    self.stat_diff_test.append(NonParametricTest())
                    self.stat_diff_test[-1].wilcoxon_test(base_data, other_data, self.alpha)
                # Report the results    
                self.stat_report(self.stat_diff_test[-1])
                    # logger.info(f"Normality assumption accepted for {other} vs {base_estimator}")     
                    # self.para_test.paired_t_test(base_data, other_data, self.alpha)
        #             logger.info(f"Paired t-test result for {other} vs bt_est_on: "
        #                         f"statistic={self.para_test.stat_value}, p_value={self.para_test.p_value}, "
        #                         f"accepted={self.para_test.accepted}")
            # Check normality assumption
        
        
        
        
        # df_grouped = df.groupby(['config_file', 'chunk_size'])
        # self.df = df_grouped
        # print(f"\nFound {len(df_grouped)} config_file-chunk_size combinations:")
        # # for name, group in df_grouped:
        # #     print(f"  {name}: {len(group)} rows")
        # for (config_file,chunk_size), grouped in df_grouped:
        #     # Group by estimator within this config-chunk combination
        #     logger.info(f"\n--- Analyzing {config_file}, chunk_size={chunk_size} ---")
        #     estimator_groups = grouped.groupby('estimator')
        #     bt_est_on_out_data = estimator_groups.get_group('bt_est_on_out')['poperr_range'].values
        #     logger.info(f"bt_est_on group: {len(bt_est_on_out_data)} observations")
            
        #     # list the other estimators
        #     other_estimators = [est for est in estimator_groups.groups.keys() if est != 'bt_est_on']
            
        #     for other in other_estimators:
        #         # Calculate differences
        #         other_data = estimator_groups.get_group(other)['poperr_range'].values
        #         differences = bt_est_on_out_data - other_data
                
        #         self.assumption_check.check_normality(differences,self.alpha)
        #         if self.assumption_check.accepted:
        #             logger.info(f"Normality assumption accepted for {other} vs bt_est_on")
        #             # Perform paired t-test
        #             self.para_test.paired_t_test(bt_est_on_out_data, other_data, self.alpha)
        #             logger.info(f"Paired t-test result for {other} vs bt_est_on: "
        #                         f"statistic={self.para_test.stat_value}, p_value={self.para_test.p_value}, "
        #                         f"accepted={self.para_test.accepted}")
        #         else:
        #             logger.info(f"Normality assumption not accepted for {other} vs bt_est_on, "
        #                         f"using Wilcoxon signed-rank test")
        #             # Perform Wilcoxon signed-rank test
        #             self.nonpara_tes.wilcoxon_test(bt_est_on_out_data, other_data, self.alpha)
        #             logger.info(f"Wilcoxon test result for {other} vs bt_est_on: "
        #                         f"statistic={self.nonpara_tes.stat_value}, p_value={self.nonpara_tes.p_value}, "
        #                         f"accepted={self.nonpara_tes.accepted}")