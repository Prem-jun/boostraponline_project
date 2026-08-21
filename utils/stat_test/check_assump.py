from dataclasses import dataclass
import os
# from typing import Tuple
import numpy as np
from scipy import stats

@dataclass
class CheckAssumption:
    """
    Class to check assumptions for statistical tests.
    
    Attributes:
        dir (str): Directory where results are stored.
        file (str): Configuration file name.
        outlier (bool): Flag to indicate if outlier detection is enabled.
    """
    H0: str = None
    nsamples: int = None
    statistics: str = None
    p_value: bool = float
    stat_value: float = None
    accepted: bool = None
    
    def check_normality(self,data: np.ndarray, alpha: float = 0.05):
        """
        Check normality using Shapiro-Wilk test.
        Returns (is_normal, p_value)
        """
        self.H0 = "data is normally distributed"
        self.nsamples = len(data)
        
        if len(data) < 3:
            self.statistics = "Insufficientkck data"
            self.p_value = np.nan
            self.stat_value = np.nan
            self.accepted = False
            return None
            
        if len(data) >= 30:
            self.statistics = 'Central Limit Theorem'
            self.p_value = np.nan
            self.stat_value = np.nan
            self.accepted = True
        else:

            # Use Shapiro-Wilk test for normality
            statistic, p_value = stats.shapiro(data)
            self.statistics = 'Shapiro-Wilk'
            self.p_value = p_value
            self.stat_value = statistic
            self.accepted = (p_value > alpha)
            return None

    # def __post_init__(self):
    #     # Ensure the directory exists
    #     os.makedirs(self.dir, exist_ok=True)