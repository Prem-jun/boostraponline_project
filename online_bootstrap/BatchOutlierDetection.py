import numpy as np
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class ZBatchOutlierDetector:
    threshold:float = None
    mean:float= None
    sd:float = None
    
    def add_init_params(self,threshold:float,mean:float, sd:float):
        self.threshold = threshold
        self.mean = mean
        self.sd = sd
    
    # def calculate_statistics(self, data):
    #     """Calculate mean and standard deviation for the batch"""
    #     n = len(data)
    #     if n == 0:
    #         return 0, float('inf')
        
    #     ## Calculate mean
    #     # mean = sum(data) / n
        
        
    #     # Calculate variance using two-pass method for better numerical stability
    #     if n < 2:
    #         return self.mean, float('inf')
            
    #     variance_sum = sum((x - mean) ** 2 for x in data)
    #     std = (variance_sum / (n - 1)) ** 0.5
        
    #     return mean, std
    
    def detect_outliers(self, data):
        """Detect outliers in a batch of data
        
        Args:
            data: List of numeric values
            
        Returns:
            List of booleans indicating whether each point is an outlier
        """
        if len(data) < 2:
            return [False] * len(data)
        
        # Calculate statistics from the entire batch
        # mean, std = self.calculate_statistics(data)
        mean = self.mean
        std = self.sd
        
        # Identify outliers
        outliers = []
        for x in data:
            if std == 0:  # Handle case where all values are identical
                outliers.append(False)
            else:
                z_score = abs(x - mean) / std
                outliers.append(z_score > self.threshold)
        
        return outliers
    
    def get_clean_data(self, data):
        """Return only the non-outlier points"""
        outliers = self.detect_outliers(data)
        return [x for i, x in enumerate(data) if not outliers[i]]
    
class EWMABatchOutlierDetector:
    """
        Initialize EWMA batch outlier detector
        
        Args:
            alpha: Smoothing factor (0 < alpha < 1) - controls how quickly the model adapts
                  Lower values give more weight to historical data
                  Higher values give more weight to recent data
            threshold: Z-score threshold for outlier detection
    """
    alpha:float = None
    threshold:float = None
    
    def add_init_params(self,alpha:float ,threshold:float):
        self.alpha = alpha
        self.threshold = threshold
    
    def detect_outliers(self, data):
        """
        Detect outliers using EWMA method
        
        Args:
            data: List of numeric values in temporal order
            
        Returns:
            List of booleans indicating whether each point is an outlier
        """
        n = len(data)
        if n < 2:
            return [False] * n
        
        # Initialize with first point
        ewma = data[0]
        ewmv = 0  # EWMA variance
        outliers = [False]  # First point can't be an outlier without prior data
        
        for i in range(1, n):
            # Current point
            x = data[i]
            
            # Compute deviation from EWMA
            deviation = x - ewma
            
            # Check if point is an outlier using current EWMA and variance
            if ewmv > 0:
                z_score = abs(deviation) / (ewmv ** 0.5)
                is_outlier = z_score > self.threshold
            else:
                is_outlier = False
            
            outliers.append(is_outlier)
            
            # Update EWMA and variance (only for non-outliers)
            if not is_outlier:
                # Update EWMA
                ewma = self.alpha * x + (1 - self.alpha) * ewma
                
                # Update EWMA variance
                ewmv = (1 - self.alpha) * (ewmv + self.alpha * deviation ** 2)
        
        return outliers
    
    def get_outlier_indices(self, data):
        """Return the indices of outliers"""
        outliers = self.detect_outliers(data)
        return [i for i, is_outlier in enumerate(outliers) if is_outlier]
    
    def get_clean_data(self, data):
        """Return only the non-outlier points"""
        outliers = self.detect_outliers(data)
        return [x for i, x in enumerate(data) if not outliers[i]]
    
    def get_ewma_values(self, data):
        """
        Calculate the EWMA values for the dataset
        Useful for visualization or further analysis
        """
        if not data:
            return []
            
        ewma_values = [data[0]]  # Initialize with first point
        ewma = data[0]
        
        for i in range(1, len(data)):
            x = data[i]
            # Skip updating EWMA if point is an outlier
            outliers = self.detect_outliers(data[:i+1])
            if not outliers[-1]:
                ewma = self.alpha * x + (1 - self.alpha) * ewma
            ewma_values.append(ewma)
            
        return ewma_values    
      