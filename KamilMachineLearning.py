# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:57:40 2021

@author: Kamil
"""

import pandas as pd
import numpy as np
from collections import Counter

class IQR_technique(object):
    
    """
    Tukey method to detect the outliers based on IQR
    
    Paramaters:
    ----------
    data: DataFrame object
        Data frame that is beeing processed
    min_outliers_number: int
        Minimal number of outliers to be detected
    features: array-like
        Feature names where we want to detect the outliers
    """

    
    def __init__(self, data, min_outliers_number, features):
        self.data = data
        self.min_outliers_number = min_outliers_number
        self.features = features
    
    def detect(self):
        """
        Returns dict of detected outliers and outlier indices, that occur more than min_outliers_number times
        """
        outliers_detected = []
        
        for feature in self.features:
            # computing 1st and 3rd quartiles (25% and 75%)
            Q1 = np.percentile(self.data[feature], 25)
            Q3 = np.percentile(self.data[feature], 75)
            
            # interquartile range - IGR
            IQR = Q3 - Q1
            
            # outlier step
            outlier_step = 1.5 * IQR
            
            outliers_list = self.data[(self.data[feature] < Q1 - outlier_step) | (self.data[feature] > Q3 + outlier_step)].index
            outliers_detected.extend(outliers_list)
            
        outliers_detected = Counter(outliers_detected)
        # taking the indices (rows in data frame object) that have multiple outliers -> in more than min_outliers_number columns
        multilple_outliers = list(key for key, value in outliers_detected.items() if value > self.min_outliers_number)
        
        return outliers_detected, multilple_outliers
    