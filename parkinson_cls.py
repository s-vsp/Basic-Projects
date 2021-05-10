# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:31:11 2021

@author: Kamil
"""

##### PARKINSON DISEASE CLASSIFICATION PROBLEM #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##-----------------------##
## 1. Data Preprocessing ##
##-----------------------##

# 1.1. Loading data #

data = pd.read_csv("parkinsons.data")

# Moving labels column into the last spot as it is more convenience
temporary = data.status
data.drop(columns="status", inplace=True)
data = pd.concat([data, temporary], axis=1)

data_dsc = data.describe()


# 1.2. Checking for missing values #

null_values = data.isnull().sum()

# There are no missing values in the dataset


# 1.3. Outliers detection #
"""
Outliers can have a significant effect on the whole process of predictions and data analysis, it is important to 
handle with them properly.

Firstly I decided to to visualize them on boxplots.

Then I decided to use the Tukey fences technique, which is based on interquartile range (IQR)
"""

plt.figure(dpi=350, figsize=(18,20))

# Due to different ranges of the values for each feature I hard-coded them into each subplot for a good visualization

plt.subplot(5,2,1)
sns.boxplot(data=data[data.columns[0:4]], palette="YlOrBr")

plt.subplot(5,2,2)
sns.boxplot(data=data[["MDVP:Jitter(%)", "MDVP:RAP", "MDVP:PPQ"]], palette="YlOrBr")

plt.subplot(5,2,3)
sns.boxplot(data=data[["Jitter:DDP", "Shimmer:APQ3", "Shimmer:APQ5", "NHR"]], palette="YlOrBr")

plt.subplot(5,2,4)
sns.boxplot(data=data[["MDVP:Shimmer", "MDVP:APQ", "Shimmer:DDA"]], palette="YlOrBr")

plt.subplot(5,2,5)
sns.boxplot(data=data[["MDVP:Shimmer(dB)","spread2","PPE"]], palette="YlOrBr")

plt.subplot(5,2,6)
sns.boxplot(data=data[["DFA", "RPDE"]], palette="YlOrBr")

plt.subplot(5,2,7)
sns.boxplot(data=data[["spread1"]], palette="YlOrBr")

plt.subplot(5,2,8)
sns.boxplot(data=data[["HNR"]], palette="YlOrBr")

plt.subplot(5,2,9)
sns.boxplot(data=data[["MDVP:Jitter(Abs)"]], palette="YlOrBr")

plt.subplot(5,2,10)
sns.boxplot(data=data[["PPE"]], palette="YlOrBr")

plt.suptitle("Boxplots to visualize outliers", fontsize=30)
#plt.savefig("Boxplots to visualize outliers.svg")

# Detecting outliers