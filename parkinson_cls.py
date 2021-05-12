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

from KamilMachineLearning import IQR_technique

###-----------------------###
### 1. Data Preprocessing ###
###-----------------------###

## 1.1. Loading data ##

data = pd.read_csv("parkinsons.data")

# Moving labels column into the last spot as it is more convenience
temporary = data.status
data.drop(columns="status", inplace=True)
data = pd.concat([data, temporary], axis=1)

data_dsc = data.describe()


## 1.2. Checking for missing values ##

null_values = data.isnull().sum()

# There are no missing values in the dataset


## 1.3. Outliers detection ##
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

# Outliers are clearly visible, let's now use Tukey method to detect samples having multiple outliers

iqr = IQR_technique(data, 4, data.columns[1:23])
outliers_dict, multi_outliers = iqr.detect()
#print(f"\nIndices with corrseponding amount of outliers: {outliers_dict}")
#print(f"\nSamples having more than {iqr.min_outliers_number} outliers: {multi_outliers}")

# Some samples have a lot of outliers in different features, although as we are facing medical data, it is not so obvious
# to drop them off. Anyway it is nice to have them detected


## 1.4. Basic data visualizations ##

# 1.4.1. Correlation Heatmap #

plt.figure(dpi=300, figsize=(16,16))
sns.heatmap(data[data.columns[0:24]].corr(), annot=True, fmt=".2f", cmap="YlOrBr")
plt.title("Correlation heatmap", fontsize=30)
#plt.savefig("Correlation heatmap.svg")

# The main thing that should be intriguing us, is correlation between features and status-label.

# The strongest correlation with status is detected for "spread1" and "PPE", but still they are not on super-high level
# We can't deny status dependance on other features.


# 1.4.2. Class count #

plt.figure(dpi=150, figsize=(12,8))
sns.countplot(data["status"], palette="magma_r")
plt.title("Class count", fontsize=20)
#plt.savefig("Class count.svg")


# 1.4.3. Skewness of features #

skewness = [data[feature].skew() for feature in data.columns[1:23]]

plt.figure(figsize=(28,14), dpi=250)
for i, skew, feature in zip(range(1,23), skewness, data.columns[1:23]):
    plt.subplot(11,2,i)
    sns.distplot(data[feature], color="#411074", label="Skewness: %.2f"%(skew))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.suptitle("Skewness of features", fontsize=30)
#plt.savefig("Skewness of features.svg")
