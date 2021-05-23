# Parkinson disease classification

Data belongs to UCI and can be found -> http://archive.ics.uci.edu/ml/datasets.php

Description of the files provided in this project's repository:
> main.ipynb - main Jupyter Notebook having all the computations <br />
> parkinson_cls.py - Python file, having all the same computations as main.ipynb, but everything is provided just in one script. Preferable to use in Spyder for better visualisations <br />
> plots - folder with all the plots and figures from the main file. Most saved in .svg vector format <br />
> OwnLibrary.py - Python file, mini-library having one function, used in the proceeding <br />
> parkinson.data - data file from UCI datasets
> raw_spyder_data.spydata - raw data saved from Spyder console


## 1. Data Preprocessing
> 1.1. Loading data <br /> 
> 1.2. Checking for missing values <br /> 
> 1.3. Outliers detection <br /> 
> 1.4. Basic data visulaizations <br /> 
>> 1.4.1. Correlation heatmap <br />
>> 1.4.2. Value counts <br />
>> 1.4.3. Skewness of features <br />
>
> 1.5. Log transform <br />

## 2. Modeling
> 2.1. Classifier selection <br />
> 2.2. Hyperparameters tuning <br />
>> 2.2.1. Nested Cross-validation
>
> 2.3. Learning curves <br />
> 2.4. Validation curves <br />
> 2.5. ROC curve and AUC <br />
> 2.6. Prediction <br />

## 3. Post-processing and final analysis
> 3.1. Confusion matrix <br />
> 3.2. Cost complexity pruning <br />
> 3.3. Trees comparison <br />
> 3.4. Tree visualization
