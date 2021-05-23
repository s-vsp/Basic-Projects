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

## 1.5. Log transformation ##

# Performing log transform on the mostly skewed features along the dataset
# For the statistical model the tail region may act as an outlier, so instead of dropping samples
# having outliers as mentioned before, I decided to perform log transform on those with highest skewness

skewed_feature = data.columns[[skewness.index(i)+1 for i in skewness if i > 3.3]] # +1 as there is still 'name' feature

# 'MDVP:RAP', 'Jitter:DDP', 'NHR' are the ones, to perform log transform on them


plt.figure(dpi=250, figsize=(24, 6))
plt.subplot(2,3,1)
sns.distplot(data["MDVP:RAP"], color="#411074", label="Skewness: %.2f"%(skewness[5]))
plt.legend(loc="best")
plt.tight_layout()

plt.subplot(2,3,2)
sns.distplot(data["Jitter:DDP"], color="#411074", label="Skewness: %.2f"%(skewness[7]))
plt.legend(loc="best")
plt.tight_layout()
plt.title("Before Log transform", fontsize=18)

plt.subplot(2,3,3)
sns.distplot(data["NHR"], color="#411074", label="Skewness: %.2f"%(skewness[14]))
plt.legend(loc="best")
plt.tight_layout()


data["MDVP:RAP"] = data["MDVP:RAP"].map(lambda x: np.log(x))
data["Jitter:DDP"] = data["Jitter:DDP"].map(lambda x: np.log(x))
data["NHR"] = data["NHR"].map(lambda x: np.log(x))


plt.subplot(2,3,4)
sns.distplot(data["MDVP:RAP"], color="#af315b", label="Skewness: %.2f"%(data["MDVP:RAP"].skew()))
plt.legend(loc="best")
plt.tight_layout()

plt.subplot(2,3,5)
sns.distplot(data["Jitter:DDP"] , color="#af315b", label="Skewness: %.2f"%(data["Jitter:DDP"].skew()))
plt.legend(loc="best")
plt.tight_layout()
plt.title("After Log Transform", fontsize=18)

plt.subplot(2,3,6)
sns.distplot(data["NHR"], color="#af315b", label="Skewness: %.2f"%(data["NHR"].skew()))
plt.legend(loc="best")
plt.tight_layout()

plt.suptitle("Log transform", fontsize=26, y =1.15, x=0.51)
#plt.savefig("Log transform.svg", bbox_inches="tight")

## 1.6. Train and test set ##

X = data.drop(["name", "status"], axis=1)
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=True, random_state=1, stratify=y)



###-------------###
### 2. Modeling ###
###-------------###

## 2.1. Classifier selection ##

# Decided to go for Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

tree = DecisionTreeClassifier(criterion="gini", random_state=1)



## 2.2. Hyperparameters tunning ##

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV



# 2.2.1. Nested Cross Validation #

# Decided to perform both tuning and validating with a 10x2 Nested Cross Validation technique

kfold_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # learning
kfold_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1) # hyperparams tuning

params = {"max_depth": list(range(3,10)), "max_features": list(range(1,23))}

searching = RandomizedSearchCV(tree, params, cv=kfold_inner, scoring="accuracy", n_iter=20)
scores = cross_val_score(searching, X_train, y_train, cv=kfold_outer, scoring="accuracy")
print('Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

searching.fit(X_train, y_train)
print("Best params: ", searching.best_params_)

print("Best score: ", searching.best_score_)

# Parameters such as min_samples_leaf or max_leaf_nodes were not tuned on the purpose, so the tree might have been 
# overfitted. It will be later on reduced, using a post-training pruning technique, like cost-complexity pruning.

tree_classifier = searching.best_estimator_


## 2.3. Learning curves ##

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(estimator=tree_classifier, X=X_train, y=y_train, 
                                                       train_sizes=np.linspace(0.1,1,10), cv=kfold_outer)

# kfold_outer iterations (10) for each train_size (10) -> shapes of scores: [10,10]

train_mean = np.mean(train_scores, axis=1) # in rows
val_mean = np.mean(val_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(dpi=200, figsize=(14,7))
plt.plot(train_sizes, train_mean, color="#411074", marker="o", label="Train accuracy")
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.2, color="#411074")

plt.plot(train_sizes, val_mean, color="#eb6527", marker="o", label="Cross-validation accuracy")
plt.fill_between(train_sizes, val_mean + val_std, val_mean - val_std, alpha=0.2, color="#eb6527")

plt.grid(True)
plt.xlabel("Number of training samples")
plt.ylabel("Mean accuracy")
plt.title("Learning curves")
plt.legend(loc="lower right", fontsize=18)
#plt.savefig("Learning curves.svg", bbox_inches="tight")

# While having over +/- 130 training samples the gap between training and validation accuracy tends to increase, 
# which corresponds to a progressive overfitting.


## 2.4. Validation curves ##

# Model's performance will be analysed for both hyperparameters (max_depth and max_features) tuned in stage 2.2.

from sklearn.model_selection import validation_curve

def get_val_curve(param_name, param_range, param_label):
    train_scores_vc, val_scores_vc = validation_curve(estimator=tree, X=X_train, y=y_train, 
                                                      param_name=param_name, param_range=param_range)

    train_mean_vc = np.mean(train_scores_vc, axis=1)
    val_mean_vc = np.mean(val_scores_vc, axis=1)
    train_std_vc = np.std(train_scores_vc, axis=1)
    val_std_vc = np.std(val_scores_vc, axis=1)

    fig = plt.plot(param_range, train_mean_vc, color="#411074", marker="o", label="Train accuracy")
    fig = plt.fill_between(param_range, train_mean_vc + train_std_vc, train_mean_vc - train_std_vc, alpha=0.2, color="#411074")

    fig = plt.plot(param_range, val_mean_vc, color="#eb6527", marker="o", label="Cross-validation accuracy")
    fig = plt.fill_between(param_range, val_mean_vc + val_std_vc, val_mean_vc - val_std_vc, alpha=0.2, color="#eb6527")

    fig = plt.grid(True)
    fig = plt.xlabel(param_label)
    fig = plt.ylabel("Mean accuracy")
    fig = plt.title(param_label + " validation curves")
    fig = plt.legend(loc="lower right")

    return fig

plt.figure(dpi=200, figsize=(14,7))
for i, param_name, param_range, param_label in zip([1,2], ["max_depth", "max_features"], 
                                                   [list(range(3,10)), list(range(1,23))], ["Max depth", "Max features"]):
    plt.subplot(2,1,i)
    get_val_curve(param_name, param_range, param_label)
    plt.tight_layout()
    plt.suptitle("Validation curves", fontsize=18)
#plt.savefig("Validation curves.svg", bbox_inches="tight")

# In fact, the best hyperparameters combination gained during RandomSearch seems to be different to the ones we can 
# conclude from the plots. The reason for that is that the procedure of plotting validation curves was performed only 
# on a single hyperparameter and it's only purpose here was to visualize how it affects the accuracy score.


## 2.5. ROC curve and AUC ##

from sklearn.metrics import roc_curve, auc

tree_classifier.fit(X_train, y_train)
probas = tree_classifier.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas[:,1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(dpi=200, figsize=(14,7))
plt.plot(fpr, tpr, color="#411074", marker="o", label="ROC (AUC = %.2f)" % (roc_auc))
plt.plot([0, 1], [0,1], linestyle="--", color="gray", label="Random guessing")
plt.plot([0,0,1], [0,1,1], linestyle=":", color="black", label="Perfect performance")
plt.legend(loc="best")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve", fontsize=18)
#plt.savefig("ROC curve.svg", bbox_inches="tight")


## 2.6. Prediction

y_pred = tree_classifier.predict(X_test)
acc_score = tree_classifier.score(X_test, y_test)

# Model seems to have a pretty good accuracy on the new (test) dataset



###---------------------------------------###
### 3. Post-processing and final analysis ###
###---------------------------------------###


## 3.1. Confusion matrix ##

from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(figsize=(10,10))
plot_confusion_matrix(tree_classifier, X_test, y_test, cmap="magma_r", display_labels=["0", "1"], ax=ax)
plt.title("Confusion matrix", fontsize=20, y=1.03)
#plt.savefig("Confusion matrix.svg", bbox_inches="tight")


## 3.2. Cost complexity pruning ##

# To avoid model overfitting, post-training pruning technique will be performed. 
# Nodes with least value of effective alpha coefficent should be pruned

path = tree_classifier.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities

print("Effective alpha coefficents: \n", ccp_alphas)
print("\nImpurities: \n", impurities)

plt.figure(dpi=250, figsize=(14,7))
plt.step(ccp_alphas, impurities, color="#411074", marker="o", where="post")
plt.xlabel("Effective alpha")
plt.ylabel("Total impurity of leaves")
plt.title("Impurity in function of effective alpha", fontsize=14)
#plt.savefig("Impurity in function of effective alpha.svg", bbox_inches="tight")

classifiers = []
for ccp_alpha in ccp_alphas:
    tree_ccp = DecisionTreeClassifier(max_depth=7, max_features=1, random_state=1, ccp_alpha=ccp_alpha) 
    # params should be changed according to the ones obtained in RandomSearch due to randomness seed
    tree_ccp.fit(X_train, y_train)
    classifiers.append(tree_ccp)

# Tree with the highest ccp_alpha is just a root node
print("Number of nodes in the last tree: ", classifiers[-1].tree_.node_count)
print("ccp_alpha for the last tree: ", ccp_alphas[-1])
print("Max depth of the last tree: ", classifiers[-1].tree_.max_depth)

training_score = [classifier.score(X_train, y_train) for classifier in classifiers]
testing_score = [classifier.score(X_test,y_test) for classifier in classifiers]

plt.figure(dpi=200, figsize=(14,7))
plt.plot(ccp_alphas, training_score, color="#411074", marker="o", label="Train accuracy")
plt.plot(ccp_alphas, testing_score, color="#eb6527", marker="o", label="Test accuracy")
plt.xlabel("Effective alpha")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.title("Accuracy vs effective alpha", fontsize=14)
#plt.savefig("Accuracy vs effective alpha.svg", bbox_inches="tight")


# Final decision tree
# Selected the one giving the highest test set accuracy score and being closest to train set accuracy score (which is still high)

final_tree = classifiers[5]
final_tree

final_tree.fit(X_train, y_train)
final_tree.predict(X_test)

final_score = final_tree.score(X_test,y_test)
print(final_score)

## 3.3. Trees comparison ##

print("Accuracy score before pruning: ", round(tree_classifier.score(X_test, y_test), 2))
print("Accuracy score after pruning: ", round(final_tree.score(X_test, y_test), 2))
print("---------------------------------------")
print("Max depth before pruning: ", tree_classifier.tree_.max_depth)
print("Max depth after pruning: ", final_tree.tree_.max_depth)
print("---------------------------------------")
print("Number of nodes before pruning: ", tree_classifier.tree_.node_count)
print("Number of nodes after pruning: ", final_tree.tree_.node_count)
print("---------------------------------------")
print("Features importances before pruning: \n", tree_classifier.feature_importances_)
print("Features importances after pruning: \n", final_tree.feature_importances_)
print("---------------------------------------")
print("Effective alpha coefficent before pruning: ", tree_classifier.ccp_alpha)
print("Effective alpha coefficent after pruning: ", final_tree.ccp_alpha)


## 3.4. Tree visualization ##

import collections

tree_data = export_graphviz(final_tree,out_file=None, feature_names=X_train.columns, 
                             max_depth=7, filled=True, rounded=True, class_names=["0", "1"])

graph = pydotplus.graph_from_dot_data(tree_data)

colors = ('#9f2f7d', '#fec68b')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

#graph.write_png("Tree visualization.png")

