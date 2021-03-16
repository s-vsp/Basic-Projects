# !!! All data belongs to UCI and comes from https://archive.ics.uci.edu/ml/datasets.php !!!
# Link to download: https://archive.ics.uci.edu/ml/datasets/Parkinsons


# -*- coding: utf-8 -*-
"""
Created on Sat Mar 06 11:19:35 2021

@author: Kamil
"""

##### DECISION TREE TO CLASSIFY PARKINSON DISEASE #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from sklearn.metrics import plot_confusion_matrix

DataFrame = pd.read_csv("parkinsons.data")


# missing values checking
missing_values = DataFrame.isnull().sum()


y = DataFrame["status"]
# getting rid of "name" at the start as it is not a feature that has a real impact on classifying diseases
X = DataFrame.drop(["status", "name"], axis=1)


"""
Want to split the data into 3 sets - train, cross-validation and test
The proportion we would like to have is 60,20,20
We will perform T-T split twice -> first we will get a train set and test set and then we will split train set
into the final train set and cross-validation set

1) 100% -> 80% & 20% (test_size = 0.2) from 195 -> 156 and 39
2) counting 20% with respect to the starting set: 156 - 100%; 39 - unknown % ----> unknown% = 39 * 100 / 156 = 25%
   so the second split: (test_size = 0.25)
"""


# first split
X_train_start, X_test, y_train_start, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# second split
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X_train_start, y_train_start, test_size=0.25, 
                                                                                random_state=1, stratify=y_train_start)


# choosing hiperparameters based on the accuracy score performance on training and cross-validation set
# want to pick depth of the tree and max features when looking for a split
"""
[max_depth   max_features   train_score   validation_score]
21 * 22 rows, 4 columns
"""


hiperparameter_selection_matrix = np.zeros((21*22,4))

row = 0
for depth in range(1,22):
    for feature_max in range(1,23):
        
        decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth, max_features=feature_max, random_state=1)
        decision_tree.fit(X_train, y_train)
        train_score = decision_tree.score(X_train, y_train)
        validation_score = decision_tree.score(X_validation, y_validation)
        
        hiperparameter_selection_matrix[row,:] = depth, feature_max, train_score, validation_score
        row = row + 1

     
        
# let me do it also separately for a better and easier visualization of the learning curves and let's start with the depth
depth_selection_matrix = np.zeros((20,3))
dsm_row = 0
for i in range(1,21):
    decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=i, random_state=1)
    decision_tree.fit(X_train, y_train)
    train_score = decision_tree.score(X_train, y_train)
    validation_score = decision_tree.score(X_validation, y_validation)
    depth_selection_matrix[dsm_row,:] = i,train_score,validation_score
    dsm_row = dsm_row + 1

plt.figure(dpi=300)
plt.plot(range(1,21),depth_selection_matrix[:,1],color="indigo",marker="*",label="Train set")
plt.plot(range(1,21),depth_selection_matrix[:,2],color="chocolate",marker="o",label="Cross Validation set")
plt.legend(loc="best")
plt.xlabel("Decision Tree maximum depth")
plt.ylabel("Accuracy Score")
plt.title("Maximum depth seleciton")
plt.xticks(np.arange(1, 21, step=2))
plt.show()

# Decision -> max_depth selection to be equal to 6 as the performance on cross validation set gives the best accuracy here

# max_features for a split selection, accorind max_depth = 6
max_features_selection_matrix = np.zeros((22,3))
mfsm_row = 0
for i in range(1,23):
    decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=1, max_features=i)
    decision_tree.fit(X_train, y_train)
    train_score = decision_tree.score(X_train, y_train)
    validation_score = decision_tree.score(X_validation, y_validation)
    max_features_selection_matrix[mfsm_row,:] = i,train_score,validation_score
    mfsm_row = mfsm_row + 1
    
plt.figure(dpi=300)
plt.plot(range(1,23),max_features_selection_matrix[:,1],color="indigo",marker="*",label="Train set")
plt.plot(range(1,23),max_features_selection_matrix[:,2],color="chocolate",marker="o",label="Cross Validation set")
plt.legend(loc="best")
plt.xlabel("Decision Tree maximum features considered in a split")
plt.ylabel("Accuracy Score")
plt.title("Maximum number of features selection")
plt.xticks(np.arange(1, 23, step=2))    
plt.show()

# here it looks like the optimum value might be either 5 or 16 -> let's choose 5 just in case to prevent overfitting,
# although 16 should be also a reasonable choice


# final model
decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6, max_features=5, random_state=1)
decision_tree.fit(X_train,y_train)

decision_tree_train_score = decision_tree.score(X_train, y_train)
decision_tree_test_score = decision_tree.score(X_test, y_test)

# predicting the disease on the new data (test set) and checking the performance using the confusion matrix
y_pred = decision_tree.predict(X_test)


fig, ax = plt.subplots(dpi=300)
plot_confusion_matrix(decision_tree, X_test, y_test, cmap="copper", display_labels=["No Parkinson", "Parkinson confirmed"], ax=ax)
plt.show()

# visualization of decisions in a tree
tree_data = export_graphviz(decision_tree,out_file=None, feature_names=X_train.columns, 
                             max_depth=6, filled=True, rounded=True, class_names=["0", "1"])

graph = pydotplus.graph_from_dot_data(tree_data)
graph.write_png("drzewko.png")