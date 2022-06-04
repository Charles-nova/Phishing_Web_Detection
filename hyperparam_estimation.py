# -*- coding: utf-8 -*-
# @Time    : $2022-5-21
# @Author  : $Yidong Ding
# @File    : $hyperparam_estimation.py
# @Software: $Pycharm

"""
To find the best hyper-parameters in SVM
Two methods:random search or grid search
"""
import numpy as np
import pandas as pd
from time import time
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import svm


# get some data
data = pd.read_csv("./dataset/phishing.csv")
features = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 27, 29]]
X = features.values
y = data.iloc[:, -1]

# build a classifier
clf = svm.SVC()


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "C": np.linspace(0.01, 10, num=20),
}

param_gamma = {
    "kernel": ['rbf'],
    "C": [7.37],
    "gamma": ['scale', 'auto'],
}

# run randomized search
n_iter_search = 2
random_search = RandomizedSearchCV(
    clf, param_distributions=param_gamma, n_iter=n_iter_search
)

start = time()
random_search.fit(X, y)
print(
    "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
    % ((time() - start), n_iter_search)
)
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "C": np.linspace(0.01, 10, num=20),
}


# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print(
    "GridSearchCV took %.2f seconds for %d candidate parameter settings."
    % (time() - start, len(grid_search.cv_results_["params"]))
)
report(grid_search.cv_results_)