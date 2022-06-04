# -*- coding: utf-8 -*-
# @Time    : $ 2022-5-22
# @Author  : $ Yidong Ding
# @File    : $ roc.py
# @Software: $ Pycharm

"""
Compute Receiver operating characteristic (ROC)
"""

import numpy as np
from sklearn.metrics import precision_score, roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def plt_roc(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1)  # 数据集划分
    clf1 = LinearDiscriminantAnalysis()
    clf2 = svm.SVC(kernel='rbf', C=7.37)
    clf3 = DecisionTreeClassifier(max_depth=5, random_state=5)
    clf4 = RandomForestClassifier(max_depth=7, random_state=0)
    clf5 = MLPClassifier(learning_rate_init=0.001)
    y_scores_fisher = clf1.fit(x_train, y_train).decision_function(x_test)
    y_scores_svm = clf2.fit(x_train, y_train).decision_function(x_test)
    temp = clf3.fit(x_train, y_train).predict_proba(x_test)
    y_scores_DT = temp[:, 1]
    temp = clf4.fit(x_train, y_train).predict_proba(x_test)
    y_scores_RF = temp[:, 1]



    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr['fisher'], tpr['fisher'], roc_auc['fisher'] = calc_roc(y_test, y_scores_fisher)
    fpr['svm'], tpr['svm'], roc_auc['svm'] = calc_roc(y_test, y_scores_svm)
    fpr['DT'], tpr['DT'], roc_auc['DT'] = calc_roc(y_test, y_scores_DT)
    fpr['RF'], tpr['RF'], roc_auc['RF'] = calc_roc(y_test, y_scores_RF)

    plt.plot(fpr["fisher"], tpr["fisher"],
             label="ROC curve of FisherDecision (area = {:.2f})"
                   .format(roc_auc["fisher"]), linestyle=':', linewidth=3)
    plt.plot(fpr["svm"], tpr["svm"],
             label="ROC curve of SVM (area = {:.2f})"
                   .format(roc_auc["svm"]), linestyle=':', linewidth=3)
    plt.plot(fpr["DT"], tpr["DT"],
             label="ROC curve of DecisionTree (area = {:.2f})"
                   .format(roc_auc["DT"]), linestyle=':', linewidth=3)
    plt.plot(fpr["RF"], tpr["RF"],
             label="ROC curve of RandomForest (area = {:.2f})"
                   .format(roc_auc["RF"]), linestyle=':', linewidth=3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Classifiers')
    plt.legend(loc="lower right")
    plt.show()


def calc_roc(y_test, y_scores):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_score = roc_auc_score(y_test, y_scores)

    return fpr, tpr, roc_score