# -*- coding: utf-8 -*-
# @Time    : $2022-5-20
# @Author  : $Yidong Ding
# @File    : $classifies.py
# @Software: $Pycharm
"""
define classifiers

accuracy, precision_score, F1_score, recall_score
"""
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
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

path = "./param/model.pkl"


def myKNN(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1)  # 数据集划分
    neigh = KNeighborsClassifier(n_neighbors=10)  # 定义KNN分类器
    neigh.fit(x_train, y_train)  # 训练模型
    y_pred = neigh.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print("f1_ score:", round(f1, 3), '  recall_score:', round(recall, 3), ' precision_score:', round(precision, 3))
    test_acc = neigh.score(x_test, y_test)  # 得到最后的准确率
    return test_acc


def mySvm(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1)
    clf = svm.SVC(kernel='linear', C=1)   # 指定SVM中使用的内核类型 C=10000, linear', 'poly', 'rbf', 'sigmoid',
    clf.fit(x_train, y_train)
    joblib.dump(clf, path)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print("f1_ score:", round(f1, 3), '  recall_score:', round(recall, 3), ' precision_score:', round(precision, 3))
    test_acc = clf.score(x_test, y_test)
    return test_acc


def myMLP(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1)
    clf = MLPClassifier(learning_rate_init=0.001).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print("f1_ score:", round(f1, 3), '  recall_score:', round(recall, 3), ' precision_score:', round(precision, 3))
    test_acc = clf.score(x_test, y_test)
    return test_acc


def myDecisionTree(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1)
    clf = DecisionTreeClassifier(max_depth=14)
    clf.fit(x_train, y_train)
    importamce = clf.feature_importances_ / np.max(clf.feature_importances_)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))
    print("f1_ score:", round(f1, 3), '  recall_score:', round(recall, 3), ' precision_score:', round(precision, 3))
    test_acc = clf.score(x_test, y_test)
    return test_acc, importamce


def myMultinomialNB(features, labels):
    # Negative values in data can not pass to MultinomialNB (input X)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1)
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print("f1_ score:", round(f1, 3), '  recall_score:', round(recall, 3), ' precision_score:', round(precision, 3))
    acc = clf.score(x_test, y_test)
    return acc


def myRandomForest(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1)
    clf = RandomForestClassifier(max_depth=12, random_state=0)
    # clf = ExtraTreesClassifier(max_depth=4, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print("f1_ score:", round(f1, 3), '  recall_score:', round(recall, 3), ' precision_score:', round(precision, 3))
    acc = clf.score(x_test, y_test)
    return acc


def myFisherClassifier(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1)
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print("f1_ score:", round(f1, 3), '  recall_score:', round(recall, 3), ' precision_score:', round(precision, 3))
    acc = clf.score(x_test, y_test)
    return acc


def classify(features, labels, model='SVM'):
    """

    :param features: the number of features is 22
    :param labels:  range from -1 and 1
    :param model: dtype = str, Use this parameter to choose the classifier.
                  It can be SVM, DecisionTree, MLP, RandomForest, FisherDecision, KNN, FisherDecision.
    :return: the result of the classification, it includes the accuracy, F1-scores, precision_scores.
    """
    if isinstance(model, str):
        pass
    else:
        raise TypeError("the name of the classifier must be a str")

    if model == 'SVM':
        # 支持向量机
        acc = mySvm(features, labels)
        print("The accuracy of SVM is :", round(acc, 4))

    elif model == 'MLP':
        # 多层感知机
        acc = myMLP(features, labels)
        print("The accuracy of mlp is :", round(acc, 4))

    elif model == 'DecisionTree':
        # 决策树分类器
        acc, imp = myDecisionTree(features, labels)
        x = np.array([i for i in range(21)])
        plt.bar(x, imp)
        print("The accuracy of DecisionTree is :", round(acc, 4))
        plt.show()


    elif model == 'NativeBayes':
        # 朴素贝叶斯分类器
        acc = myMultinomialNB(features, labels)
        print("The accuracy of NativeBayes is :", round(acc, 4))

    elif model == 'RandomForest':
        # 随机森林分类器
        acc = myRandomForest(features, labels)
        print("The accuracy of RandomForest is :", round(acc, 4))

    elif model == 'FisherDecision':
        # Fisher判别器
        acc = myFisherClassifier(features, labels)
        print("The accuracy of FisherClassifier is :",round(acc, 4))

    elif model == 'KNN':
        # K最近邻分类器
        acc = myKNN(features, labels)
        print("The accuracy of KNN is :", round(acc, 4))

    else:
        raise NameError(model, 'is not in the classifier list')