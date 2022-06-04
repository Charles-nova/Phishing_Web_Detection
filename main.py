from classifiers import classify
import pandas as pd
from roc import plt_roc
import time
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    data = pd.read_csv("./dataset/phishing.csv")
    # features = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 27, 29]]
    # features = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11]]
    # features = data.iloc[:, [12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 27, 29]]
    # features = data.iloc[:, [12, 13, 14, 15, 16, 17]]
    # features = data.iloc[:, [22, 23, 24, 25, 27, 29]]
    features = data.iloc[:, [12]]
    features = features.values
    labels = data.iloc[:, -1]
    start = time.time()
    classify(features, labels, model="DecisionTree")
    end = time.time()
    print(end - start)


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
