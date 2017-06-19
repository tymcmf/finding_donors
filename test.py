#!/usr/bin/python
# _*_ coding: utf-8 _*_

# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # 允许为DataFrame使用display()

# 导入附加的可视化代码visuals.py
# import visuals as vs


# 导入人口普查数据
data = pd.read_csv("census.csv")

# 成功 - 显示第一条记录
display(data.head(n=1))

# data.education_level.dtype



# TODO：总的记录数
n_records = data.size

# TODO：被调查者的收入大于$50,000的人数
n_greater_50k = data[data.income == ">50K"].size

# TODO：被调查者的收入最多为$50,000的人数
n_at_most_50k = data[data.income == "<=50K"].size

# TODO：被调查者收入大于$50,000所占的比例
greater_percent = (float(n_greater_50k) / (n_greater_50k+n_at_most_50k))*100

# 打印结果
# print "Total number of records: {}".format(n_records)
# print "Individuals making more than $50,000: {}".format(n_greater_50k)
# print "Individuals making at most $50,000: {}".format(n_at_most_50k)
# print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 可视化经过log之后的数据分布
# vs.distribution(features_raw, transformed = True)

# 导入sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)


# TODO：将'income_raw'编码成数字值
income = income_raw.copy()
income[income=="<=50K"] = 0
income[income==">50K"] = 1
income = income.astype(np.int)
print(income[:5])


# # 打印经过独热编码之后的特征数量
# encoded = list(features.columns)
# print "{} total features after one-hot encoding.".format(len(encoded))
#
# # 移除下面一行的注释以观察编码的特征名字
# # print encoded
#
# # 导入 train_test_split
# from sklearn.model_selection import train_test_split
#
# # 将'features'和'income'数据切分成训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(features.values, income.values, test_size = 0.2, random_state = 0)
# # print X_train
#
#
# # 显示切分的结果
# print "Training set has {} samples.".format(X_train.shape[0])
# print "Testing set has {} samples.".format(X_test.shape[0])
#
#
#
# # TODO： 计算准确率
# accuracy = float(income[income==1].size) / income.size
#
# # TODO： 使用上面的公式，并设置beta=0.5计算F-score
# fscore = (1+0.5*0.5)*(accuracy)/(0.5*0.5+accuracy + 1)
#
# # 打印结果
# print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)
#
# # TODO：从sklearn中导入两个评价指标 - fbeta_score和accuracy_score
# from sklearn.metrics import fbeta_score, accuracy_score
#
#
# def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
#     '''
#     inputs:
#        - learner: the learning algorithm to be trained and predicted on
#        - sample_size: the size of samples (number) to be drawn from training set
#        - X_train: features training set
#        - y_train: income training set
#        - X_test: features testing set
#        - y_test: income testing set
#     '''
#
#     results = {}
#
#     # TODO：使用sample_size大小的训练数据来拟合学习器
#     # TODO: Fit the learner to the training data using slicing with 'sample_size'
#     start = time()  # 获得程序开始时间
#     X_train_s = X_train[:sample_size]
#     y_train_s = y_train[:sample_size]
#     learner = learner.fit(X_train_s, y_train_s)
#     end = time()  # 获得程序结束时间
#
#     # TODO：计算训练时间
#     results['train_time'] = end - start
#
#     # TODO: 得到在测试集上的预测值
#     #       然后得到对前300个训练数据的预测结果
#     start = time()  # 获得程序开始时间
#     predictions_test = learner.predict(X_test[:300])
#     predictions_train = learner.predict(X_train[:300])
#     end = time()  # 获得程序结束时间
#
#     # TODO：计算预测用时
#     results['pred_time'] = end - start
#
#     # TODO：计算在最前面的300个训练数据的准确率
#     results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
#
#     # TODO：计算在测试集上的准确率
#     results['acc_test'] = accuracy_score(y_test[:300], predictions_test)
#
#     # TODO：计算在最前面300个训练数据上的F-score
#     results['f_train'] = fbeta_score(y_train[:300], predictions_train)
#
#     # TODO：计算测试集上的F-score
#     results['f_test'] = fbeta_score(y_test[:300], predictions_test)
#
#     # 成功
#     print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
#
#     # 返回结果
#     return results
#
#
# # TODO：从sklearn中导入三个监督学习模型
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
#
# # TODO：初始化三个模型
# clf_A = GaussianNB()
# clf_B = DecisionTreeClassifier()
# clf_C = SVC()
#
# # TODO：计算1%， 10%， 100%的训练数据分别对应多少点
# samples_1 = int(X_train.shape[0]*0.01)
# samples_10 = int(X_train.shape[0]*0.1)
# samples_100 = X_train.shape[0]
#
# # 收集学习器的结果
# results = {}
# for clf in [clf_A, clf_B, clf_C]:
#     clf_name = clf.__class__.__name__
#     results[clf_name] = {}
#     for i, samples in enumerate([samples_1, samples_10, samples_100]):
#         results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)
