import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#用于划分数据集，即将原始数据集划分成测试集和训练集两部分的函数。
from collections import Counter
#对字符串\列表\元祖\字典进行计数,返回一个字典类型的数据,键是元素,值是元素出现的次数
import math
#创建数据集
def create_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label']=iris.target
    df.columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data=np.array(df.iloc[:100,:])#截取0~99行
    #print(data)
    return data[:, :-1], data[:, -1]
# 返回截取所有行，不包括最后一列的所有列
# 返回截取最后一列的所有行
X,y=create_data()
#print(X,y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
#测试集占0.3
class NaiveBayes:
    def __init__(self):
        self.model = None
        self.prior={}
    # 数学期望
    @staticmethod#静态函数
    def mean(X):
        return sum(X) / float(len(X))
    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))
    # 高斯概率密度函数(可以求取类条件概率)
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
    # 求每个类别下每个特征向量的的均值和方差
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]#*train_data为可变参数，这里zip()函数把data字典里的值每种多个特征向量组成一个元组
        return summaries
    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))#set()函数可以达到去重效果，里面不能包含重复的元素，接收一个list作为参数
        data = {label: [] for label in labels}
        print(data)#{0.0: [], 1.0: []}
        #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        for f, label in zip(X, y):
            data[label].append(f)
        print(data)#{0.0: [array([5.1, 3.5, 1.4, 0.2]), array([4.4, 3. , 1.3, 0.2])...),1.0:[array([5.7, 3. , 4.2, 1.2]), ...)]
        self.model = {
            label: self.summarize(value)#这里把data字典中的value给了summarize()函数
            for label, value in data.items()
        }#这里实现了求出每个类别下各个特征值的均值和方差，有label
        return 'gaussianNB train done!'
    #求每个类别的先验概率
    def _get_prior(self, y):
        cnt = Counter(y)#对元组进行计数，返回字典类型的数据，键是元素，值是元素出现的次数
        for label, count in cnt.items():
            self.prior[label] = count / len(y)
    # 利用贝叶斯公式计算后验概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            prob = self.prior[label]
            for i in range(len(value)):
                mean, stdev = value[i]
                prob *= self.gaussian_probability(
                    input_data[i], mean, stdev)
            probabilities[label]=prob
        return probabilities
    #比较后验概率的大小，预测数据的类别
    def predict(self, x_test):
        label = sorted(
            self.calculate_probabilities(x_test).items(),
            key=lambda x: x[-1])[-1][0]
        #按probabilities字典中最后一列从小到打排序，并把最后一行第一列的标签输出（0/1）
        return label
    #对测试集求取该模型的准确率
    def score(self, x_test, y_test):
        right = 0
        for X, y in zip(x_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        return right / float(len(x_test))
model = NaiveBayes()
model.fit(x_train, y_train)
model._get_prior(y_train)
print('样本[4.4, 3.2, 1.3, 0.2]的预测结果: %d' % model.predict([4.4, 3.2, 1.3, 0.2]))
print('测试集的准确率: %f' % model.score(x_test, y_test))

#这个方法更简单
# class GaussianNaiveBayes:
#     def __init__(self):
#         self.parameters = {}
#         self.prior = {}
#
#     # 训练过程就是求解先验概率和高斯分布参数的过程
#     # X:(样本数，特征维度） Y:(样本数，)
#     def fit(self,X, Y):
#         parameters={}
#         labels = set(Y)
#         for label in labels:
#             samples = X[Y == label]#先把y=0对应x的数据给sambles，然后接着再把y=1的数据给sambles就可以简单地实现分类了，之后计算均值和方差很简单
#             print(samples)
#             # 计算高斯分布的参数：均值和标准差
#             means = np.mean(samples, axis=0)
#             stds = np.std(samples, axis=0)
#             parameters[label] = {
#                 'means': means,
#                 'stds': stds
#             }
#         print(parameters)
#     # 计算每个类别的先验概率
#     def _get_prior(self, Y):
#         cnt = Counter(Y)
#         for label, count in cnt.items():
#             self.prior[label] = count / len(Y)
#
#     # 高斯分布
#     def _gaussian(self, x, mean, std):
#         exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
#         return (1 / (math.sqrt(2 * math.pi) * std)) * exponent
#
#     # 计算样本x属于每个类别的似然概率
#     def _cal_likelihoods(self, x):
#         likelihoods = {}
#         for label, params in self.parameters.items():
#             means = params['means']
#             stds = params['stds']
#             prob = self.prior[label]
#             # 计算每个特征的条件概率,P(xi|yk)
#             for i in range(len(means)):
#                 prob *= self._gaussian(x[i], means[i], stds[i])
#             likelihoods[label] = prob
#         return likelihoods
#     # x:单个样本
#     def predict(self, x):
#         probs = sorted(self._cal_likelihoods(x).items(), key=lambda x: x[-1])  # 按概率从小到大排序
#         return probs[-1][0]
#
#     # 计算模型在测试集的准确率
#     # X_test:（测试集样本个数，特征维度）
#     def evaluate(self, X_test, Y_test):
#         true_pred = 0
#         for i, x in enumerate(X_test):
#             label = self.predict(x)
#             if label == Y_test[i]:
#                 true_pred += 1
#         return true_pred / len(X_test)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train, y_train)
print(clf.predict([[4.4,  3.2,  1.3,  0.2]]))
print(clf.score(x_test, y_test))
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
mu=MultinomialNB()
mu.fit(x_train, y_train)
print(mu.predict([[4.4,  3.2,  1.3,  0.2]]))
print(mu.score(x_test, y_test))

