import pickle
import sys
import os
import time
import numpy as np
import pandas
from sklearn.cluster import DBSCAN


sys.path.append("..")
from util_pakage.IOUtil import CSVDataLoader

cwd = os.path.abspath(os.path.join(os.getcwd(), ".."))
path1 =cwd+"\\data\\filter_lng.csv"
data_loader = CSVDataLoader(path1)
print("start")
data = data_loader.load_csv()
pandas_data = pandas.DataFrame(data)


#使用经度聚类：
##获取最大经度和最小经度
long_positive_max = pandas_data.iloc[:,0].max()
long_negative_min = pandas_data.iloc[:,0].min()

#DBScan参数规定
eps = 1
min_samples = 5


accurate_ratio = 0.01
#聚类结果
labels = []

#是否全是-1
def is_all_minus_one(arr):
    for num in arr:
        if num != -1:
            return False
    return True



counter = 0
#对经度为正的作聚类：

for i in np.arange(0,long_positive_max, accurate_ratio):
    data = pandas_data.loc[(pandas_data[0]<i+accurate_ratio)&(pandas_data[0]>=i)]
    if not data.empty:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data)
        tmp = dbscan.labels_
        if i != 0.0:
            for k in range(tmp.size):
                if(tmp[k] != -1):
                    tmp[k] = tmp[k] + 1 + counter
        labels.append(tmp)
        if(not is_all_minus_one(dbscan.labels_)):
            counter = np.max(tmp)
        print("counter:{}".format(counter))
        print("第{}次聚类：".format(i))
        for k, label in enumerate(tmp):
            print(f"数据点 {k}: 类别 {label}")
#对经度为负的作聚类

counter = 0
for i in np.arange(0, long_negative_min, accurate_ratio):
    data = pandas_data.loc[(pandas_data[0]>i-accurate_ratio)&(pandas_data[0]<=i)]
    if not data.empty:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data)
        if i != 0.0:
            for k in range(data.size):
                if(data[k] != -1):
                    data[k] = data[k] + 1 + counter
        labels.append(data)
        if(not is_all_minus_one(dbscan.labels_)):
            counter =np.max(data)
        labels.append(dbscan.labels_)
        for k, label in enumerate(data):
            print(f"数据点 {k}: 类别 {label}")



