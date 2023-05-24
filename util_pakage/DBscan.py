import sys
import os
import time

import numpy as np
from sklearn.cluster import DBSCAN

sys.path.append(".")
from util_pakage.IOUtil import CSVDataLoader

def dbscan():
    cwd = os.getcwd()
    path1 = cwd + "/data/filter_lng.csv"
    data_loader = CSVDataLoader(path1)
    print("start")
    data = data_loader.load_csv()

    # 设置DBSCAN参数
    epsilon = 0.1  # 两个样本之间被视为邻居的最大距离
    min_samples = 5  # 形成核心点所需的最小邻居样本数

    # 创建DBSCAN对象并拟合数据
    print("start dbscan")
    start = time.time()
    dbscan_ = DBSCAN(eps=epsilon, min_samples=min_samples,n_jobs=-1)
    clusters = dbscan_.fit_predict(data)
    end = time.time()
    print("dbscan time: ",end - start)

    # 打印聚类结果
    print("聚类结果:", set(clusters))

    # 您还可以打印每个聚类中的点数
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    for cluster, count in zip(unique_clusters, counts):
        print("聚类", cluster, "包含", count, "个点。")
