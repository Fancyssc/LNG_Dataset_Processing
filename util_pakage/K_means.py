import time
import os
import sys
import pickle
from matplotlib import pyplot as plt

sys.path.append(".")
from sklearn.cluster import KMeans

from util_pakage.Pre_process import preprocess, convert_to_labeledcsv
from util_pakage.IOUtil import CSVDataLoader


def k_means():
    #convert_to_labeledcsv()
    #preprocess()
    cwd = os.getcwd()
    path1 = cwd + "/data/filter_lng.csv"
    data_loader = CSVDataLoader(path1)
    print("start")
    data = data_loader.load_csv()

    '''
    start = time.time()
    print("start")
    clustering_kmeans = KMeans(n_clusters=400, init='random',verbose=1,max_iter=30).fit(data)
    end = time.time()
    '''

    '''
    # 将模型保存到文件
    with open('kmeans_model_400.pkl', 'wb') as f:
        pickle.dump(clustering_kmeans, f)
    print("Kmeans_time:",end-start)
    '''


    # 加载保存的模型
    with open('kmeans_model.pkl', 'rb') as f:
        clustering_kmeans = pickle.load(f)

    print("总数: ", len(clustering_kmeans.labels_))
    print("聚类中心点数: ", len(clustering_kmeans.cluster_centers_))
    for center in clustering_kmeans.cluster_centers_:
         print("中心坐标：",center ,"\n")
    inertia = clustering_kmeans.inertia_

    centers = clustering_kmeans.cluster_centers_
    plt.scatter(centers[:,0:1],centers[:,1:2],s=2)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.savefig(cwd+"/figure_package/K-means.jpg")
    plt.show()

    print("方差",inertia)