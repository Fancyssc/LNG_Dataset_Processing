import time
import os
import sys
import pickle

sys.path.append(".")
from sklearn.cluster import KMeans

from util_pakage.Pre_process import preprocess, convert_to_labeledcsv
from util_pakage.IOUtil import CSVDataLoader

#convert_to_labeledcsv()
#preprocess()
cwd = os.getcwd()
path1 = cwd + "/data/filter_lng.csv"
data_loader = CSVDataLoader(path1)
print("start")
data = data_loader.load_csv()

start = time.time()
print("start")
clustering_kmeans = KMeans(n_clusters=400, init='random',max_iter=40).fit(data)
end = time.time()

# 将模型保存到文件
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(clustering_kmeans, f)

print("Kmeans_time:",end-start)
print("总数: ", len(clustering_kmeans.labels_))
print("聚类中心点数: ", len(clustering_kmeans.cluster_centers_))
for center in clustering_kmeans.cluster_centers_:
     print("中心坐标：",center ,"\n")



