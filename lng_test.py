import time
import os

from sklearn.cluster import KMeans

from venv.Include.util_package.Pre_process import preprocess, convert_to_labeledcsv
from venv.Include.util_package.IOUtil import CSVDataLoader

if __name__ == '__main__':
    # convert_to_labeledcsv()
    # preprocess()
    cwd = os.getcwd()
    path1 = cwd + "/data/filter_lng.csv"
    data_loader = CSVDataLoader(path1)
    data = data_loader.load_csv()
    start = time.time()
    clustering_kmeans = KMeans(n_clusters=400, init='random',max_iter=40).fit(data)
    end = time.time()
    print("Kmeans_time:"+end-start)
    for center in clustering_kmeans.cluster_centers_:
         print("中心坐标："+center +"\n")

