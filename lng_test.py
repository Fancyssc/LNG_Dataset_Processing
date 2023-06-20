import time
import os
import sys
import pickle
from matplotlib import pyplot as plt

from util_pakage.Judge import judge

sys.path.append(".")
from util_pakage.K_means import k_means
from util_pakage.DBscan import dbscan
from util_pakage.mini_batch_kmeans import mini_batch_kmeans

#k_means()
#dbscan()
#mini_batch_kmeans()
#judge('kmeans_model.pkl',800,'k_means_lng_result_list.json')
judge('mini_batch_kmeans_model_400.pkl',400,'mini_batch_k_means_lng_result_list.json')