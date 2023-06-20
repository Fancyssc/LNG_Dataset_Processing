import sys
import os
import pickle
import folium
import webbrowser
from matplotlib import pyplot as plt
from folium.plugins import MarkerCluster
sys.path.append(".")

m = folium.Map()
cwd = os.getcwd()

with open('mini_batch_kmeans_model_1600.pkl', 'rb') as f:
    clustering_kmeans = pickle.load(f)

print(clustering_kmeans.cluster_centers_)
#只保留经纬度
centers = clustering_kmeans.cluster_centers_[:,:2]
plt.scatter(centers[:,0:1],centers[:,1:2],s=2)
plt.show()

marker_cluster = MarkerCluster().add_to(m)

for lat, lng  in centers:
 folium.Marker(
  location=[lng, lat],
  icon=None,
  popup= 1 ,
 ).add_to(marker_cluster)
# add marker_cluster to map
m.add_child(marker_cluster)

m.save(cwd+"/figure_package/station.html")

webbrowser.open(cwd+"/figure_package/station.html")