import json
import time
import os
import sys
import pickle
import random
import numpy as np

sys.path.append(".")


from util_pakage.Pre_process import preprocess, convert_to_labeledcsv
from util_pakage.IOUtil import CSVDataLoader

def judge(used_file,cluster_nums,output_file):
    in_ = 0
    out_ = 0
    not_ = 0

    # 读入数据
    start = time.time()

    cwd = os.getcwd()
    path1 = cwd + "/data/filter_lng.csv"
    data_loader = CSVDataLoader(path1)
    print("start")
    data = data_loader.load_bigger_csv()

    # 加载保存的模型
    with open(used_file, 'rb') as f:
        clustering_result = pickle.load(f)

    #得到标签集合
    labels = clustering_result.labels_

    # 创建一个字典，用于将类别标签映射到数据点列表
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(data[i])

    print("cluster计算完成")

    initial_value = 'not'  # 设置初始值为 'not'
    category = [initial_value] * cluster_nums
    # 在每个类别中随机选择10个数据点
    for i in range(cluster_nums):
        selected_points = []
        # 检查类别中的数据点数量是否足够10个
        if i in clusters and clusters[i]:
            if len(clusters[i]) >= 10:
                selected_points = random.sample(clusters[i], k=10)
            else:
                selected_points.extend(clusters[i])  # 如果类别中的数据点数量小于10个，则将所有点都选入
        print(selected_points)
        print("开始计算吃水: ",i)
        #开始进行吃水判断
        in_count = 0
        out_count = 0
        not_count = 0
        for point in selected_points:
            currentMmsi = point[0]
            currentDraft = point[4]
            currentTime = point[1]
            for p in selected_points:
                newtime = p[1]
                if p[0] == currentMmsi and newtime != currentTime:
                    newDraft = p[4]
                    diff = newDraft - currentDraft

                    #泊点
                    if diff == 0:
                        not_count += 1

                    #出站口
                    elif (newtime - currentTime)*diff > 0:
                        out_count += 1

                    #进站口
                    else:
                        in_count += 1
                    break

        counts = {'in': in_count, 'out': out_count, 'not': not_count}

        max_count = max(counts, key=counts.get)

        category[i] = max_count

        if(max_count == 'in'):
            in_ += 1
        elif(max_count == 'out'):
            out_ += 1
        else:
            not_ += 1

    print("出站口共有：",out_)
    print("入站口共有：", in_)
    print("泊点共有：", not_)

    data_list = []
    print("开始写入json")
    # 遍历 category 列表，根据元素的值构建对应的字典数据
    cluster_centers = clustering_result.cluster_centers_
    for i, category_value in enumerate(category):
        if category_value == 'not':
            data = {
                'code': i + 1,
                'latitude': '{:.8f}'.format(float(cluster_centers[i][0])),
                'longitude':'{:.8f}'.format(float( cluster_centers[i][1])),
                'isLNG': False,
                'IN': False
            }
        elif category_value == 'in':
            data = {
                'code': i + 1,
                'latitude': '{:.8f}'.format(float(cluster_centers[i][0])),
                'longitude': '{:.8f}'.format(float(cluster_centers[i][1])),
                'isLNG': True,
                'IN': True
            }
        elif category_value == 'out':
            data = {
                'code': i + 1,
                'latitude': '{:.8f}'.format(float(cluster_centers[i][0])),
                'longitude': '{:.8f}'.format(float(cluster_centers[i][1])),
                'isLNG': True,
                'IN': False
            }
        # 将数据添加到数据列表中
        data_list.append(data)
    # 将数据列表写入 JSON 文件
    end = time.time()
    print("判断耗时：",end - start)
    cwd = os.path.join(os.getcwd())
    with open(cwd+'/output_json/'+output_file, 'w') as json_file:

        json.dump(data_list, json_file, indent=4)

    return category









