import pandas as pd
import numpy as np
import sys
import math
import os

sys.path.append(".")
def convert_to_labeledcsv():
    cwd = os.getcwd()
    path1 = cwd + "/data/lng2.csv"
    path2 = cwd + "/data/temp.csv"

    content = open(path1)
    with open(path2, "w") as f:
        for line in content:
            f.write(line.replace(" ", ","))
    f.close()
    with open(path2, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write('mmsi,timestamp,status,velocity,longitude,latitude,draft\n' +
                content)
    f.close()


def distance(target, destination):
    pi = 3.14
    dis = 6371000 * math.acos(
        math.cos(target[1] * pi / 180) * math.cos(destination[1] * pi / 180) *
        math.cos((target[0] - destination[0]) * pi / 180) +
        math.sin(target[1] * pi / 180) * math.sin(destination[1] * pi / 180) -
        1e-12)
    return dis


def preprocess():
    print("start process")
    # 读取原始CSV文件
    cwd = os.path.join(os.getcwd(), "..")
    path1 = cwd + "/data/temp.csv"
    path2 = cwd + "/data/filter_lng_highVelo.csv"
    df = pd.read_csv(path1)

    # 筛选速度大于1节并且吃水大于0的数据
    filtered_df = df[(df['velocity'] >= 3) & (df['draft'] != 0) & (df['draft'] < 300)]

    # 将筛选后的数据保存到新的CSV文件
    filtered_df.to_csv(path2, index=False)
    print("finish process")


def preprocess_DBscan():
    # 读取原始CSV文件
    cwd = os.path.join(os.getcwd(), "..")
    path = cwd + "/data/filter_lng_highVelo.csv"
    path0 = cwd + "/data/filter_lng_lowVelo.csv"
    path1 = cwd + "/data/filter_lng.csv"
    path2 = cwd + "/data/filter_lng_DBscan.csv"
    df = pd.read_csv(path)
    df1 = pd.read_csv(path1)
    final_df = []
    tar_long_lati = df1[['longitude', 'latitude']]
    des_long_lati = df[['longitude', 'latitude']]
    df = df.to_numpy()
    df1 = df1.to_numpy()
    des_long_lati = des_long_lati.to_numpy()
    tar_long_lati = tar_long_lati.to_numpy()
    tar = [[]]

    # 筛选速度小于1节并且吃水大于0的数据
    num_highVelo = 0
    print("start DBscan process")
    for i in range(len(df1)):
        # for i in range(20):
        num_highVelo = 0
        target = tar_long_lati[i]
        print(i)
        for j in range(len(df)):
            # for j in range(20):
            destination = des_long_lati[j]
            if (distance(target, destination) < 5000):
                num_highVelo = num_highVelo + 1
                if (num_highVelo > 5):
                    break
        if (num_highVelo <= 5):
            final_df.append(df1[i])

        # targrt.append(num_highVelo)
        # tar.append(target)
    print("finish scan")
    final_df = pd.DataFrame(final_df)
    low_df = pd.DataFrame(tar_long_lati)
    # low_df = pd.DataFrame(tar)

    # 将筛选后的数据保存到新的CSV文件
    final_df.to_csv(path2, index=False)
    # tar.to_csv(path0, index=False)
    low_df.to_csv(path0, index=False)
    print("finish DBscan process")

preprocess()
preprocess_DBscan()