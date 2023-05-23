import pandas as pd
import os


def convert_to_labeledcsv():
    cwd = os.getcwd()
    path1 = cwd + "/DataSet/lng2.csv"
    path2 = cwd + "/DataSet/temp.csv"

    content = open(path1)
    with open(path2,"w") as f:
        for line in content:
            f.write(line.replace(" ",","))
    f.close()
    with open(path2,"r+") as f:
        content = f.read()
        f.seek(0,0)
        f.write('mmsi,timestamp,status,velocity,longitude,latitude,draft\n' + content)
    f.close()


def preprocess():
    # 读取原始CSV文件
    cwd = os.getcwd()
    path1 = cwd + "/DataSet/temp.csv"
    path2 = cwd + "/DataSet/filter_lng.csv"
    df = pd.read_csv(path1)

    # 筛选速度小于1节并且吃水大于0的数据
    filtered_df = df[(df['velocity'] <= 1) & (df['draft'] != 0) &(df['draft'] < 300)]

    # 将筛选后的数据保存到新的CSV文件
    filtered_df.to_csv(path2, index=False)
