# _*_ coding : utf-8 _*_
# @Time : 2023/9/5 23:36
# @Author : Confetti-Lxy
# @File : run
# @Project : project

import configparser
import csv
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from models import model_dict
from utils import evaluation

# 读取ini配置文件
config = configparser.ConfigParser()
config.read('config.ini')
result_path = config['path']['result_path']
filePaths = result_path + 'segment2\\'
Label_path = result_path + 'label.csv'

# 读取命令行参数
args = sys.argv
num_args = len(args)
model = None
if num_args > 1:
    model = args[1]


# 读取数据
def Read():
    LabelDict = {}
    with open(Label_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            LabelDict[row[1].replace("/", "_") + '.csv'] = row[2]
    del LabelDict['repo.csv']
    count = 0
    X, Y = [], []
    # 读取csv文件
    length = 0
    for filename in os.listdir(filePaths):
        if filename.endswith('.csv'):
            if LabelDict[filename] != "-1":
                file_path = os.path.join(filePaths, filename)
                df = pd.read_csv(file_path)
                length = max(length, len(df.iloc[:, -1]))
    for filename in os.listdir(filePaths):
        if filename.endswith('.csv'):
            if LabelDict[filename] != "-1":
                file_path = os.path.join(filePaths, filename)
                count += 1
                df = pd.read_csv(file_path)
                loc_labels = ["commit_frequency_activity", "activity_score_activity", "commit_frequency_codequality"]
                #loc_labels = ["commit_frequency_codequality"]
                value = df.loc[:, loc_labels].values
                ts = np.concatenate((np.zeros((1, len(loc_labels))), value), axis=0)
                if len(ts) < length + 1:
                    ts = np.concatenate((ts, np.zeros((length + 1 - len(ts), len(loc_labels)))), axis=0)
                ts[0][0] = len(value)
                X.append(np.array(ts))
                Y.append(int(LabelDict[filename]))
    return X, Y


if __name__ == '__main__':
    x_data, y_data = Read()
    x_data, y_data = np.array(x_data), np.array(y_data)
    # X_train, X_test, y_train, y_test = train_test_split(np.array(x_data), np.array(y_data), test_size=0.2)
    n_splits = 10
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # 分层抽样交叉验证循环
    if model is None and model != "all":
        print("you need provide a model name")
    elif model == "all":
        for name, m in model_dict.items():
            print("==================================" + name + "==================================")
            for train_index, test_index in kf.split(x_data, y_data):
                X_train, X_test = x_data[train_index], x_data[test_index]
                y_train, y_test = y_data[train_index], y_data[test_index]
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                evaluation.evaluate(y_test, y_pred, name)
                print("==================================")
    else:
        for train_index, test_index in kf.split(x_data, y_data):
            X_train, X_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            # 否则只尝试指定的模型
            print(f"the model is {model}")
            m = model_dict[model]
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            evaluation.evaluate(y_test, y_pred, model)
            print("==================================")
