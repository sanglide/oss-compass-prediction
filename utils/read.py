import os
import csv
import pandas as pd
import numpy as np
import configparser
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

# 读取ini配置文件
config = configparser.ConfigParser()
config.read('config.ini','utf-8')
result_path = config['path']['result_path']


def Read():
    X, Y = [], []
    df = pd.read_csv('data/old/data.csv', header=None)
    datas = df.to_numpy()
    for data in datas:
        X.append(data[1:])
        Y.append(data[0])
    X, Y = np.array(X), np.array(Y)
    return X, Y


def multiRead():
    filePaths = result_path + 'segment2/'
    Label_path = result_path + 'label.csv'
    LabelDict = {}
    with open(Label_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            LabelDict[row[1].replace("/", "_") + '.csv'] = row[2]
    del LabelDict['repo.csv']
    X, Y = [], []
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
                df = pd.read_csv(file_path)
                # loc_labels = ["commit_frequency_activity", "activity_score_activity", "commit_frequency_codequality"]
                loc_labels = ["commit_frequency_activity"]
                value = df.loc[:, loc_labels].values
                ts = np.concatenate((np.zeros((1, len(loc_labels))), value), axis=0)
                if len(ts) < length + 1:
                    ts = np.concatenate((ts, np.zeros((length + 1 - len(ts), len(loc_labels)))), axis=0)
                ts[0][0] = len(value)
                X.append(np.array(ts))
                Y.append(int(LabelDict[filename]))
    X, Y = np.array(X), np.array(Y)
    return X, Y


def featureRead():
    featurePath = result_path + 'features\\features.csv'
    print(featurePath)
    df = pd.read_csv(featurePath)
    X, Y = df.iloc[:, 1:-1], df.iloc[:, -1]
    X[X.columns] = X[X.columns].astype(float)
    impute(X)   # 去除NaN数据
    features_filtered = select_features(X, Y)   # selected_features
    # features_filtered.columns are the names of selected_features, it can be used for prediction
    selected = features_filtered.columns
    selected = selected.to_numpy()
    np.savetxt("selected_features.txt", selected, delimiter=',', fmt='%s')
    # selected_features is needed for future prediction,so we save it to txt file
    return features_filtered.values, Y.values



read_dict = {
    "read": Read,
    "multi-read": multiRead,
    "feature-read": featureRead
}
