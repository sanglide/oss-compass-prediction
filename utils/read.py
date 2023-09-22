import os
import csv
import pandas as pd
import numpy as np
import configparser
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from .wrapper import deprecated

# 读取ini配置文件
config = configparser.ConfigParser()
config.read('config.ini')
result_path = config['path']['result_path']


def commonRead():
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


@deprecated
def featureRead():
    featurePath = result_path + 'features/features.csv'
    df = pd.read_csv(featurePath)
    X, Y = df.iloc[:, 1:-1], df.iloc[:, -1]
    X[X.columns] = X[X.columns].astype(float)
    impute(X)   # 去除NaN数据
    features_filtered = select_features(X, Y, fdr_level=1e-6)   # 挑选特征
    # features_filtered.columns为挑选的特征的名字
    selected = features_filtered.columns
    selected =  selected.to_numpy()
    # if not os.path.exists("select_features.txt"):
    np.savetxt("select_features.txt", selected, delimiter=',', fmt='%s')
    return features_filtered.values, Y.values


def FixedFeatureRead():
    featurePath = result_path + 'features/features.csv'
    df = pd.read_csv(featurePath)
    df = df.fillna(0)
    X, Y = df.iloc[:, 1: -1], df.iloc[:, -1]
    X[X.columns] = X[X.columns].astype(float)
    with open("select_features.txt", 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        features_filtered = X[lines]
        return features_filtered.values, Y.values


read_dict = {
    "common-read": commonRead,
    "feature-read": featureRead,
    "fixed-feature-read": FixedFeatureRead,
}
