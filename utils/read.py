import os
import csv
import pandas as pd
import numpy as np

result_path = '/home/confetti/oss/oss-compass-result/'
filePaths = result_path + 'segment2/'
Label_path = result_path + 'label.csv'

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


read_dict = {
    "read": Read,
    "multi-read": multiRead,
}