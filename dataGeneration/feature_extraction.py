import os
import csv
import pandas as pd
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
import configparser

# 读取ini配置文件
config = configparser.ConfigParser()
config.read('../config.ini','utf-8')
result_path = config['path']['result_path']
filePaths = result_path + 'segment2/'
featurePaths = result_path + 'features/'
Label_path = result_path + 'label.csv'


labels = ['name', 'grimoire_creation_date', 'commit_frequency_activity', 'activity_score_activity']


def get_data_feature(Label_path, filePaths):
    LabelDict = {}
    with open(Label_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            LabelDict[row[1].replace("/", "_") + '.csv'] = row[2]
    del LabelDict['repo.csv']
    df_new = pd.DataFrame()
    for filename in os.listdir(filePaths):
        if filename.endswith('.csv'):
            if LabelDict[filename] != "-1":
                file_path = os.path.join(filePaths, filename)
                df = pd.read_csv(file_path)
                df.rename(columns={'Unnamed: 0': 'name'}, inplace=True)
                df['name'] = filename
                df = df.loc[:, labels]
                X = extract_features(df, column_id='name', column_sort='grimoire_creation_date',
                                     default_fc_parameters=EfficientFCParameters())
                X = X.reset_index()
                X = X.iloc[:, 1:]
                X['label'] = LabelDict[filename]
                df_new = pd.concat([df_new, X])
    df_new.to_csv(featurePaths + "features.csv")


if __name__ == '__main__':
    get_data_feature(Label_path=Label_path, filePaths=filePaths)
