import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class OssDataset(Dataset):
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        df = pd.read_csv(csv_dir, header=None)
        self.datas = df.to_numpy()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        ts = self.datas[index][1:].reshape(1, -1)
        label = self.datas[index][0]
        # label = np.array(label)
        if label == 1.0:
            label = np.array([0, 1])
        else:
            label = np.array([1, 0])
        return torch.from_numpy(ts).to(torch.float32), torch.from_numpy(label)


# class FixedOssDataset(Dataset):
#     def __init__(self, result_path):
#         featurePath = result_path + 'features/features.csv'
#         df = pd.read_csv(featurePath)
#         df = df.fillna(0)
#         X, Y = df.iloc[:, 1: -1], df.iloc[:, -1]
#         X[X.columns] = X[X.columns].astype(float)
#         with open("../data/selectedData/select_features_596.txt", 'r') as file:
#             lines = file.readlines()
#             lines = [line.strip() for line in lines]
#             features_filtered = X[lines]
#         self.x = features_filtered.values
#         self.y = Y.values

#     def __len__(self):
#         assert len(self.x) == len(self.y)
#         return len(self.x)
    
#     def __getitem__(self, index):
#         ts = self.x[index]
#         label = self.y[index]
#         if label == 1.0:
#             label = np.array([0, 1])
#         else:
#             label = np.array([1, 0])
#         return torch.from_numpy(ts).to(torch.float32), torch.from_numpy(label)

