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
        if label == 1.0:
            label = np.array([0, 1])
        else:
            label = np.array([1, 0])
        return torch.from_numpy(ts).to(torch.float32), torch.from_numpy(label)
