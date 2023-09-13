import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from MLmodel.model_dict import MLmodel_dict
from utils.evaluation import test
from utils.read import Read


if __name__ == '__main__':
    x_data, y_data = Read()
    # TODO

    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    args = sys.argv
    if len(args) == 1:
        print("you need provide a model name")
    elif args[1] == "all":
        for name, _ in MLmodel_dict.items():
            print("==================================" + name + "==================================")
            test(name, x_data, y_data, kf)
    else:
        test(args[1], x_data, y_data, kf)