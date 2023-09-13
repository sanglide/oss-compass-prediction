from utils.read import multiRead
import configparser
import sys
from MLmodel.model_dict import MLmodel_dict
from sklearn.model_selection import StratifiedKFold
from utils.evaluation import test

config = configparser.ConfigParser()
config.read('config.ini')
result_path = config['path']['result_path']
filePaths = result_path + 'segment2/'
Label_path = result_path + 'label.csv'


if __name__ == '__main__':
    x_data, y_data = multiRead(Label_path=Label_path, filePaths=filePaths)
    print(x_data.shape)
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