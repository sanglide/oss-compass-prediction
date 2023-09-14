import yaml
from sklearn.model_selection import StratifiedKFold
from MLmodel.model_dict import MLmodel_dict
from utils.evaluation import test
from utils.read import read_dict

with open('config.yaml', 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

read_func = config['read']
model = config['model']['name']
split_count = config['model']['split_count']


def main():
    x_data, y_data = read_dict[read_func]()
    kf = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=42)
    if model == "all":
       for name, _ in MLmodel_dict.items():
            test(name, x_data, y_data, kf) 
    elif MLmodel_dict.get(model) is None:
        print("you need provide a right model name")
    else:
        test(model, x_data, y_data, kf)    

if __name__ == '__main__':
    main()