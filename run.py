import yaml
import os
from sklearn.model_selection import StratifiedKFold
from MLmodel.model_dict import MLmodel_dict
from utils.evaluation import test, mix_test
from utils.read import read_dict
from utils.wrapper import timeit
from generate_html.generate_html import generate_html
import warnings
import shutil

warnings.filterwarnings("ignore")

with open('config.yaml', 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

model_name = config['model']['name']
split_count = config['model']['split_count']
is_html = config['visualization']['html']
is_mix = config['model']['isMix']
model_list = config['model']['model_list']


@timeit
def main():
    kf = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=42)
    if is_html:
        if os.path.exists('data/html'):
            shutil.rmtree('data/html')
        if os.path.exists('generate_html/outputs'):
            shutil.rmtree('generate_html/outputs')
    if is_mix:
        x_data, y_data = read_dict[MLmodel_dict[model_list[0]].get_read_func()]()
        print(model_list)
        mix_test(model_list, x_data, y_data, kf)
    elif model_name == "all":
        for name, model in MLmodel_dict.items():
            x_data, y_data = read_dict[model.get_read_func()]()
            test(name, x_data, y_data, kf)
    elif model_name.startswith("all"):
        kind = model_name[3:-1]
        for name, model in MLmodel_dict.items():
            if name.startswith(kind):
                x_data, y_data = read_dict[model.get_read_func()]()
                test(name, x_data, y_data, kf)
    elif MLmodel_dict.get(model_name) is None:
        print("you need provide a right model name")
    else:
        x_data, y_data = read_dict[MLmodel_dict[model_name].get_read_func()]()
        test(model_name, x_data, y_data, kf)
    if is_html:
        generate_html()


if __name__ == '__main__':
    main()
