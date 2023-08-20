import configparser
from collections import Counter
import os
import pandas as pd
'''

Process raw data and store it according to requirements,
with one CSV for each project, and the data format is as follows：

<timeline, feature 1, feature 2,...,feature n>

The timeline of each feature is common.

'''

config = configparser.ConfigParser()
config.read('config.ini')
file_list = config['path']['file_list'].split(",")
problem_repo = config['path']['problem_repo'].split(",")
result_path = config['path']['result_path']



def observe_terminal_event(repo_raw_data, init_idx,data_period_days, forecast_gap_days, label_period_days):
    # return whether a terminal event is observed in repo_raw_data starting from the record with index init_idx (inclusive)
    for i in range(label_period_days):
        if repo_raw_data[init_idx+i]['num_commits'] > 0:
            return False
    return True


def split_appropriate_timeline(repo_full_name, data_period_days, forecast_gap_days, label_period_days):
    '''
    Both input and output are data matrices, only cutting appropriate data segments according to preset methods

    :param data:
    :return:
    '''
    # 1. read raw data from repo_full_name
    # 2. scan and judge whether terminal event happens

    repo_raw_data=pd.read_csv(f'{result_path}segment/{repo_full_name.replace("/","_")}.csv')

    terminal_event_start_idx = -1
    for idx, record in enumerate(repo_raw_data):
        # read each record in the repo's raw data in time increasing order
        if observe_terminal_event(repo_raw_data, idx):
            # observed a terminal event starting from the idx (inclusive)
            terminal_event_start_idx = idx
            break

    # if terminal_event_start_idx == -1:
        #　未观察到终止事件，该项目是活跃的
        #　ｆｏｒｅｃａｓｔ　ｇａｐ的终结时间为当前最新的记录时间
    # else:
        #　观察到终止时间，该项目是不活跃的
        #　ｆｏｒｅｃａｓｔ　ｇａｐ的终结时间为　ｉｄｘ－１那条记录对应的时间（包含）

    # 按照上述ｆｏｒｅｃａｓｔ　ｇａｐ的终止时间开始，向前倒退ｆｏｒｅｃａｓｔ　ｇａｐ时间
    # 再向前倒退ｄａｔａ　ｐｅｒｉｏｄ时间，提取出来作为后续处理的数据

    pass


# if __name__=="__main__":
