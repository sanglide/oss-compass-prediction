import configparser
from collections import Counter
import os
import pandas as pd
import datetime

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
data_period_days, forecast_gap_days, label_period_days = int(config['time_value']['data_period_days']), int(
    config['time_value'][
        'forecast_gap_days']), int(config['time_value']['label_period_days'])


def observe_terminal_event(repo_raw_data, init_idx, data_period_days, forecast_gap_days, label_period_days):
    # return whether a terminal event is observed in repo_raw_data starting from the record with index init_idx (inclusive)
    # for i in range(label_period_days):
    #     if repo_raw_data[init_idx+i]['commit_frequency'] > 0:
    #         return False
    if repo_raw_data.loc[init_idx, 'commit_frequency']=='NaN':
        return False
    if float(repo_raw_data.loc[init_idx, 'commit_frequency']) > 0:
        return False
    else:
        return True


def split_appropriate_timeline(repo_full_name, data_period_days, forecast_gap_days, label_period_days):
    # 1. read raw data from repo_full_name
    # 2. scan and judge whether terminal event happens
    if os.path.exists(f'{result_path}segment_data/{repo_full_name.replace("/", "_")}.csv'):
        repo_raw_data = pd.read_csv(f'{result_path}segment_data/{repo_full_name.replace("/", "_")}.csv')
        repo_raw_data = repo_raw_data[~repo_raw_data['grimoire_creation_date'].isin(['grimoire_creation_date'])]

    else:
        print(f'{repo_full_name} has no data!')
        return 0,pd.DataFrame()

    terminal_event_start_idx = -1
    repo_raw_data_index=repo_raw_data.index
    for idx, record in enumerate(repo_raw_data):
        # read each record in the repo's raw data in time increasing order
        if observe_terminal_event(repo_raw_data, repo_raw_data_index[idx], data_period_days, forecast_gap_days, label_period_days):
            # observed a terminal event starting from the idx (inclusive)
            terminal_event_start_idx = repo_raw_data_index[idx]
            # print(repo_raw_data.loc[idx,])
            break

    temp_date = list(repo_raw_data['grimoire_creation_date'])
    temp_date.sort()
    while True:
        if temp_date[len(temp_date)-1]=='grimoire_creation_date':
            temp_date.pop()
        else:
            break




    timeline_start_time, timeline_end_time = (min(temp_date))[:10], (max(temp_date))[:10]
    print(f'start:{timeline_start_time}  ,end: {timeline_end_time}')

    start_time_d = datetime.datetime.strptime(timeline_start_time, '%Y-%m-%d')
    end_time_d = datetime.datetime.strptime(timeline_end_time, '%Y-%m-%d')
    forecast_gap_days_d, data_period_days, label_period_days = datetime.timedelta(
        days=forecast_gap_days), datetime.timedelta(days=data_period_days), datetime.timedelta(days=label_period_days)

    if terminal_event_start_idx == -1:
        label = 1  # active repo
        # 　ｆｏｒｅｃａｓｔ　ｇａｐ的终结时间为当前最新的记录时间
        end_time_proper = end_time_d - forecast_gap_days_d
        start_time_proper = end_time_proper - data_period_days
        if start_time_proper < start_time_d:
            start_time_proper = start_time_d
        if end_time_proper < start_time_proper:
            print(f'empty data!!')
            end_time_proper = start_time_proper
    elif terminal_event_start_idx==0:
        print(f'{repo_full_name} has data problem!!!')
        label=0
        end_time_proper=start_time_d
        start_time_proper=start_time_d
    else:
        label = 0  # inactive repo
        # 　ｆｏｒｅｃａｓｔ　ｇａｐ的终结时间为　ｉｄｘ－１那条记录对应的时间（包含）
        end_time_proper = datetime.datetime.strptime(
            repo_raw_data['grimoire_creation_date'][terminal_event_start_idx - 1][:10], '%Y-%m-%d')- forecast_gap_days_d
        start_time_proper = end_time_proper - data_period_days
        if start_time_proper < start_time_d:
            start_time_proper = start_time_d
        if end_time_proper < start_time_proper:
            print(f'empty data!!')
            end_time_proper = start_time_proper

    # 按照上述ｆｏｒｅｃａｓｔ　ｇａｐ的终止时间开始，向前倒退ｆｏｒｅｃａｓｔ　ｇａｐ时间
    # 再向前倒退ｄａｔａ　ｐｅｒｉｏｄ时间，提取出来作为后续处理的数据
    data_proper = repo_raw_data[(repo_raw_data['grimoire_creation_date'] >= (start_time_proper.strftime('%Y-%m-%d'))+"T00:00:00+00:00") & (
            repo_raw_data['grimoire_creation_date'] <= (end_time_proper.strftime('%Y-%m-%d'))+"T00:00:00+00:00")]

    print(f'labels : {label} , data_init : {len(repo_raw_data)}, data_proper : {len(data_proper)}')
    return label, data_proper


def main_split_timeline():
    df = pd.read_csv(result_path + "repo_list.csv")
    repo_list = list(df['name'])
    labels = []
    for repo in repo_list:
        label, data = split_appropriate_timeline(repo, data_period_days, forecast_gap_days, label_period_days)
        labels.append({'repo': repo, 'label': label})
        data.to_csv(f'{result_path}segment2/{repo.replace("/", "_")}.csv')
    df_label = pd.DataFrame(labels)
    df_label.to_csv(f'{result_path}label.csv')


if __name__ == "__main__":
    main_split_timeline()
