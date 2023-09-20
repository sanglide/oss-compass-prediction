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
config.read('../config.ini')
file_list = config['path']['file_list'].split(",")
problem_repo = config['path']['problem_repo'].split(",")
result_path = config['path']['result_path']
data_period_days, forecast_gap_days, label_period_days = int(config['time_value']['data_period_days']), int(
    config['time_value'][
        'forecast_gap_days']), int(config['time_value']['label_period_days'])
frequency_threshold = config['active_value']['frequency_threshold']
active_count_value = config['active_value']['active_count']


def split_appropriate_timeline(repo_full_name, data_period_days, forecast_gap_days, label_period_days):
    # 1. read raw data from repo_full_name
    # 2. scan and judge whether terminal event happens
    if os.path.exists(f'{result_path}segment_data/{repo_full_name.replace("/", "_")}.csv'):
        repo_raw_data = pd.read_csv(f'{result_path}segment_data/{repo_full_name.replace("/", "_")}.csv')
        repo_raw_data = repo_raw_data[~repo_raw_data['grimoire_creation_date'].isin(['grimoire_creation_date'])]
        repo_raw_data = repo_raw_data.drop_duplicates('grimoire_creation_date', keep='first')
        repo_raw_data.sort_values(by="grimoire_creation_date", axis=0, ascending=False, inplace=True)
        repo_raw_data.reset_index(drop=True)

        # log
        temp_date = list(repo_raw_data['grimoire_creation_date'])
        temp_date.sort()

        timeline_start_time, timeline_end_time = (min(temp_date))[:10], (max(temp_date))[:10]
        print(f'start:{timeline_start_time}  ,end: {timeline_end_time}')

        start_time_d = datetime.datetime.strptime(timeline_start_time, '%Y-%m-%d')
        end_time_d = datetime.datetime.strptime(timeline_end_time, '%Y-%m-%d')
        updated_d=datetime.datetime.strptime(list(repo_raw_data['metadata__enriched_on_activity'])[0][:10], '%Y-%m-%d')

        forecast_gap_days_d, data_period_days, label_period_days = datetime.timedelta(
            days=forecast_gap_days), datetime.timedelta(days=data_period_days), datetime.timedelta(
            days=label_period_days)

        terminal_event_start_idx = -1
        repo_raw_data_index = repo_raw_data.index

        if end_time_d<updated_d-label_period_days:
            terminal_event_start_idx=repo_raw_data_index[len(repo_raw_data)-1]
        else:

            for idx in range(len(repo_raw_data)):
                idx_date = datetime.datetime.strptime(
                    repo_raw_data.loc[repo_raw_data_index[idx], 'grimoire_creation_date'][:10], '%Y-%m-%d')
                if idx_date < end_time_d - label_period_days:
                    break
                if float(repo_raw_data.loc[repo_raw_data_index[idx], 'commit_frequency_activity']) <= float(frequency_threshold):
                    active_count = 0
                    for i in range(int(active_count_value)):
                        if float(repo_raw_data.loc[repo_raw_data_index[idx], 'commit_frequency_activity']) <= float(frequency_threshold):
                            active_count = active_count + 1
                        else:
                            break
                    if active_count>=float(active_count_value):
                        terminal_event_start_idx = repo_raw_data_index[idx]
                        break

        # for idx, record in enumerate(repo_raw_data):
        #     # read each record in the repo's raw data in time increasing order
        #     idx_date = datetime.datetime.strptime(
        #         repo_raw_data.loc[repo_raw_data_index[idx], 'grimoire_creation_date'][:10], '%Y-%m-%d')
        #     if idx_date > end_time_d - label_period_days and float(
        #             repo_raw_data.loc[repo_raw_data_index[idx], 'commit_frequency_activity']) <= 0:
        #         # observed a terminal event starting from the idx (inclusive)
        #         terminal_event_start_idx = repo_raw_data_index[idx]
        #         # print(repo_raw_data.loc[idx,])
        #         break

        if terminal_event_start_idx == -1:
            label = 1  # active repo
            # The end time of the forecast gap is the latest recorded time
            end_time_proper = end_time_d - forecast_gap_days_d
            start_time_proper = end_time_proper - data_period_days
            if start_time_proper < start_time_d:
                start_time_proper = start_time_d
            if end_time_proper < start_time_proper:
                print(f'empty data!!')
                label = -1
                end_time_proper = start_time_proper
        elif terminal_event_start_idx == 0:
            print(f'{repo_full_name} has data problem!!!')
            label = -1
            end_time_proper = start_time_d
            start_time_proper = start_time_d
        else:
            label = 0  # inactive repo
            # The ending time of the forecast gap is the time corresponding to the record idx-1 (inclusive)
            # Starting from the termination time of the aforementioned forecastgap, reverse the forecastgap time forward
            # Step back the dataperiod time and extract it as the data for subsequent processing
            end_time_proper = datetime.datetime.strptime(
                repo_raw_data['grimoire_creation_date'][terminal_event_start_idx - 1][:10],
                '%Y-%m-%d') - forecast_gap_days_d
            start_time_proper = end_time_proper - data_period_days
            if start_time_proper < start_time_d:
                start_time_proper = start_time_d
            if end_time_proper < start_time_proper:
                print(f'empty data!!')
                label = -1
                end_time_proper = start_time_proper

        data_proper = repo_raw_data[
            (repo_raw_data['grimoire_creation_date'] >= (
                start_time_proper.strftime('%Y-%m-%d')) + "T00:00:00+00:00") & (
                    repo_raw_data['grimoire_creation_date'] <= (
                end_time_proper.strftime('%Y-%m-%d')) + "T00:00:00+00:00")]

        print(f'labels : {label} , data_init : {len(repo_raw_data)}, data_proper : {len(data_proper)}')
        return label, data_proper

    else:
        print(f'{repo_full_name} has no data!')
        return -2, pd.DataFrame()


def main_split_timeline():
    df = pd.read_csv(result_path + "repo_list.csv")
    repo_list = list(df['name'])
    labels = []
    count_label = []
    for repo in repo_list:
        print(repo)
        label, data = split_appropriate_timeline(repo, data_period_days, forecast_gap_days, label_period_days)
        count_label.append(label)
        if not data.empty:
            labels.append({'repo': repo, 'label': label})
            data.to_csv(f'{result_path}segment2/{repo.replace("/", "_")}.csv')
    print(Counter(count_label))
    df_label = pd.DataFrame(labels)
    df_label.to_csv(f'{result_path}label.csv')


if __name__ == "__main__":
    main_split_timeline()
