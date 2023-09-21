import configparser
import os
from collections import Counter

import pandas as pd

'''
read data from CSV files by [login_name/repo_name]
'''

config = configparser.ConfigParser()
config.read('config.ini')
data_path = config['path']['data_path']
problem_repo = config['path']['problem_repo'].split(",")
file_list = config['path']['file_list'].split(",")
result_path = config['path']['result_path']

if not os.path.exists(f'{result_path}raw/'):
    os.makedirs(f'{result_path}raw/')
if not os.path.exists(f'{result_path}segment_data/'):
    os.makedirs(f'{result_path}segment_data/')
if not os.path.exists(f'{result_path}segment2/'):
    os.makedirs(f'{result_path}segment2/')

def process_raw_data():
    # 1. generate "repo list"
    # Alert: ignore  https://github.com/prestodb/presto , dirty data
    reader_1 = pd.read_csv(data_path + file_list[1], iterator=True)
    repo_list = []
    i = 0
    while True:
        try:
            df_1 = reader_1.get_chunk(10000)
            repo_list_temp = list(set(df_1['label']))
            repo_list.extend(repo_list_temp)
            i = i + 1
        except StopIteration:
            break

    repo_list = [repo_list[i].replace("https://github.com/", "") for i in range(len(repo_list))]
    repo_list = list(set(repo_list))
    repo_df = pd.DataFrame(columns=['name'], data=repo_list)
    repo_df.to_csv(result_path + 'repo_list.csv')

    print("--------- finish list.csv ---------")

    # 2. for every compass_metric_model_{}.csv, for every repo, generate "raw csv"
    for metric in file_list:
        metric_suffix = metric.replace("compass_metric_model_", "")
        metric_file_path = result_path + "raw/"
        # 2.1 create file
        reader_temp = pd.read_csv(data_path + metric, iterator=True)
        while True:
            try:
                df = reader_temp.get_chunk(10000)
                # 2.1 storage data by repo
                # get repo name of 10000 lines, filter lines by reponame
                repo_name_part = list(set(df['label']))
                repo_name_part = [i.replace("\n", "") for i in repo_name_part]

                for repo in repo_name_part:
                    repo_file_path = f'{metric_file_path + repo.replace("https://github.com/", "").replace("/", "_")}_{metric_suffix}'
                    df[df["label"] == repo].to_csv(repo_file_path, mode='a', index=False)

                i = i + 1
            except StopIteration:
                break


def validation_timeline(repo):
    # review the start time and the end time of data, and the length of timeline
    print(f"==== {repo} ====")
    length = []

    df_new = pd.DataFrame()

    for f in file_list:
        file_suffix = f.replace("compass_metric_model", "")
        if os.path.exists(f'{result_path}raw/{repo.replace("/", "_")}{file_suffix}'):
            temp_data_df = pd.read_csv(f'{result_path}raw/{repo.replace("/", "_")}{file_suffix}',low_memory=False)

            temp_data_df.drop("_index", axis=1, inplace=True)
            temp_data_df.drop("uuid", axis=1, inplace=True)
            temp_data_df.drop("level", axis=1, inplace=True)
            temp_data_df.drop("type", axis=1, inplace=True)
            temp_data_df.drop("label", axis=1, inplace=True)
            temp_data_df.drop("model_name", axis=1, inplace=True)
            # temp_data_df.drop("metadata__enriched_on", axis=1, inplace=True)

            temp_data_df = temp_data_df[~temp_data_df['grimoire_creation_date'].isin(['grimoire_creation_date'])]
            temp_data_df = temp_data_df.drop_duplicates('grimoire_creation_date', keep='first')

            old_columns_name=temp_data_df.columns
            new_columns_name=[]
            for i in old_columns_name:
                if i=='grimoire_creation_date':
                    new_columns_name.append(i)
                else:
                    new_columns_name.append(i+f'{file_suffix.replace(".csv","")}')
            temp_data_df.columns = new_columns_name

            # temp_data_df.sort_values(by='grimoire_creation_date')
            # print(temp_data_df)

            if df_new.empty:
                df_new = temp_data_df
                df_new = df_new.set_index('grimoire_creation_date')
            else:
                temp_data_df = temp_data_df.set_index('grimoire_creation_date')
                # df_new = pd.merge(df_new, temp_data_df,on='grimoire_creation_date')

                df_new = pd.concat([df_new, temp_data_df], axis=1, join='inner')

            # print(
            #     f'start time : {temp_timeline[0]} ,end time : {temp_timeline[len(temp_timeline) - 1]} , time length : {len(temp_timeline)}')
            length.append(len(temp_data_df))
        else:
            print(f'!!!! {repo.replace("/", "_")}{file_suffix}  is not exists  !!!!')
    df_new.to_csv(f'{result_path}segment_data/{repo.replace("/", "_")}.csv')
    return len(set(length)) == 1 and length[0] == len(df_new)


def main_validation_timeline():
    df = pd.read_csv(result_path + "repo_list.csv")
    repo_list = list(df['name'])
    value = []
    # no_exists = []
    for repo in repo_list:
        if repo in problem_repo:
            print(f'!!!! {repo}  has some problem  !!!!')
        else:
            temp_value = validation_timeline(repo)
            value.append(temp_value)
    print(Counter(value))


if __name__ == "__main__":
    process_raw_data()
    main_validation_timeline()