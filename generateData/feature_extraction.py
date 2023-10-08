import os
import csv
import pandas as pd
from tsfresh import extract_features
import configparser

# 读取ini配置文件
config = configparser.ConfigParser()
config.read('config.ini')
result_path = config['path']['result_path']
filePaths = result_path + 'segment2/'
featurePaths = result_path + 'features/'
Label_path = result_path + 'label.csv'

labels = [   
    'name',
    'grimoire_creation_date',
    'contributor_count_activity',
    'contributor_count_bot_activity',
    'contributor_count_without_bot_activity',
    'active_C2_contributor_count_activity',
    'active_C1_pr_create_contributor_activity',
    'active_C1_pr_comments_contributor_activity',
    'active_C1_issue_create_contributor_activity',
    'active_C1_issue_comments_contributor_activity',
    'commit_frequency_activity',
    'commit_frequency_bot_activity',
    'commit_frequency_without_bot_activity',
    'org_count_activity',
    'comment_frequency_activity',
    'code_review_count_activity',
    'updated_since_activity',
    'closed_issues_count_activity',
    'updated_issues_count_activity', 
    'recent_releases_count_activity',
    'activity_score_activity',
    'contributor_count_codequality',
    'contributor_count_bot_codequality',
    'contributor_count_without_bot_codequality',
    'active_C2_contributor_count_codequality',
    'active_C1_pr_create_contributor_codequality',
    'active_C1_pr_comments_contributor_codequality',
    'commit_frequency_codequality',
    'commit_frequency_bot_codequality',
    'commit_frequency_without_bot_codequality',
    'commit_frequency_inside_codequality',
    'commit_frequency_inside_bot_codequality',
    'commit_frequency_inside_without_bot_codequality',
    'is_maintained_codequality',
    'LOC_frequency_codequality',
    'lines_added_frequency_codequality',
    'lines_removed_frequency_codequality',
    'pr_issue_linked_ratio_codequality',
    'code_review_ratio_codequality',
    'code_merge_ratio_codequality',
    'pr_count_codequality',
    'pr_merged_count_codequality',
    'pr_commit_count_codequality',
    'pr_commit_linked_count_codequality',
    'git_pr_linked_ratio_codequality',
    'code_quality_guarantee_codequality',
    'issue_first_reponse_avg_community',
    'issue_first_reponse_mid_community',
    'issue_open_time_avg_community',
    'issue_open_time_mid_community',
    'bug_issue_open_time_avg_community',
    'bug_issue_open_time_mid_community',
    'pr_open_time_avg_community',
    'pr_open_time_mid_community',
    'pr_first_response_time_avg_community',
    'pr_first_response_time_mid_community',
    'comment_frequency_community',
    'code_review_count_community',
    'updated_issues_count_community',
    'closed_prs_count_community',
    'community_support_score_community',
    'contributor_count_group_activity',
    'contributor_count_bot_group_activity',
    'contributor_count_without_bot_group_activity',
    'contributor_org_count_group_activity',
    'commit_frequency_group_activity',
    'commit_frequency_bot_group_activity',
    'commit_frequency_without_bot_group_activity',
    'commit_frequency_org_group_activity',
    'commit_frequency_org_percentage_group_activity',
    'commit_frequency_percentage_group_activity',
    'org_count_group_activity',
    'contribution_last_group_activity',
    'organizations_activity_group_activity']

custom_feature_parameters = {
    "length": None,
    "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}],
    "mean": None,
    "maximum": None,
    "minimum": None,
    "sum_values": None,
    "variance": None,
    "skewness": None,
    "kurtosis": None,
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_change": None,
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
}


def get_data_feature(Label_path, filePaths):
    LabelDict = {}
    with open(Label_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            LabelDict[row[1].replace("/", "_") + '.csv'] = row[2]
    del LabelDict['repo.csv']
    df_new = pd.DataFrame()
    for filename in os.listdir(filePaths):
        if filename.endswith('.csv'):
            if LabelDict[filename] != "-1":
                file_path = os.path.join(filePaths, filename)
                df = pd.read_csv(file_path)
                df.rename(columns={'Unnamed: 0': 'name'}, inplace=True)
                df['name'] = filename
                df = df.loc[:, labels]
                df = df.fillna(0)
                X = extract_features(df, column_id='name', column_sort='grimoire_creation_date',
                                     default_fc_parameters=custom_feature_parameters)
                X = X.reset_index()
                X = X.iloc[:, 1:]
                X['label'] = LabelDict[filename]
                df_new = pd.concat([df_new, X])
    df_new.to_csv(featurePaths + "features2.csv")


if __name__ == '__main__':
    get_data_feature(Label_path=Label_path, filePaths=filePaths)
