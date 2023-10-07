import time

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def closePlots():
    plt.clf()
    plt.cla()
    plt.close("all")
    time.sleep(0.5)
def drawPlot(df):
    # sns.kdeplot(df, shade=True)
    # sns.boxplot(x=df)
    sns.violinplot(data=df)
    plt.xticks(rotation=70)
    plt.show()
def drawPlotByLabel(df,name):
    sns.violinplot(x='labels',y=name,data=df)
    plt.show()
def draw_TS():

    # Parameters
    n_samples, n_timestamps = 100, 48

    # Toy dataset
    rng = np.random.RandomState(41)
    X = rng.randn(n_samples, n_timestamps)

    # Plot the first time series
    plt.figure(figsize=(6, 5))
    plt.plot(X[0], 'o-')
    plt.xlabel('Time', fontsize=14)
    plt.title('Plotting a time series', fontsize=16)
    plt.tight_layout()
    plt.show()
def draw_plot(df,name,label):
    df.plot(x='grimoire_creation_date',y=['activity_score_activity',
             'community_support_score_community', 'code_quality_guarantee_codequality','organizations_activity_group_activity'])
    plt.savefig(f'./fig/{name}_label_{label}.png')
    closePlots()


if __name__=="__main__":
    result_path="C:\phd-one\project\oss-compass-result\\"

    df1=pd.read_csv(result_path+"shapelets\index_train1.csv")
    df2=pd.read_csv(result_path+"shapelets\index_test1.csv")

    df=pd.concat([df1,df2],axis=0)
    df = df.drop('grimoire_creation_date', axis=1)
    # 1. draw violin pic of 4 metrics
    # drawPlot(df)


    list_repo_name=list(df['repo_name'])
    list_label=[]

    df_label=(pd.read_csv(result_path+"label.csv"))

    dict_label={}
    df_label_repo_list=list(df_label["repo"])
    df_label_label_list=list(df_label["label"])
    # for i in range(len(df_label)):
    #     dict_label[df_label_repo_list[i]]=str(df_label_label_list[i])
    #
    # for i in list_repo_name:
    #     list_label.append(dict_label[i])

    for i in range(len(df_label_repo_list)):
        name=df_label_repo_list[i].replace("/","_")
        df=pd.read_csv(f'E:\phd-one\project\oss-compass-result\segment2\{name}.csv')
        draw_plot(df,name,df_label_label_list[i])


    df_new=df.join(pd.DataFrame(list_label,columns=["labels"]))
    print(df_new.columns)

    # 2.draw violin pics of 4 metrics by labels
    # drawPlotByLabel(df_new.iloc[:,[0,5]],'activity_score_activity')
    # drawPlotByLabel(df_new.iloc[:,[2,5]],'community_support_score_community')
    # drawPlotByLabel(df_new.iloc[:,[1,5]],'code_quality_guarantee_codequality')
    # drawPlotByLabel(df_new.iloc[:,[3,5]],'organizations_activity_group_activity')

    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    sns.violinplot(x='labels',y='activity_score_activity',data=df_new.iloc[:,[0,5]],ax=ax1)
    sns.violinplot(x='labels',y='community_support_score_community',data=df_new.iloc[:,[2,5]],ax=ax2)
    sns.violinplot(x='labels',y='code_quality_guarantee_codequality',data=df_new.iloc[:,[1,5]],ax=ax3)
    # ax1.set_ylabel('')
    # ax2.set_ylabel('')
    # ax3.set_ylabel('')

    plt.show()

