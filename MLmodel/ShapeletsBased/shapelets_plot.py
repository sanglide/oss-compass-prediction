import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
if __name__=="__main__":
    df1=pd.read_csv("shapelets\index_train1.csv")
    df2=pd.read_csv("shapelets\index_test1.csv")
    df=pd.concat([df1,df2],axis=0)
    df = df.drop('grimoire_creation_date', axis=1)
    # 1. draw violin pic of 4 metrics
    drawPlot(df)


    list_repo_name=list(df['repo_name'])
    list_label=[]

    df_label=(pd.read_csv("label.csv"))
    dict_label={}
    df_label_repo_list=list(df_label["repo"])
    df_label_label_list=list(df_label["label"])
    for i in range(len(df_label)):
        dict_label[df_label_repo_list[i]]=str(df_label_label_list[i])

    for i in list_repo_name:
        list_label.append(dict_label[i])


    df_new=df.join(pd.DataFrame(list_label,columns=["labels"]))
    print(df_new.columns)

    # 2.draw violin pics of 4 metrics by labels
    drawPlotByLabel(df_new.iloc[:,[0,5]],'activity_score_activity')
    drawPlotByLabel(df_new.iloc[:,[2,5]],'community_support_score_community')
    drawPlotByLabel(df_new.iloc[:,[1,5]],'code_quality_guarantee_codequality')
    drawPlotByLabel(df_new.iloc[:,[3,5]],'organizations_activity_group_activity')