import configparser
import time
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from MLmodel.ShapeletsBased import classify_supervised_learning
import random
import os
from pyts.classification import LearningShapelets
from pyts.transformation import ShapeletTransform
from sklearn.metrics import classification_report

config = configparser.ConfigParser()
config.read('../../config.ini', 'utf-8')
result_path = config['path']['result_path']
problem_repo = config['path']['problem_repo'].split(",")


def store_csv_with_repo_list(repo_list, filename, y):
    X_test1, X_test2, X_test3, X_test4, y_new = [], [], [], [], []

    read_path = f'{result_path}segment2\\'
    df_list = []
    count = 0
    for repo in repo_list:
        if repo not in problem_repo and y[count] != -1:
            # y_new.append(y[count])
            df = pd.read_csv(f'{read_path}{repo.replace("/", "_")}.csv', usecols=[1, 21, 47, 63, 79])
            df.fillna(0, inplace=True)

            if len(df) < 105:
                lst1, lst2, lst3, lst4 = list(df.iloc[:, 1]), list(df.iloc[:, 2]), list(df.iloc[:, 3]), list(
                    df.iloc[:, 4])
                zero = [float(0) for i in range(105 - len(df))]
                lst1.extend(zero)
                lst2.extend(zero)
                lst3.extend(zero)
                lst4.extend(zero)
                X_test1.append(lst1)
                X_test2.append(lst2)
                X_test3.append(lst3)
                X_test4.append(lst4)

                y_new.append(y[count])

                print(f'jump {repo}')
            else:
                X_test1.append(list(df.iloc[:, 1]))
                X_test2.append(list(df.iloc[:, 2]))
                X_test3.append(list(df.iloc[:, 3]))
                X_test4.append(list(df.iloc[:, 4]))

                y_new.append(y[count])

            repo_name_list = [repo for i in range(len(df))]
            df['repo_name'] = repo_name_list
            df_list.append(df)
        count = count + 1
    df_new = pd.concat(df_list, axis=0, ignore_index=True)
    # df_new=df_new['grimoire_creation_date', 'activity_score_activity',
    #         'community_support_score_community', 'code_quality_guarantee_codequality',
    #         'organizations_activity_group_activity']
    print(df_new.shape)
    if not os.path.exists(f'{result_path}shapelets\\'):
        os.makedirs(f'{result_path}shapelets\\')
    df_new.to_csv(f'{result_path}shapelets\\{filename}.csv', index=False)
    return X_test1, X_test2, X_test3, X_test4, y_new
def closePlots():
    plt.clf()
    plt.cla()
    plt.close("all")
    time.sleep(0.5)

def shapelets_selection():
    # 1. 首先需要将四个维度单独分成训练集和测试集合（两个csv），并写到csv中（注意参考原来文件的写法）
    df = pd.read_csv(result_path + "label.csv")
    repo_list = list(df['repo'])
    label = list(df['label'])
    lst = random.sample(range(len(repo_list)), 500)
    set_lst = set(lst)
    not_lst = [item for item in range(len(repo_list)) if item not in set_lst]
    # 选择100个仓库作为训练集，剩下的作为测试集
    repo_list_train = [repo_list[i] for i in lst]
    y_train = [label[i] for i in lst]
    repo_list_test = [repo_list[i] for i in not_lst]
    y_test = [label[i] for i in not_lst]

    X_train1, X_train2, X_train3, X_train4, y_train = store_csv_with_repo_list(repo_list_train, "index_train1", y_train)
    X_test1, X_test2, X_test3, X_test4, y_test = store_csv_with_repo_list(repo_list_test, "index_test1", y_test)

    X_train = [[X_train1[i][j] + X_train2[i][j] + X_train3[i][j] + X_train4[i][j] for j in range(len(X_train1[i]))] for
               i in range(len(X_train1))]
    # X_train=[[X_train1[i][j] + X_train2[i][j]+ X_train3[i][j]   for j in range(len(X_train1[i]))] for i in range(len(X_train1))]
    X_test = [[X_test1[i][j] + X_test2[i][j] + X_test3[i][j] + X_test4[i][j] for j in range(len(X_test1[i]))] for i in
              range(len(X_test1))]
    # X_test=[[X_test1[i][j] + X_test2[i][j]+ X_test3[i][j] for j in range(len(X_test1[i])) ]for i in range(len(X_test1))]

    print(f'================== start learning =====================')
    print(f'train set : {len(X_train)} * {len(X_train[0])}')
    print(f'test set : {len(X_test)} * {len(X_test[0])}')

    print(f'y_test : {Counter(y_test)} , y_train : {Counter(y_train)}')

    # clf=LearningShapelets(random_state=42,tol=0.01)
    clf=LearningShapelets(random_state=42,tol=0.01)
    clf.fit(X_train,y_train)
    y_predict=clf.predict(X_test)
    print(classification_report(y_test, y_predict))

    # project_name,time,score1..., score4  所有的项目都拼接在一个csv里面
    print(f'================ find shapelets ====================')
    st = ShapeletTransform(window_sizes=[3, 6, 9, 12],
                           random_state=42, sort=True)
    X_new = st.fit_transform(X_train, y_train)

 #    # print(st.indices_[:4])
 #    aa=[[ 15,63 ,69],
 # [126 ,32 ,38],
 # [ 15 , 84 , 93],
 # [253 , 22 , 31]]

    # Visualize the four most discriminative shapelets
    print(f'===================== end train ============================')
    for i, index in enumerate(st.indices_):
    # for i, index in enumerate(aa):
        plt.figure(figsize=(6, 4))
        idx, start, end = index
        # plt.plot(X_train[idx], color='C{}'.format(i),
        #          label='Sample {}'.format(idx))
        plt.plot(np.arange(start, end), np.array(X_train1)[idx, start:end],
                 color='C{}'.format(i), label='activity_score {}'.format(idx))
        plt.plot(np.arange(start, end), np.array(X_train2)[idx, start:end],
                 color='C{}'.format(i+1), label='community_support_score {}'.format(idx))
        plt.plot(np.arange(start, end), np.array(X_train3)[idx, start:end],
                 color='C{}'.format(i+2), label='code_quality_guarantee {}'.format(idx))
        plt.xlabel('Time', fontsize=12)
        plt.title(f'The {i}-th most discriminative shapelets', fontsize=14)
        plt.legend(loc='best', fontsize=8)
        plt.savefig(f'./shapelets_fig/shapelets_{i}_{idx}.png')

        closePlots()

    # 2. 调用方法，记得更改路径


if __name__ == "__main__":
    shapelets_selection()
