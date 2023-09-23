import configparser
import copy
import os.path

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pylab as plt
import sklearn.naive_bayes
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# from utilsPY import openreadtxt
import util

from classify_simple_dist import ClassifySimpleDist


def drawMatrix(y_test, y_pred, name):
    print("----------draw {0}-------------".format(name))
    # 混淆矩阵并可视化
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)  # 输出混淆矩阵
    print(confmat)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()
    plt.savefig("confusion_matrix_" + name + ".jpg", dpi=150)
    # 召回率、准确率、F1
    print('precision:%.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('recall:%.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1:%.3f' % f1_score(y_true=y_test, y_pred=y_pred))


def ten_fold_cross_validation(repos, x, y, model):
    # pipeline = make_pipeline(StandardScaler(), model)

    # 设置交叉验证折数cv=10 表示使用带有十折的StratifiedKFold，再把管道和数据集传到交叉验证对象中
    # scores = cross_val_score(pipeline, X=x, y=y, cv=10, n_jobs=1, scoring='accuracy')
    # print('Cross Validation accuracy scores: %s' % scores)
    # print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    # 创建一个用于得到不同训练集和测试集样本的索引的StratifiedKFold实例，折数为10
    strtfdKFold = StratifiedKFold(n_splits=10, shuffle=True)
    # 把特征和标签传递给StratifiedKFold实例
    kfold = strtfdKFold.split(x, y)
    y_pred_sum = []
    y_true_sum = []
    repos_sum = []
    # 循环迭代，（K-1）份用于训练，1份用于验证，把每次模型的性能记录下来。
    scores = []
    for k, (train, test) in enumerate(kfold):
        # pipeline.fit(x.iloc[train], y.iloc[train])
        model.fit(pd.DataFrame(x.iloc[train]), y.iloc[train])
        # y_pred = pipeline.predict(x.iloc[test])
        y_pred = model.predict(pd.DataFrame(x.iloc[test]))
        y_pred_sum.extend(y_pred)
        y_true_sum.extend(y.iloc[test])
        repos_sum.extend(repos.iloc[test])
    return y_pred_sum, y_true_sum, repos_sum


def direct_classify(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred, y_test


def execute_classify_direct(label_period_months, forecast_gap_months, data_period_months,
                            n_shapelets, window_size, window_step, n_channels,
                            test_data_path, result_folder_path, train_data_path, evolution_event_selection,
                            is_multi_sizes=False, do_discritize=False):
    # shapelet_file_name = util.get_shapelet_file_name(label_period_months, forecast_gap_months, data_period_months,
    #                                                  n_shapelets, window_size, window_step, n_channels, train_data_path,
    #                                                  evolution_event_selection)
    #
    # train_feature_label_file_name = f"{result_folder_path}features/{shapelet_file_name}_32.csv"
    # if is_multi_sizes:
    #     train_feature_label_file_name = f"{train_feature_label_file_name}.multi_sizes.csv"
    train_feature_label_data = pd.read_csv(train_feature_label_file_name)
    y_train = train_feature_label_data['label']
    heads = [col for col in train_feature_label_data.columns]
    # heads.remove('Unnamed: 0')
    heads.remove('label')
    heads.remove('repo')
    x_train = pd.DataFrame(train_feature_label_data[heads])
    if do_discritize:
        disc_est = sklearn.preprocessing.KBinsDiscretizer(n_bins=[7 for i in range(len(heads))], encode='ordinal').fit(
            x_train)
        x_train = disc_est.transform(x_train)

    test_feature_label_file_name = f"{result_folder_path}features/{shapelet_file_name}_696.csv"
    if is_multi_sizes:
        test_feature_label_file_name = f"{test_feature_label_file_name}.multi_sizes.csv"
    test_feature_label_data = pd.read_csv(test_feature_label_file_name)
    y_test = test_feature_label_data['label']
    test_repos = test_feature_label_data['repo']
    heads = [col for col in test_feature_label_data.columns]
    # heads.remove('Unnamed: 0')
    heads.remove('label')
    heads.remove('repo')
    x_test = pd.DataFrame(test_feature_label_data[heads])
    if do_discritize:
        x_test = disc_est.transform(x_test)

    models = [
        # sklearn.naive_bayes.GaussianNB(),
        sklearn.tree.DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=10),
        RandomForestClassifier(n_estimators=50),
        RandomForestClassifier(n_estimators=70),
        RandomForestClassifier(n_estimators=100),
        # RandomForestClassifier(n_estimators=150),
        # RandomForestClassifier(n_estimators=170),
        # AdaBoostClassifier(n_estimators=100),
        # BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
        # BaggingClassifier(sklearn.tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5),
        # sklearn.svm.SVC()
        # sklearn.linear_model.LogisticRegression()
    ]
    model_names = [
        # "GaussianNB",
        "DecisionTreeClassifier",
        "RandomForestClassifier_10",
        "RandomForestClassifier_50",
        "RandomForestClassifier_70",
        "RandomForestClassifier_100",
        # "RandomForestClassifier_150",
        # "RandomForestClassifier_170",
        # "AdaBoostClassifier_100",
        # "BaggingClassifier_KNN",
        # "BaggingClassifier_DecisionTree",
        # "SVM_SVC_RBF",
        # "LogisticRegression"
    ]

    if do_discritize:
        models_name_disc = []
        for name in model_names:
            models_name_disc.append(f"{name}_discritize")
        model_names = models_name_disc

    for i in range(len(models)):
        y_pred_sum, y_true_sum = direct_classify(x_train, y_train, x_test, y_test, models[i])
        cs = ClassifySimpleDist()
        cs.load(f"{result_folder_path}{shapelet_file_name}.model")
        # def report(self, test_repos, y_true, y_pred, label_period_months, forecast_gap_months, data_period_months, result_folder_path,
        #                report_file_name="prediction_report.csv", detailed_file_name="prediction_report_details.csv"):  
        cs.report(test_repos, y_true_sum, y_pred_sum, label_period_months, forecast_gap_months, data_period_months,
                  result_folder_path, evolution_event_selection,
                  report_file_name=f"prediction_report_direct_{model_names[i]}.csv",
                  detailed_file_name=f"prediction_report_direct_{model_names[i]}_detail.csv")


def execute_classify_ten_fold(label_period_months, forecast_gap_months, data_period_months,
                              n_shapelets, window_size, window_step, n_channels,
                              data_path, result_folder_path, train_data_path, evolution_event_selection,
                              is_multi_sizes=False):
    shapelet_file_name = util.get_shapelet_file_name(label_period_months, forecast_gap_months, data_period_months,
                                                     n_shapelets, window_size, window_step, n_channels, train_data_path,
                                                     evolution_event_selection)
    feature_label_file_name = None
    if "index_productivity_32" in data_path:
        feature_label_file_name = f"{result_folder_path}features/{shapelet_file_name}_32.csv"
    else:
        feature_label_file_name = f"{result_folder_path}features/{shapelet_file_name}_696.csv"

    if is_multi_sizes:
        feature_label_file_name = f"{feature_label_file_name}.multi_sizes.csv"
    feature_label_data = pd.read_csv(feature_label_file_name)
    y = feature_label_data['label']
    test_repos = feature_label_data['repo']
    heads = [col for col in feature_label_data.columns]
    # heads.remove('Unnamed: 0')
    heads.remove('label')
    heads.remove('repo')
    x = pd.DataFrame(feature_label_data[heads])

    models = [
        # sklearn.naive_bayes.GaussianNB(),
        sklearn.tree.DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=10),
        RandomForestClassifier(n_estimators=50),
        RandomForestClassifier(n_estimators=70),
        RandomForestClassifier(n_estimators=100),
        # RandomForestClassifier(n_estimators=150),
        # RandomForestClassifier(n_estimators=170),
        # AdaBoostClassifier(n_estimators=100),
        # BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
        # BaggingClassifier(sklearn.tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5),
        # MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
        # sklearn.linear_model.LogisticRegression()
    ]
    model_names = [
        # "GaussianNB",
        "DecisionTreeClassifier",
        "RandomForestClassifier_10",
        "RandomForestClassifier_50",
        "RandomForestClassifier_70",
        "RandomForestClassifier_100",
        # "RandomForestClassifier_150",
        # "RandomForestClassifier_170",
        # "AdaBoostClassifier_100",
        # "BaggingClassifier_KNN",
        # "BaggingClassifier_DecisionTree",
        # "MLPClassifier",
        # "LogisticRegression"
    ]

    if "index_productivity_32" in data_path:
        models_name_disc = []
        for name in model_names:
            models_name_disc.append(f"{name}_dataset32")
        model_names = models_name_disc
    else:
        models_name_disc = []
        for name in model_names:
            models_name_disc.append(f"{name}_dataset696")
        model_names = models_name_disc

    for i in range(len(models)):
        y_pred_sum, y_true_sum, repos_sum = ten_fold_cross_validation(test_repos, x, y, models[i])
        cs = ClassifySimpleDist()
        cs.load(f"{result_folder_path}{shapelet_file_name}.model")
        cs.report(repos_sum, y_true_sum, y_pred_sum, label_period_months, forecast_gap_months, data_period_months,
                  result_folder_path, evolution_event_selection,
                  report_file_name=f"prediction_report_tenfold_{model_names[i]}.csv",
                  detailed_file_name=f"prediction_report_tenfold_{model_names[i]}_detail.csv")


def script_classification_ml_multi_sizes(time_of_execution):
    config = configparser.ConfigParser()
    config.read('../../config.ini')
    result_root_dir = config['path']['result_path']
    data_dir = config['path']['data_path']
    result_folder_path = f"{result_root_dir}{time_of_execution}/"

    n_shapelets = int(config['shapelets']['n_shapelets'])
    window_step = int(config['shapelets']['window_step'])
    n_channels = int(config['shapelets']['n_channels'])
    n_jobs = int(config['shapelets']['n_jobs'])


    list_data_period_months, list_forecast_gap_months, list_label_period_months = [int(
        config['time_value']['data_period_days']) / 30], [int(
        config['time_value'][
            'forecast_gap_days'])/30], [int(config['time_value']['label_period_days']) / 30]
    # 以data point为单位
    list_window_size = [int(x) for x in config['shapelets']['list_window_size'].split(',')]
    list_evolution_event_selection = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2],
                                [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]

    train_data_path = f"{data_dir}index_train1.csv"
    test_data_path = f"{data_dir}index_test1.csv"

    count = 1
    total_count = len(list_label_period_months) * len(list_forecast_gap_months) * len(list_data_period_months) * len(
        list_evolution_event_selection)

    for label_period_months in list_label_period_months:
        for forecast_gap_months in list_forecast_gap_months:
            for data_period_months in list_data_period_months:
                for evolution_event_selection in list_evolution_event_selection:
                    util.test_kill_and_exit()
                    window_size = 3  # this is a dummy value

                    print(f"************ Classify {count}/{total_count} direct from train to test")

                    execute_classify_direct(label_period_months, forecast_gap_months, data_period_months,
                                            n_shapelets, window_size, window_step, n_channels,
                                            test_data_path, result_folder_path, train_data_path,
                                            evolution_event_selection, is_multi_sizes=True, do_discritize=False)

                    print(f"************ Classify {count}/{total_count} ten fold on test")

                    execute_classify_ten_fold(label_period_months, forecast_gap_months, data_period_months,
                                              n_shapelets, window_size, window_step, n_channels,
                                              test_data_path, result_folder_path, train_data_path,
                                              evolution_event_selection, is_multi_sizes=True)

                    execute_classify_ten_fold(label_period_months, forecast_gap_months, data_period_months,
                                              n_shapelets, window_size, window_step, n_channels,
                                              train_data_path, result_folder_path, train_data_path,
                                              evolution_event_selection, is_multi_sizes=True)
                    count += 1
