"""

Input: Model, Test Set

Output: Formatted evaluation results

The evaluation report should include the following indicators:
1. confusion matrix
2. precision, accuracy, recall, F1-score
3. ROC, AUC

"""

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import os
from MLmodel.model_dict import MLmodel_dict


def plot_matrix(conf_matrix, name):
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    class_accuracy = conf_matrix / conf_matrix.sum(axis=1)[:, None]
    plt.imshow(class_accuracy, cmap=plt.get_cmap('Blues'))
    plt.grid(False)
    plt.colorbar()
    axis_label = np.array(['Negative', 'Positive'])
    num_local = np.array(range(len(axis_label)))
    plt.xticks(num_local, axis_label, fontsize=12)
    plt.yticks(num_local, axis_label, fontsize=12)
    thresh = 0.5
    for i in range(2):
        for j in range(2):
            plt.text(j, i, class_accuracy[i][j], ha="center", va="center",
                     color="black" if conf_matrix[i][j] < thresh else "white")
    path = 'data/pic/' + name + '/'
    path_html = 'data/html/' + name + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path_html):
        os.makedirs(path_html)
    plt.savefig(path + name + 'Confusion_Matrix' + '.png')
    plt.savefig(path_html + name + 'Confusion_Matrix' + '.png')
    plt.close()


def evaluate(y_true, y_pred, name, draw=False):
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # 打印性能指标
    print("混淆矩阵:")
    print(conf_matrix)
    print("准确率:", accuracy)
    print("精确率:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    # 打印AUC和ROC曲线相关信息
    print("AUC:", auc)
    print("FPR:", fpr)
    print("TPR:", tpr)
    if draw:
        plot_matrix(conf_matrix, name)
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        path = 'data/pic/' + name
        path_html = 'data/html/'+name        
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_html):
            os.makedirs(path_html)
        plt.savefig(path + '/' + name + '.png')
        plt.savefig(path_html + '/' + name + '.png')
        plt.close()


def test(name, x_data, y_data, kf):
    print("==================================" + name + "==================================")
    m = MLmodel_dict[name]
    Y_pred, Y_test = np.array([]), np.array([])
    for train_index, test_index in kf.split(x_data, y_data):
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        Y_pred = np.concatenate([Y_pred, y_pred])
        Y_test = np.concatenate([Y_test, y_test])
    evaluate(Y_test, Y_pred, name, draw=True)


def mix_test(model_list, x_data, y_data, kf):
    model_type = MLmodel_dict.get(model_list[0]).get_base_model()
    for name in model_list:
        m = MLmodel_dict.get(name)
        if model_type != m.get_base_model():
            print("the models inherit from different parent classes")
            assert 0
    name = "-".join(model_list)
    print("==================================" + name + "==================================")
    Y_preds, Y_test = [], []
    for train_index, test_index in kf.split(x_data, y_data):
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        y_cnt = np.zeros(len(y_test), dtype=int)
        for mixed_model in model_list:
            m = MLmodel_dict.get(mixed_model)
            if m is not None:
                m.fit(X_train, y_train)
                y_hat = m.predict(X_test)
                y_cnt += (y_hat == 1).astype(int) - (y_hat == 0).astype(int)
        y_pred = (y_cnt >= 0).astype(int)
        Y_preds.extend(y_pred)
        Y_test.extend(y_test)
    evaluate(np.array(Y_test), np.array(Y_preds), name, draw=True)
