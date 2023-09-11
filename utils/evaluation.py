# _*_ coding : utf-8 _*_
# @Time : 2023/9/6 00:27
# @Author : Confetti-Lxy
# @File : evaluation
# @Project : project


"""

Input: Model, Test Set

Output: Formatted evaluation results

The evaluation report should include the following indicators:
1. confusion matrix
2. precision, accuracy, recall, F1-score
3. ROC, AUC

"""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


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
            plt.text(j, i, '{:.2f}'.format(class_accuracy[i][j] * 100) + '%', ha="center", va="center",
                     color="black" if conf_matrix[i][j] > thresh else "while")
    plt.savefig(name + 'Confusion_Matrix' + '.png')
    plt.close()


def evaluate(y_true, y_pred, name, printRaw=False, draw=False):
    if printRaw:
        print(f"the predict:{y_pred}")
        print(f"the label:{y_true}")
    # 导入混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    # 计算精确率
    precision = precision_score(y_true, y_pred)
    # 计算召回率
    recall = recall_score(y_true, y_pred)
    # 计算F1分数
    f1 = f1_score(y_true, y_pred)
    # 打印性能指标
    print("混淆矩阵:")
    print(conf_matrix)
    print("准确率:", accuracy)
    print("精确率:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # 计算AUC
    auc = roc_auc_score(y_true, y_pred)
    # 打印AUC和ROC曲线相关信息
    print("AUC:", auc)
    print("FPR:", fpr)
    print("TPR:", tpr)
    if draw:
        plot_matrix(conf_matrix, name)
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(name + 'ROC Curve' + '.png')
        plt.close()
