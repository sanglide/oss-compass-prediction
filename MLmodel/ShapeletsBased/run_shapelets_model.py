from MLmodel.ShapeletsBased import classify_supervised_learning

time_of_execution="222"


def shapelets_selection():
    # 1. 首先需要将四个维度单独分成训练集和测试集合（两个csv），并写到csv中（注意参考原来文件的写法）
    # project_name,time,score1..., score4  所有的项目都拼接在一个csv里面

    # 2. 调用方法，记得更改路径
    classify_supervised_learning.script_classification_ml_multi_sizes(time_of_execution)