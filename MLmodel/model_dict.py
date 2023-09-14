# 用于存放模型的字典类

from .FeatureBased.KNN import KNN as FeatureKNN
from .FeatureBased.Logistic import Logistic
from .InstanceBased.KNN import KNN as InstanceKNN

MLmodel_dict = {
    "Feature-Logistic-Euclidean": Logistic('Euclidean'),
    "Feature-Logistic-AvgEuclidean": Logistic('AvgEuclidean'),
    "Feature-Logistic-DTW": Logistic('DTW'),
    "Feature-Logistic-Manhattan": Logistic('Manhattan'),
    "Feature-KNN-Euclidean": FeatureKNN('Euclidean'),
    "Feature-KNN-AvgEuclidean": FeatureKNN('AvgEuclidean'),
    "Feature-KNN-DTW": FeatureKNN('DTW'),
    "Feature-KNN-Manhattan": FeatureKNN('Manhattan'),
    "Instance-KNN-Euclidean": InstanceKNN('Euclidean'),
    "Instance-KNN-AvgEuclidean": InstanceKNN('AvgEuclidean'),
    "Instance-KNN-DTW": InstanceKNN('DTW'),
    "Instance-KNN-Manhattan": InstanceKNN('Manhattan')
}