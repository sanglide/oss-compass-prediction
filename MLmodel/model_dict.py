# 用于存放模型的字典类

from .DistanceBased.KNN import KNN as DistanceKNN
from .DistanceBased.Logistic import Logistic
from .InstanceBased.KNN import KNN as InstanceKNN
from .FeatureBased.KNN import KNN as FeatureKNN
from .FeatureBased.SVM import SVM as FeatureSVM
from .FeatureBased.RandomForest import RandomForest
from .FeatureBased.XGBoost import XGBoost
from .FeatureBased.LogisticRegression import Logistic as FeatureLogistic

MLmodel_dict = {
    "Distance-Logistic-Euclidean": Logistic('Euclidean'),
    "Distance-Logistic-AvgEuclidean": Logistic('AvgEuclidean'),
    "Distance-Logistic-DTW": Logistic('DTW'),
    "Distance-Logistic-Manhattan": Logistic('Manhattan'),
    "Distance-KNN-Euclidean": DistanceKNN('Euclidean'),
    "Distance-KNN-AvgEuclidean": DistanceKNN('AvgEuclidean'),
    "Distance-KNN-DTW": DistanceKNN('DTW'),
    "Distance-KNN-Manhattan": DistanceKNN('Manhattan'),
    "Instance-KNN-Euclidean": InstanceKNN('Euclidean'),
    "Instance-KNN-AvgEuclidean": InstanceKNN('AvgEuclidean'),
    "Instance-KNN-DTW": InstanceKNN('DTW'),
    "Instance-KNN-Manhattan": InstanceKNN('Manhattan'),
    "Feature-KNN": FeatureKNN(),
    "Feature-SVM": FeatureSVM(),
    "Feature-RandomForest": RandomForest(),
    "Feature-XGBoost": XGBoost(),
    "Feature-Logistic": FeatureLogistic(),
}
