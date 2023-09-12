import numpy as np
from fastdtw import fastdtw


def EuclideanDistance(x1, x2):
    X1 = x1.copy()
    X2 = x2.copy()
    X1[0] = X2[0].copy()
    return np.sqrt(np.sum(np.square(X1 - X2)))


def AvgEuclideanDistance(x1, x2):
    length = min(int(x1[0][0]), int(x2[0][0]))
    x1, x2 = x1[1: length + 1], x2[1: length + 1]
    return np.sqrt(np.sum(np.square(x1 - x2))) / length


def ManhattanDistance(x1, x2):
    X1 = x1.copy()
    X2 = x2.copy()
    X1[0] = X2[0].copy()
    return np.sum(np.abs(X1 - X2))


def DTW(x1, x2):
    len1 = int(x1[0][0])
    len2 = int(x2[0][0])
    X1 = x1[1:len1 + 1]
    X2 = x2[1:len2 + 1]
    d, _ = fastdtw(X1, X2)
    return d


distance_measure_dict = {
    "Euclidean": EuclideanDistance,
    "DTW": DTW,
    "Manhattan": ManhattanDistance,
    "AvgEuclidean": AvgEuclideanDistance
}
