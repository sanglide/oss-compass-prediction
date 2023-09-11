import numpy as np
from fastdtw import fastdtw


def EuclideanDistance(x1, x2):
    X1 = x1.copy()
    X2 = x2.copy()
    X1[0] = X2[0].copy()
    return np.sqrt(np.sum(np.square(X1 - X2)))


def AvgEuclideanDistance(x1, x2):
    length = min(int(x1[0][0]), int(x2[0][0]))
    total = 0
    for i in range(1, length + 1):
        total += np.sum(np.square(x1[i] - x2[i]))
    return np.sqrt(total) / length


def ManhattanDistance(x1, x2):
    X1 = x1.copy()
    X2 = x2.copy()
    X1[0] = X2[0].copy()
    return np.sum(np.abs(X1 - X2))


def DTW(x1, x2):
    len1 = int(x1[0][0])
    len2 = int(x2[0][0])
    X1 = np.zeros((len1, len(x1[0])))
    X2 = np.zeros((len2, len(x2[0])))
    for i in range(0, len1):
        X1[i] = x1[i + 1].copy()
    for i in range(0, len2):
        X2[i] = x2[i + 1].copy()
    d, _ = fastdtw(X1, X2, dist=lambda x, y: np.sum(np.square(x, y)))
    return d


distance_measure_dict = {
    "Euclidean": EuclideanDistance,
    "DTW": DTW,
    "Manhattan": ManhattanDistance,
    "AvgEuclidean": AvgEuclideanDistance
}
