import numpy as np
from fastdtw import fastdtw


def EuclideanDistance(x1, x2):
    if len(x1) == 106:
        x1[0] = x2[0]
    return np.sqrt(np.sum(np.square(x1 - x2)))


def DTW(x1, x2):
    if len(x1) == 106:
        len1, len2 = x1[0][0], x2[0][0]
        x1, x2 = x1[1 : len1 + 1], x2[1 : len2 + 1]
    d, _ = fastdtw(x1, x2)
    return d

def AvgEuclideanDistance(x1, x2):
    if len(x1) == 106:
        lenth = min(len(x1[0][0], x2[0][0]))
        x1, x2 = x1[1: lenth + 1], x2[1 : lenth + 1]
    else:
        lenth = min(len(x1), len(x2))
        x1, x2 = x1[0 : lenth], x2[0 : lenth]
    return EuclideanDistance(x1, x2)/lenth

def ManhattanDistance(x1, x2):
    if len(x1) == 106:
        x1[0] = x2[0]
    return np.sum(np.abs(x1 - x2))


distance_measure_dict = {
    "Euclidean": EuclideanDistance,
    "DTW": DTW,
    "Manhattan": ManhattanDistance,
    "AvgEuclidean": AvgEuclideanDistance
}
