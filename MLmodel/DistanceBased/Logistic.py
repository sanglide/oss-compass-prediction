from .DistanceBasedModel import DistanceBasedModel
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


class Logistic(DistanceBasedModel):
    def __init__(self, distName="Euclidean"):
        super(Logistic, self).__init__(distName)
        self.logistic = LogisticRegression(max_iter=100000)
        self.raw = []

    def fit(self, X_train, Y_train):
        distances = []
        self.raw = X_train
        pbar = tqdm(total=len(X_train))
        for i in range(len(X_train)):
            distance = []
            for j in range(len(Y_train)):
                dist = self.distanceMeasure(X_train[i], X_train[j])
                distance.append(dist)
            distances.append(distance)
            pbar.update(1)
        pbar.close()
        self.logistic.fit(np.array(distances), Y_train)

    def predict(self, X_test):
        distances = []
        pbar = tqdm(total=len(X_test))
        for i in range(len(X_test)):
            distance = []
            for j in range(len(self.raw)):
                dist = self.distanceMeasure(self.raw[j], X_test[i])
                distance.append(dist)
            distances.append(distance)
            pbar.update(1)
        pbar.close()
        return self.logistic.predict(np.array(distances))
