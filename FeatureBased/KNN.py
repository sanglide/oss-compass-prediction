from FeatureBased.DistanceBasedModel import DistanceBasedModel
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNN(DistanceBasedModel):
    def __init__(self, distName="Euclidean"):
        super(KNN, self).__init__(distName)
        self.knn = KNeighborsClassifier()
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
        self.knn.fit(np.array(distances), Y_train)

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
        return self.knn.predict(np.array(distances))
