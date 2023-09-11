from InstanceBased.InstanceBasedModel import InstanceBasedModel
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNN(InstanceBasedModel):
    def __init__(self, distName="Euclidean"):
        super(KNN, self).__init__(distName)
        self.knn = KNeighborsClassifier(metric=self.distanceMeasure)

    def fit(self, X_train, Y_train):
        X_in = X_train.reshape(len(X_train), -1)
        self.knn.fit(np.array(X_in), Y_train)

    def predict(self, X_test):
        X_in = X_test.reshape(len(X_test), -1)
        return self.knn.predict(np.array(X_in))
