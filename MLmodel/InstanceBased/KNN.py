from .InstanceBasedModel import InstanceBasedModel
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNN(InstanceBasedModel):
    def __init__(self, distName="Euclidean"):
        super(KNN, self).__init__(distName)
        self.knn = KNeighborsClassifier(metric=self.distanceMeasure)

    def fit(self, X_train, Y_train):
        self.knn.fit(np.array(X_train), Y_train)

    def predict(self, X_test):
        return self.knn.predict(np.array(X_test))
