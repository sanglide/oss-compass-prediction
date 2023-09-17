import numpy as np
from .FeatureBasedModel import FeatureBasedModel
from sklearn.linear_model import LogisticRegression


class Logistic(FeatureBasedModel):
    def __init__(self, distName="Euclidean"):
        super(Logistic, self).__init__()
        self.logistic = LogisticRegression(max_iter=10000000)

    def fit(self, X_train, Y_train):
        self.logistic.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.logistic.predict(X_test)
