from .FeatureBasedModel import FeatureBasedModel
from sklearn.naive_bayes import GaussianNB

class Gaussian_NB(FeatureBasedModel):
    def __init__(self):
        super(Gaussian_NB, self).__init__()
        self.model = GaussianNB()

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)