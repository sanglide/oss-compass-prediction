from .FeatureBasedModel import FeatureBasedModel
from sklearn.naive_bayes import BernoulliNB

class Bernoulli_NB(FeatureBasedModel):
    def __init__(self):
        super(Bernoulli_NB, self).__init__()
        self.model = BernoulliNB()

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)