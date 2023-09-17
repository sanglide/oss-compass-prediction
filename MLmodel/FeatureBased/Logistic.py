from .FeatureBasedModel import FeatureBasedModel
from sklearn.linear_model import LogisticRegression

class Logistic(FeatureBasedModel):
    def __init__(self):
        super(Logistic, self).__init__()
        self.model = LogisticRegression()

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)