from .FeatureBasedModel import FeatureBasedModel
from sklearn.ensemble import RandomForestClassifier

class RandomForest(FeatureBasedModel):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.model = RandomForestClassifier()

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)