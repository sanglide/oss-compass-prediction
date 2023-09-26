from .FeatureBasedModel import FeatureBasedModel
from sklearn.neighbors import KNeighborsClassifier

class KNN(FeatureBasedModel):
    def __init__(self):
        super(KNN, self).__init__()
        self.read_func = "feature-read_134"
        self.model = KNeighborsClassifier()

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
