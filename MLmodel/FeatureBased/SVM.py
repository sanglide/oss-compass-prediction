from .FeatureBasedModel import FeatureBasedModel
from sklearn.svm import LinearSVC


class SVM(FeatureBasedModel):
    def __init__(self):
        super(SVM, self).__init__()
        self.model = LinearSVC()


    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
