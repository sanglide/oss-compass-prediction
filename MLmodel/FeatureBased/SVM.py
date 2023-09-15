from .FeatureBasedModel import FeatureBasedModel
from sklearn import svm

class SVM(FeatureBasedModel):
    def __init__(self):
        super(SVM, self).__init__()
        self.model = svm.SVC(kernel='linear')

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)