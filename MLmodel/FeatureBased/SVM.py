from .FeatureBasedModel import FeatureBasedModel
from sklearn.svm import LinearSVC, SVC

class SVM(FeatureBasedModel):
    def __init__(self):
        super(SVM, self).__init__()
        self.read_func = "feature-read_134"
        self.model = LinearSVC()
        # self.model = SVC(kernel='sigmoid', C=1.0) 

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)