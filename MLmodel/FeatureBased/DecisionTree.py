from .FeatureBasedModel import FeatureBasedModel
from sklearn import tree

class DecisionTree(FeatureBasedModel):
    def __init__(self):
        super(DecisionTree, self).__init__()
        self.model = tree.DecisionTreeClassifier()

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)