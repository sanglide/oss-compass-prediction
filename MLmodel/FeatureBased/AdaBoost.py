from .FeatureBasedModel import FeatureBasedModel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 

class AdaBoost(FeatureBasedModel):
    def __init__(self):
        super(AdaBoost, self).__init__()
        base_classifier = DecisionTreeClassifier(max_depth=3) 
        n_estimators = 100   
        self.model = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=n_estimators, random_state=42)

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
