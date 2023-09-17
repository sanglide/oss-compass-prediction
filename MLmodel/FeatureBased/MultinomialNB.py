from .FeatureBasedModel import FeatureBasedModel
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

class Multinomial_NB(FeatureBasedModel):
    def __init__(self):
        super(Multinomial_NB, self).__init__()
        self.model = MultinomialNB()

    def fit(self, X_train, Y_train):
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        self.model.fit(X_train_scaled, Y_train)

    def predict(self, X_test):
        scaler = MinMaxScaler()
        scaler.fit(X_test)
        X_test_scaled = scaler.transform(X_test)
        return self.model.predict(X_test_scaled)