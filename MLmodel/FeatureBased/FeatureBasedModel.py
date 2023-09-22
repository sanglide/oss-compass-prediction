from ..baseModel import BaseModel


class FeatureBasedModel(BaseModel):
    def __init__(self):
        super(FeatureBasedModel, self).__init__()
        self.read_func = "fixed-feature-read"
        self.baseModel = "feature"

    def fit(self, X_train, Y_train):
        """
        Train the model based on X_train and Y_train
        :param X_train: the name list of repository
        :param Y_train: success/failure of repository
        :return:
        """
        pass

    # def save(self):
    #     '''
    #     Save the model in an appropriate way
    #
    #     :return:
    #     '''
    #     pass

    def predict(self, X_test):
        """
        Use the model to predict the Y_test corresponding to X_test
        :param X_test: the data needed to predict
        :return: the Y_test corresponding to X_test
        """
        pass
