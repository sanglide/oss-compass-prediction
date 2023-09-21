from ..baseModel import BaseModel
from ..distance_measure import *


class InstanceBasedModel(BaseModel):
    def __init__(self, distName):
        super(InstanceBasedModel, self).__init__()
        self.distanceMeasure = distance_measure_dict[distName]
        self.read_func = "common-read"
        self.baseModel = "instance"

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
