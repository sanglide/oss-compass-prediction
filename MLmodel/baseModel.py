# This class is the parent class of all ML prediction methods

class BaseModel:

    def __init__(self):
        """初始化属性name和age"""
        self.read_func = None
        self.baseModel = None

    def get_read_func(self):
        return self.read_func
    
    def get_base_model(self):
        return self.baseModel

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
