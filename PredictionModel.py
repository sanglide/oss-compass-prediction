# This class is the parent class of all prediction methods

class PredictionModel:
    """一次模拟小狗的简单尝试"""

    # def __init__(self, xx):
    #     """初始化属性name和age"""
    #     self.xx = xx

    def fit(self,X,y):
        '''
        Train the model based on X and y
        :param X: features matrix
        :param y: success/failure of repository
        :return:
        '''
        pass

    # def save(self):
    #     '''
    #     Save the model in an appropriate way
    #
    #     :return:
    #     '''
    #     pass

    def predict(self,X):
        '''
        Using a trained model and X to predict y

        :return:
        '''
        pass