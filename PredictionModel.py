# This class is the parent class of all prediction methods

class PredictionModel:

    # def __init__(self, xx):
    #     """初始化属性name和age"""
    #     self.xx = xx

    def fit(self,repo_list,y):
        '''
        Train the model based on X and y
        :param repo_list: the name list of repository
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

    def predict(self,repo_list):
        pass