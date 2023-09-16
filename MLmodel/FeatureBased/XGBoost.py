from .FeatureBasedModel import FeatureBasedModel
from xgboost import XGBClassifier

class XGBoost(FeatureBasedModel):
    def __init__(self):
        super(XGBoost, self).__init__()
        self.model = XGBClassifier(
                    objective='binary:logistic',  # 二分类问题
                    learning_rate=0.1,            # 学习率
                    n_estimators=100,             # 树的数量
                    max_depth=3,                  # 树的最大深度
                    random_state=42               # 随机种子，可选
                    )


    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
