from xgboost import XGBRegressor
from .basemodel import BaseModel

class XGBRModel(BaseModel):
    def define_params(self, alpha, **kwargs):
        return {
            "quantile_alpha": alpha,
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "random_state": 0,
            **kwargs
        }

    def __init__(self, trial, alpha, **kwargs):
        super().__init__(trial, **kwargs)
        self.params = self.define_params(alpha, **self.params)
        self.model = XGBRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)