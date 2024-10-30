from lightgbm import LGBMRegressor
from .basemodel import BaseModel

class LGBRModel(BaseModel):
    def define_params(self, alpha, **kwargs):
        return {
            "alpha": alpha,
            "objective": "quantile",
            "boosting_type": "gbdt",
            "force_col_wise": True,
            "random_state": 0,
            "verbose": -1,
            **kwargs
        }

    def __init__(self, trial, alpha, **kwargs):
        super().__init__(trial, **kwargs)
        self.params = self.define_params(alpha, **self.params) 
        self.model = LGBMRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)