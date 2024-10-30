from sklearn.ensemble import HistGradientBoostingRegressor
from .basemodel import BaseModel

class HGBRModel(BaseModel):
    def define_params(self, alpha, **kwargs):
        return {
            "alpha": alpha,
            "loss": "quantile",
            "random_state": 0,
            **kwargs
        }

    def __init__(self, trial, alpha, **kwargs):
        super().__init__(trial, **kwargs)
        self.params = self.define_params(alpha, **self.params)
        self.model = HistGradientBoostingRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)