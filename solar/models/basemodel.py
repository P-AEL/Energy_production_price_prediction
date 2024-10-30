import yaml
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, trial, config_path, **kwargs):
        """Initialize the model with parameters from Optuna trial and config file."""
        self.trial = trial
        self.params = self.load_params(config_path, **kwargs)

    def load_params(self, config_path, **kwargs):
        """Load hyperparameters from YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        params = {}
        for param_name, param_range in config["hyperparameters"].items():
            if isinstance(param_range, list) and len(param_range) == 2:
                # Apply suggestion based on parameter type (int or float range)
                if isinstance(param_range[0], int):
                    params[param_name] = self.trial.suggest_int(param_name, param_range[0], param_range[1])
                elif isinstance(param_range[0], float):
                    params[param_name] = self.trial.suggest_float(param_name, param_range[0], param_range[1])
            else:
                params[param_name] = param_range
        params.update(kwargs)  # Additional parameters
        return params

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass