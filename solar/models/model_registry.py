import os
from .hgbr_model import HGBRModel
from .lgbr_model import LGBRModel
from .xgbr_model import XGBRModel

MODEL_REGISTRY = {
    "hgbr": HGBRModel,
    "lgbr": LGBRModel,
    "xgbr": XGBRModel
}

def get_model(model_name, trial, alpha, config_base_path):
    config_path = os.path.join(config_base_path, f"{model_name}_config.yaml")
    model_class = MODEL_REGISTRY.get(model_name)
    if not model_class:
        raise ValueError(f"Unknown model: {model_name}")
    return model_class(trial=trial, alpha=alpha, config_path=config_path)   