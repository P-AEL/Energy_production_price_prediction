import optuna
from sklearn.metrics import mean_pinball_loss
from models.model_registry import get_model


def objective(trial, model_name, X_train, X_test, y_train, y_test, alpha, config_base_path):
    model = get_model(model_name=model_name, trial=trial, alpha=alpha, config_base_path=config_base_path)

    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    loss = mean_pinball_loss(y_test, y_pred, alpha=alpha)

    trial.report(loss, 0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return loss   