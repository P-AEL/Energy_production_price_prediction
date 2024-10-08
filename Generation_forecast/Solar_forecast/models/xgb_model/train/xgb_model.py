import os 
import logging
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import joblib
from copy import deepcopy
logging.basicConfig(level=logging.INFO)


# Set paths
BASE_PATH = os.getenv('BASE_PATH', "/Users/florian/Documents/github/DP2/Energy_production_price_prediction/")
DATA_PATH = os.path.join(BASE_PATH, "HEFTcom24/data/features.csv")
FILEPATH_STUDY = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/models/xgb_model/logs")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/models/xgb_model/models")

# Load data
data = pd.read_csv(DATA_PATH)
df = deepcopy(data)
df = df.drop(columns=["Unnamed: 0"])
df = df.dropna()
X = df.drop(columns=["Solar_MWh_credit", "valid_time"])
y = df["Solar_MWh_credit"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["valid_time", "Solar_MWh_credit"]), df["Solar_MWh_credit"], test_size=0.2, random_state=42, shuffle= False)
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# Custom pinball loss function
def custom_pinball_loss(y_true, y_pred):
    """
    Custom pinball loss function for XGBoost. The loss function is defined as:
    L(y_true, y_pred) = alpha * (y_true - y_pred) if y_true >= y_pred else (alpha - 1) * (y_true - y_pred)

    args:   y_true: np.array, the true target values
            y_pred: np.array, the predicted target values
    returns: np.array, the loss value for each sample
    """
    y_true = y_true.get_label() if hasattr(y_true, 'get_label') else y_true
    delta = y_pred - y_true
    grad = np.where(delta >= 0, alpha, alpha-1)
    hess = np.ones_like(y_true)  # Hessian is 1 for pinball loss
    return grad, hess

# Optuna objective function
def objective(trial, alpha):
    """
    Objective function for the Optuna optimization. Trains a XGBoost Regressor model with the given hyperparameters.

    args:   trial: optuna.trial.Trial
            alpha: float, the quantile to be used in the loss function        
    returns: float
    """
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 9)
    subsample = trial.suggest_float("subsample", 0.2, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 1e1)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 1e1)
    
    # Train xgb model
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=0,
        tree_method="hist",
        objective= custom_pinball_loss
        #eval_metric=mean_pinball_loss,
        #early_stopping_rounds=10
    )

    model.fit(X_train, y_train, verbose= False) # eval_set=[(X_test, y_test)] verbose= False)
    y_pred = model.predict(X_test)
    loss = mean_pinball_loss(y_test, y_pred, alpha=alpha)
    
    # Log the step, loss and alpha
    logging.info(f"Trial {trial.number} - Alpha: {alpha}, Loss: {loss}, Params: {trial.params}")

    # Report the loss for pruning
    trial.report(loss, 0)

    # Prune trial if needed
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return loss


if __name__ == "__main__":
    # Get the iteration number from env
    iteration = int(os.getenv("ITERATION", 1))

    best_params = {}
    all_trials = []

    for alpha in alphas:
        study = optuna.create_study(direction="minimize", study_name= f"study_{alpha}")
        study.optimize(lambda trial: objective(trial, alpha), n_trials= 20)

        trial = study.best_trial
        logging.info(f"Best trial for alpha {alpha}:")
        logging.info(f"  Value: {trial.value}")
        logging.info(f"  Params: ")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")

        best_params[alpha] = trial.params

        # Train the best model with the best hyperparameters
        best_model = XGBRegressor(
            n_estimators=trial.params["n_estimators"],
            learning_rate=trial.params["learning_rate"],
            max_depth=trial.params["max_depth"],
            subsample=trial.params["subsample"],
            colsample_bytree=trial.params["colsample_bytree"],
            reg_alpha=trial.params["reg_alpha"],
            reg_lambda=trial.params["reg_lambda"],
            random_state=42,
            tree_method="hist",
            objective=custom_pinball_loss,
        )

        best_model.fit(X_train, y_train)

        # Save the best model with iteration in the filename
        model_filename = os.path.join(MODEL_SAVE_PATH, f"xgb_q_{alpha}_iter_{iteration}.pkl")
        joblib.dump(best_model, model_filename)
        logging.info(f"Saved best model for alpha {alpha} to {model_filename}")

        # Save the trials dataframe for the current study
        trials_df = study.trials_dataframe()
        trials_df["alpha"] = alpha
        all_trials.append(trials_df)

    # Save the best hyperparameters for each quantile
    best_params_filename = os.path.join(FILEPATH_STUDY, f"b_params_iter_{iteration}.csv")
    best_params_df = pd.DataFrame(best_params).T
    best_params_df.to_csv(best_params_filename, index= False)

    # Combine all trials dataframes and save to a CSV file
    combined_trials_filename = os.path.join(FILEPATH_STUDY, f"trials_iter_{iteration}.csv")
    combined_trials_df = pd.concat(all_trials, ignore_index=True)
    combined_trials_df.to_csv(combined_trials_filename, index=False)

