import os
import sys
import logging
import optuna
from optuna.trial import TrialState
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
import pandas as pd
from copy import deepcopy
import joblib
logging.basicConfig(level=logging.INFO)

# Set paths
BASE_PATH = os.getenv('BASE_PATH', "/Users/florian/Documents/github/DP2/Energy_production_price_prediction/") 
DATA_PATH = os.path.join(BASE_PATH, "HEFTcom24/data/features.csv")
FILEPATH_STUDY = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/models/hgbr_model/logs")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/models/hgbr_model/models")

# Load data
data = pd.read_csv(DATA_PATH)
df = deepcopy(data)
df = df.drop(columns=["Unnamed: 0"])   
df = df.dropna()
X = df.drop(columns=["Solar_MWh_credit", "valid_time"])
y = df["Solar_MWh_credit"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def objective(trial, alpha):
    """
    Objective function for the Optuna optimization. Trains a Gradient Boosting Regressor model with the given hyperparameters.

    args:   trial: optuna.trial.Trial
            alpha: float, the quantile to be used in the loss function
    returns: float
    """
    max_iter = trial.suggest_int("max_iter", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 40)
    l2_regularization = trial.suggest_float("l2_regularization", 0.0, 1.0)

    model = HistGradientBoostingClassifier(
        loss="quantile",
        alpha=alpha,
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
        tol=0.01
    )
    
    model.fit(X_train, y_train)
    
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
    # Get the iteration number from an environment variable or default to 1
    iteration = int(os.getenv("ITERATION", 1))

    best_params = {}
    all_trials = []

    for alpha in alphas:
        study = optuna.create_study(direction="minimize", study_name= f"study_{alpha}")
        study.optimize(lambda trial: objective(trial, alpha), n_trials= 20)

        trial = study.best_trial
        logging.info(f"Best trial for alpha {alpha}:")
        logging.info(f"  Value: {trial.value}")
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")

        best_params[alpha] = trial.params

        # Train the best model with the best hyperparameters
        best_model = HistGradientBoostingClassifier(
            loss="quantile",
            alpha=alpha,
            max_iter=trial.params["max_iter"],
            max_depth=trial.params["max_depth"],
            learning_rate=trial.params["learning_rate"],
            min_samples_leaf=trial.params["min_samples_leaf"],
            l2_regularization=trial.params["l2_regularization"],
            random_state=0,
            early_stopping=True,
            validation_fraction=0.1,
            tol=0.01
        )
        
        best_model.fit(X_train, y_train)

        # Save the best model with iteration in the filename
        model_filename = os.path.join(MODEL_SAVE_PATH, f"bm_q_{alpha}_iter_{iteration}.pkl")
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