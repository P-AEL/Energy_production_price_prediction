import os
import logging
import optuna
import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import StackingRegressor
import joblib
import papermill as pm

logging.basicConfig(level=logging.INFO)

# Set paths
BASE_PATH = os.getenv("BASE_PATH", "/Users/florian/Documents/github/DP2/Energy_production_price_prediction/") 
DATA_PATH = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/data/train_norm.csv")    
FILEPATH_STUDY = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/models/lgbr_stacking/logs")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/models/lgbr_stacking/models")

# Load data
data = pd.read_csv(DATA_PATH)
df = deepcopy(data)
df.dropna(inplace=True)

X = df.drop(columns="Target_Capacity_MWP_%")
y = df["Target_Capacity_MWP_%"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def objective(trial, alpha):
    """
    Objective function for the Optuna optimization. Trains a pipeline with LGBMRegressor and ARDRegression.

    args:   trial: optuna.trial.Trial
            alpha: float, the quantile to be used in the loss function
    returns: float
    """
    lgbm_params = {
        "boosting_type": "gbdt",
        "objective": "quantile",
        "alpha": alpha,
        "force_col_wise": True,
        "verbose": -1,  
        "random_state": 0,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha" : trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda" : trial.suggest_float("reg_lambda", 0.0, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),           
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10)
    }

    ard_params = {
        "alpha_1": trial.suggest_float("alpha_1", 1e-6, 1e-1, log=True),
        "alpha_2": trial.suggest_float("alpha_2", 1e-6, 1e-1, log=True),
        "lambda_1": trial.suggest_float("lambda_1", 1e-6, 1e-1, log=True),
        "lambda_2": trial.suggest_float("lambda_2", 1e-6, 1e-1, log=True)
    }

    # Train LGBMRegressor
    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_train, y_train)

    # Get predictions from LGBMRegressor
    X_train_lgbm = lgbm_model.predict(X_train).reshape(-1, 1)
    X_test_lgbm = lgbm_model.predict(X_test).reshape(-1, 1)

    # Train ARDRegression on LGBMRegressor predictions
    ard_model = ARDRegression(**ard_params)
    ard_model.fit(X_train_lgbm, y_train)

    # Get predictions from ARDRegression
    y_pred = ard_model.predict(X_test_lgbm)
    loss = mean_pinball_loss(y_test, y_pred, alpha=alpha)

    return loss

if __name__ == "__main__":
    # Get the iteration number from an environment variable or default to 1
    iteration = int(os.getenv("ITERATION", 1))

    # Create iteration-specific directories
    iteration_logs_path = os.path.join(FILEPATH_STUDY, f"i{iteration}_logs")
    iteration_models_path = os.path.join(MODEL_SAVE_PATH, f"i{iteration}_models")
    os.makedirs(iteration_logs_path, exist_ok=True)
    os.makedirs(iteration_models_path, exist_ok=True)

    best_params = {}
    all_trials = []

    lgbm_param_keys = [
        "n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_split_gain",
        "min_child_weight", "subsample_freq"
    ]

    for alpha in alphas:
        # Optimize pipeline hyperparameters
        study = optuna.create_study(direction="minimize", study_name=f"study_pipeline_{alpha}")
        study.optimize(lambda trial: objective(trial, alpha), n_trials=20)

        trial = study.best_trial
        logging.info(f"Best trial for alpha {alpha}:")
        logging.info(f"  Value: {trial.value}")
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")

        # Train the best LGBMRegressor with the best hyperparameters
        lgbm_params = {k: v for k, v in trial.params.items() if k in lgbm_param_keys}
        lgbm_model = LGBMRegressor(
            boosting_type="gbdt",
            objective="quantile",
            alpha=alpha,
            random_state=0,
            force_col_wise=True,
            verbose=-1,
            **lgbm_params
        )
        lgbm_model.fit(X_train, y_train)

        # Train the best ARDRegression with the best hyperparameters
        ard_params = {k: v for k, v in trial.params.items() if k not in lgbm_param_keys}
        ard_model = ARDRegression(**ard_params)

        # Combine models using StackingRegressor
        stacking_model = StackingRegressor(
            estimators=[('lgbm', lgbm_model)],
            final_estimator=ard_model
        )
        stacking_model.fit(X_train, y_train)

        # Save the best hyperparameters for the current alpha
        trial.params["quantile"] = alpha
        trial.params["loss"] = trial.value
        best_params[alpha] = trial.params

        # Save the best models with iteration in the filename
        alpha_str = str(alpha).replace("0.", "q")
        stacking_filename = os.path.join(iteration_models_path, f"stacking_{alpha_str}.pkl")
        joblib.dump(stacking_model, stacking_filename)
        logging.info(f"Saved best StackingRegressor for alpha {alpha} to {stacking_filename}")

        # Save the trials dataframe for the current study
        trials_df = study.trials_dataframe()
        trials_df["quantile"] = alpha
        all_trials.append(trials_df)

    # Save the best hyperparameters for each quantile
    best_params_filename = os.path.join(iteration_logs_path, f"best_params.csv")
    best_params_df = pd.DataFrame(best_params).T
    best_params_df.to_csv(best_params_filename, index=False)

    # Combine all trials dataframes and save to a CSV file
    combined_trials_filename = os.path.join(iteration_logs_path, f"trials.csv")
    combined_trials_df = pd.concat(all_trials, ignore_index=True)
    combined_trials_df.to_csv(combined_trials_filename, index=False)

    # Run the evaluation notebook using papermill
    eval_notebook_path = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/eval/eval_solar.ipynb")
    output_notebook_path = os.path.join(iteration_logs_path, f"i{iteration}_eval.ipynb")
    pm.execute_notebook(eval_notebook_path, output_notebook_path, parameters=dict(BASE_PATH=BASE_PATH, model_name="lgbr_stacking", iter=iteration))