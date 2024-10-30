import os
import joblib
import yaml
import logging
import pandas as pd
import optuna
import papermill as pm
from trainer.objective import objective
from models.model_registry import get_model
from trainer.data_loader import DataLoader
logging.basicConfig(level=logging.INFO)

class TrainerManager:
    def __init__(self, config_path, iteration):
        # Lädt die zentrale `train_manager`-Konfiguration und die Modell-spezifische Konfiguration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.iteration = iteration
        self.config_path = os.path.dirname(config_path)
        print(self.config_path)

    def train_models(self):
        """Train models, save the best model, and log parameters."""
        data_loader = DataLoader(config_path=self.config["data_loader"])
        X_train, X_test, y_train, y_test = data_loader.load_data()
        
        best_params = {}
        all_trials = []

        # Iteriere über Modelle und Alphas
        for model_name in self.config["models"]:

            self.iteration_logs_path = os.path.join(self.config["base_path"], f"logs/{model_name}_i{self.iteration}_logs")
            self.iteration_models_path = os.path.join(self.config["base_path"], f"models/{model_name}/i{self.iteration}_models")

            # Check if iteration for this model already exists
            while os.path.exists(self.iteration_logs_path) or os.path.exists(self.iteration_models_path):
                self.iteration += 1  # Increase iteration number
                self.iteration_logs_path = os.path.join(self.config["base_path"], f"logs/{model_name}/i{self.iteration}_logs")
                self.iteration_models_path = os.path.join(self.config["base_path"], f"models/{model_name}/i{self.iteration}_models")

            os.makedirs(self.iteration_logs_path, exist_ok=True)
            os.makedirs(self.iteration_models_path, exist_ok=True)

            for alpha in self.config["alphas"]:
                # Optuna-Studie und Optimierung für jedes Modell und jeden Alpha
                study = optuna.create_study(direction="minimize", study_name=f"{model_name}_study_{alpha}")
                study.optimize(
                    lambda trial: objective(trial, model_name, X_train, X_test, y_train, y_test, alpha, self.config_path),
                    n_trials=self.config["n_trials"]
                )

                # Beste Ergebnisse und Parameter
                trial = study.best_trial
                logging.info(f"Best trial for alpha {alpha}:")
                logging.info(f"  Value: {trial.value}")
                logging.info("  Params: ")
                for key, value in trial.params.items():
                    logging.info(f"    {key}: {value}")

                # Initialisiere und trainiere das beste Modell
                model = get_model(model_name=model_name, trial=trial, alpha=alpha, config_base_path=self.config_path)
                model.train(X_train, y_train)
                
                trial.params["alpha_q"] = alpha
                trial.params["pinball_loss"] = trial.value       
                best_params[alpha] = trial.params

                # Speichere das Modell und die Parameter
                alpha_str = str(alpha).replace("0.", "q")
                model_filename = os.path.join(self.iteration_models_path, f"{model_name}_{alpha_str}.pkl")
                joblib.dump(model, model_filename)
                logging.info(f"Saved best model for alpha {alpha} to {model_filename}")

                # Speichere alle Trial-Daten
                trials_df = study.trials_dataframe()
                trials_df["alpha_q"] = alpha
                all_trials.append(trials_df)

        # Beste Parameter protokollieren
        best_params_df = pd.DataFrame(best_params).T
        best_params_df.to_csv(os.path.join(self.iteration_logs_path, "best_params.csv"), index=False)

        # Kombiniere alle Trials und speichere sie
        all_trials_df = pd.concat(all_trials, ignore_index=True)
        all_trials_df.to_csv(os.path.join(self.iteration_logs_path, "trials.csv"), index=False)

    def execute_eval_notebook(self, model_name):
        """Run evaluation notebook using Papermill."""
        eval_notebook_path = os.path.join(self.config["base_path"], "eval/eval_solar.ipynb")
        output_notebook_path = os.path.join(self.iteration_logs_path, f"i{self.iteration}_eval_{model_name}.ipynb")
        pm.execute_notebook(
            eval_notebook_path,
            output_notebook_path,
            parameters={
                "ITERATION": self.iteration,
                "model_name": model_name
            }
        )
        logging.info(f"Executed evaluation notebook and saved to {output_notebook_path}")

