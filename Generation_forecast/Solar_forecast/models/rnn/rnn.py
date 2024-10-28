import os
import logging
import optuna
import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

logging.basicConfig(level=logging.INFO)

# Set paths
BASE_PATH = os.getenv("BASE_PATH", "/Users/florian/Documents/github/DP2/Energy_production_price_prediction/")
DATA_PATH = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/data/train.csv")
FILEPATH_STUDY = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/models/lgbr_model/logs")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/models/lgbr_model/models")

# Load data
data = pd.read_csv(DATA_PATH)
df = deepcopy(data)

# Drop NaN values
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns="Solar_MWh_credit").values
y = df["Solar_MWh_credit"].values

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

kf = KFold(n_splits=5, shuffle=False, random_state=0)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.swish = nn.SiLU()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def pinball_loss(y_true, y_pred, quantiles):
    errors = y_true - y_pred
    quantiles = quantiles.view(1, -1).expand_as(y_pred)  # (batch_size, 9)
    loss = torch.max((quantiles - 1) * errors, quantiles * errors)
    return torch.mean(loss)

def create_dataloader(X, y, batch_size):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def objective(trial, alpha):
    input_size = X_train.shape[1]
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = RNN(input_size, hidden_size, num_layers, 1, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = lambda y_true, y_pred: pinball_loss(y_true, y_pred, torch.tensor(alpha).to(device))

    losses = []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 10
    min_relative_improvement = 1e-4

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        train_loader = create_dataloader(X_train_fold, y_train_fold, batch_size)
        val_loader = create_dataloader(X_val_fold, y_val_fold, batch_size)

        for epoch in range(50):  # Number of epochs can be adjusted
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                y_pred = model(X_batch)
                loss = criterion(y_batch, y_pred)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    val_loss = criterion(y_batch, y_pred)
                    val_losses.append(val_loss.item())

            mean_val_loss = sum(val_losses) / len(val_losses)
            losses.append(mean_val_loss)

            # Check for early stopping and relative improvement
            if mean_val_loss < best_val_loss:
                relative_improvement = (best_val_loss - mean_val_loss) / best_val_loss
                if relative_improvement < min_relative_improvement and epoch > 10:
                    print(f"Stopping due to small relative improvement: {relative_improvement:.6f}")
                    break

                best_val_loss = mean_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Optional: Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{50}, Train Loss: {loss.item():.6f}, Val Loss: {mean_val_loss:.6f}")

            # Pruning
            trial.report(mean_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return sum(losses) / len(losses)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_params = {}
    all_trials = []

    for alpha in alphas:
        study = optuna.create_study(direction="minimize", study_name=f"study_{alpha}")
        study.optimize(lambda trial: objective(trial, alpha), n_trials=20)

        trial = study.best_trial
        logging.info(f"Best trial for alpha {alpha}:")
        logging.info(f"  Value: {trial.value}")
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")

        # Train the final model with the best hyperparameters on the entire training data
        input_size = X_train.shape[1]
        hidden_size = trial.params["hidden_size"]
        num_layers = trial.params["num_layers"]
        dropout = trial.params["dropout"]
        batch_size = trial.params["batch_size"]
        learning_rate = trial.params["learning_rate"]

        model = RNN(input_size, hidden_size, num_layers, 1, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = lambda y_true, y_pred: pinball_loss(y_true, y_pred, torch.tensor(alpha).to(device))

        train_loader = create_dataloader(X_train, y_train, batch_size)

        for epoch in range(50):  # Number of epochs can be adjusted
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_batch, y_pred)
                loss.backward()
                optimizer.step()

        # Save the best model
        alpha_str = str(alpha).replace("0.", "q")
        model_filename = os.path.join(MODEL_SAVE_PATH, f"rnn_{alpha_str}.pt")
        torch.save(model.state_dict(), model_filename)
        logging.info(f"Saved best model for alpha {alpha} to {model_filename}")

        # Save the best hyperparameters for the current alpha
        trial.params["alpha"] = alpha
        trial.params["loss"] = trial.value
        best_params[alpha] = trial.params

        # Save the trials dataframe for the current study
        trials_df = study.trials_dataframe()
        trials_df["alpha"] = alpha
        all_trials.append(trials_df)

    # Save the best hyperparameters for each quantile
    best_params_filename = os.path.join(FILEPATH_STUDY, f"best_params.csv")
    best_params_df = pd.DataFrame(best_params).T
    best_params_df.to_csv(best_params_filename, index=False)
    logging.info(f"Saved best hyperparameters to {best_params_filename}")

    # Combine all trials dataframes and save to a CSV file
    combined_trials_filename = os.path.join(FILEPATH_STUDY, f"trials.csv")
    combined_trials_df = pd.concat(all_trials, ignore_index=True)
    combined_trials_df.to_csv(combined_trials_filename, index=False)
    logging.info(f"Saved all trials to {combined_trials_filename}")