# %%
import os
import logging
import optuna
import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

logging.basicConfig(level=logging.INFO)

# Set paths
BASE_PATH = os.getenv("BASE_PATH", "/Users/florian/Documents/github/DP2/Energy_production_price_prediction/")
DATA_PATH = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/data/train_norm.csv")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/models/rnn")
API_TEST_DATA_PATH = os.path.join(BASE_PATH, "Generation_forecast/Solar_forecast/data/test_norm.csv")

# Load train and test data
train_data = pd.read_csv(DATA_PATH)
test_data = pd.read_csv(API_TEST_DATA_PATH)
df_train = deepcopy(train_data)
df_test = deepcopy(test_data)

# Drop NaN values of train data
df_train.dropna(inplace=True)

# Separate features and target
X = df_train.drop(columns="Target_Capacity_MWP_%").values
y = df_train["Target_Capacity_MWP_%"].values

X_api = df_test.drop(columns="Target_Capacity_MWP_%").values
y_api = df_test["Target_Capacity_MWP_%"].values

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_api)

# Convert data to PyTorch tensors and create DataLoaders 
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_api, dtype=torch.float32).view(-1, 1)

# Create TensorDataset and DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

# Create TensorDataset and DataLoader for testing
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

input_size = X_train_scaled.shape[1]
output_size = 9
num_epochs = 100
quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32)
# Save the scaler
#joblib.dump(scaler, os.path.join(MODEL_SAVE_PATH, "scaler.pkl"))

# %%
# Define RNN model
class RNN_Model(nn.Module):
    """
    RNN model for quantile regression.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(RNN_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

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

# %%
# Define custom pinball loss function
def pinball_loss(y_true, y_pred, quantiles):
    """
    Compute the pinball loss for a batch of predictions.
    """
    errors = y_true - y_pred
    quantiles = quantiles.view(1, -1).expand_as(y_pred)  # (batch_size, 9)
    loss = torch.max((quantiles - 1) * errors, quantiles * errors)
    return torch.mean(loss)

# %%
def objective(trial):
    # Hyperparameters to optimize
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128)

    # Create DataLoader with the suggested batch size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = RNN_Model(input_size, hidden_size, num_layers, output_size, dropout)

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    patience = 10
    rel_improvement_threshold = 0.00000001
    previous_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Reshape input to add sequence length dimension
            X_batch = X_batch.unsqueeze(1)  # (batch_size, 1, input_size)
            # Forward pass
            y_pred = model(X_batch)  # (batch_size, 9) where 9 is the number of quantiles
            # Expand y_batch to match y_pred dimensions
            y_batch_expanded = y_batch.repeat(1, len(quantiles))  # (batch_size, 9)
            # Compute the loss
            loss = pinball_loss(y_batch_expanded, y_pred, quantiles)
            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Relative stopping
        rel_improvement = (previous_loss - epoch_loss) / previous_loss
        if rel_improvement < rel_improvement_threshold:
            break
        previous_loss = epoch_loss

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_loss

# %%
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    trial = study.best_trial
    logging.info(f"Best trial:")
    logging.info(f"  Value: {trial.value}")
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

    # Train the best model with the best hyperparameters
    best_hidden_size = trial.params["hidden_size"]
    best_num_layers = trial.params["num_layers"]
    best_dropout = trial.params["dropout"]
    best_learning_rate = trial.params["learning_rate"]
    best_batch_size = trial.params["batch_size"]

    # Create DataLoader with the best batch size
    train_loader = DataLoader(dataset=train_dataset, batch_size=best_batch_size, shuffle=False)

    # Instantiate the model with the best hyperparameters
    model = RNN_Model(input_size, best_hidden_size, best_num_layers, output_size, best_dropout)

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    patience = 10
    rel_improvement_threshold = 0.00000001
    previous_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Reshape input to add sequence length dimension
            X_batch = X_batch.unsqueeze(1)  # (batch_size, 1, input_size)

            # Forward pass
            y_pred = model(X_batch)  # (batch_size, 9) where 9 is the number of quantiles

            # Expand y_batch to match y_pred dimensions
            y_batch_expanded = y_batch.repeat(1, len(quantiles))  # (batch_size, 9)

            # Compute the loss
            loss = pinball_loss(y_batch_expanded, y_pred, quantiles)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Relative stopping
        rel_improvement = (previous_loss - epoch_loss) / previous_loss
        if rel_improvement < rel_improvement_threshold:
            break
        previous_loss = epoch_loss

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load the best model
    model.load_state_dict(best_model)

    # Save the best model
    joblib.dump(model, os.path.join(MODEL_SAVE_PATH, "rnn_model.pkl"))

    # Test the model
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Reshape input to add sequence length dimension
            X_batch = X_batch.unsqueeze(1)  # (batch_size, 1, input_size)

            # Forward pass
            y_pred = model(X_batch)  # (batch_size, 9)

            # Expand y_batch to match y_pred dimensions
            y_batch_expanded = y_batch.repeat(1, len(quantiles))  # (batch_size, 9)

            # Compute the loss
            loss = pinball_loss(y_batch_expanded, y_pred, quantiles)
            test_loss += loss.item()

        print(f'Test Loss: {test_loss/len(test_loader):.7f}')

# %%
#joblib.dump(model, os.path.join(MODEL_SAVE_PATH, "rnn_model.pkl"))
