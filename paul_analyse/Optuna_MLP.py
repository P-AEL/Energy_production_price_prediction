import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the MLP model
class QuantileMLP(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_rates):
        super(QuantileMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        prev_size = input_size
        for size, dropout_rate in zip(layer_sizes, dropout_rates):
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.SiLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        # Final layer always has 9 outputs for the different quantiles
        self.layers.append(nn.Linear(prev_size, 9))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def pinball_loss(y_true, y_pred, quantiles):
    # Expand y_true to match y_pred shape
    y_true_expanded = y_true.repeat(1, len(quantiles))
    errors = y_true_expanded - y_pred
    
    # Calculate loss for each quantile
    quantiles_tensor = torch.tensor(quantiles, device=y_pred.device).reshape(1, -1)
    loss = torch.max((quantiles_tensor - 1) * errors, quantiles_tensor * errors)
    return torch.mean(loss)


# Training function
def train_model(model, train_loader, val_loader, optimizer, quantiles, device, epochs):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 15  # Early stopping patience
    min_relative_improvement = 0.0001  # 0.1% improvement threshold
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = pinball_loss(targets, outputs, quantiles)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += pinball_loss(targets, outputs, quantiles).item()
        val_loss /= len(val_loader)
        
        history.append(val_loss)
        
        # Check for early stopping and relative improvement
        if val_loss < best_val_loss:
            relative_improvement = (best_val_loss - val_loss) / best_val_loss
            if relative_improvement < min_relative_improvement and epoch > 10:
                print(f"Stopping due to small relative improvement: {relative_improvement:.6f}")
                break
            
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Optional: Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return best_val_loss, history

# Updated objective function to use the enhanced training function
def objective(trial, X, y, device, quantiles):
    # Data splitting
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Dataset and DataLoader
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    
    batch_size = trial.suggest_int('batch_size', 512, 1024, step=512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model hyperparameters
    n_layers = trial.suggest_int('n_layers', 3, 7)
    layer_sizes = [trial.suggest_int(f'layer_{i}_size', 64, 512, step=32) for i in range(n_layers)]
    dropout_rates = [trial.suggest_float(f'dropout_{i}', 0.1, 0.3) for i in range(n_layers)]
    
    # Create model
    input_size = X.shape[1]
    model = QuantileMLP(input_size, layer_sizes, dropout_rates).to(device)
    
    # Training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 150
    
    # Train and evaluate with enhanced stopping criteria
    best_val_loss, history = train_model(model, train_loader, val_loader, optimizer, quantiles, device, epochs)
    
    # Report intermediate values for Optuna to plot
    for epoch, loss in enumerate(history):
        trial.report(loss, epoch)
        
        # Handle pruning based on reported values
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_loss

# Main optimization function with pruning
def optimize_quantile_mlp(X, y, quantiles, n_trials=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1
    )
    
    study = optuna.create_study(
        direction='minimize',
        pruner=pruner
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y, device, quantiles),
        n_trials=n_trials
    )
    
    return study

# Example usage remains the same
if __name__ == "__main__":
    train_df = pd.read_csv("train.csv")
    X = train_df.drop(columns=["Solar_MWh_credit"]).values
    y = train_df["Solar_MWh_credit"].values
    
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    study = optimize_quantile_mlp(X, y, quantiles, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    best_params = study.best_trial.params
    n_layers = best_params['n_layers']
    layer_sizes = [best_params[f'layer_{i}_size'] for i in range(n_layers)]
    dropout_rates = [best_params[f'dropout_{i}'] for i in range(n_layers)]
    print(f"Layer sizes: {layer_sizes}")
    print(f"Dropout rates: {dropout_rates}")
    print(f"Number of layers: {n_layers}")