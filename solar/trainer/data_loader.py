import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy

class DataLoader:
    def __init__(self, config_path):
        # Konfiguration aus YAML laden
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        # Datenpfade aus Konfiguration
        self.base_path = self.config["base_path"]
        self.data_path = os.path.join(self.base_path, self.config["data_path"])
        self.api_data_path = os.path.join(self.base_path, self.config["api_data_path"])
        
        # Train-Test-Split-Parameter aus Konfiguration
        self.train_test_split_config = self.config["train_test_split"]
        self.target_column = self.config["target_column"]

    def load_data(self):
        # Trainingsdatensatz laden
        data = pd.read_csv(self.data_path)
        df = deepcopy(data)
        
        # Feature- und Zielvariablen definieren
        X = df.drop(columns=self.target_column)
        y = df[self.target_column]

        if self.train_test_split_config["enabled"]:
            # Daten aufteilen in Training und Test, basierend auf den Split-Parametern
            X_train, X_test, y_train, y_test = train_test_split(
                X, 
                y, 
                test_size=self.train_test_split_config["ratio"], 
                shuffle=self.train_test_split_config["shuffle"],
                random_state=self.train_test_split_config["random_state"] 
            )
        else:
            # Wenn kein Split gewünscht ist, verwende die API-Daten als Testdatensatz
            api_data = pd.read_csv(self.api_data_path)
            X_train, y_train = X, y
            X_test = api_data.drop(columns=self.target_column)
            y_test = api_data[self.target_column]
        
        return X_train, X_test, y_train, y_test