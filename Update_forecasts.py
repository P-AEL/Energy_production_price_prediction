import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle
import comp_utils
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from datetime import datetime, timedelta
import pickle as pkl

if os.path.exists('proxies.txt'):
    with open('proxies.txt', 'r') as file:
        proxiy = file.read()
    proxies = {'http': proxiy, 'https': proxiy}
else:
    print("No proxies.txt file found.")
    proxies = None


api_key = open("team_key.txt").read()
api_key_stripped = api_key.strip()
different_values_to_update = ["day_ahead_demand_forecast", "margin_forecast"]
rebase_api_client = comp_utils.RebaseAPI(api_key=api_key_stripped, proxy=proxies)

def Update(value_to_update):
    """
    Fetches value_to_update from the Rebase API and appends them to a CSV file.
    The CSV file is created if it does not exist.
    :param min_date: The earliest date to fetch data for, in the format 'YYYY-MM-DD'
    :param csv_path: The path to the CSV file to read/write
    :return: None
    """
    mapping_value_to_columns = {
        "day_ahead_demand_forecast": ['timestamp_utc', 'settlement_date', 'settlement_period', 'boundary',
                    'publish_time_utc', 'transmission_system_demand', 'national_demand'],
        "margin_forecast": ['forecast_date', 'publish_time_utc', 'margin']
    }
    try:
        csv_path = f"{value_to_update}.csv"
        
        # Initialize an empty DataFrame
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"Loaded existing CSV with {len(df)} rows")
            except Exception as e:
                print(f"Error reading existing CSV: {e}")
                df = pd.DataFrame(columns=mapping_value_to_columns[value_to_update])
        else:
            df = pd.DataFrame(columns=mapping_value_to_columns[value_to_update])
            print("Creating new CSV file")
        
        # Generate dates to update
        try:
       
            if value_to_update == "day_ahead_demand_forecast":
                prices = rebase_api_client.get_day_ahead_demand_forecast()
            elif value_to_update == "margin_forecast":
                prices = rebase_api_client.get_margin_forecast()
            if len(prices):
                new_df = pd.DataFrame(prices)
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                print(f"No data returned for ")
            df.drop_duplicates(inplace=True)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error fetching data for: {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    for value_to_update in different_values_to_update:
        Update(value_to_update)