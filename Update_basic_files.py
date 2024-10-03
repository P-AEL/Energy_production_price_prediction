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
    proxies = None
    print("No proxies.txt file found.")

min_date = "2024-05-01"
api_key = open("team_key.txt").read()
print("API key loaded. Length:", len(api_key))
rebase_api_client = comp_utils.RebaseAPI(api_key=open("team_key.txt").read(), proxy=proxies)

different_values_to_update = ["day_ahead_price", "imbalance_price", "market_index", "solar_total_production", "wind_total_production", "solar_wind_forecast"]

def Update(min_date, value_to_update):
    """
    Fetches value_to_update from the Rebase API and appends them to a CSV file.
    The CSV file is created if it does not exist.
    :param min_date: The earliest date to fetch data for, in the format 'YYYY-MM-DD'
    :param csv_path: The path to the CSV file to read/write
    :return: None
    """
    mapping_value_to_columns = {
        "day_ahead_price": ['timestamp_utc', 'settlement_date', 'settlement_period', 'price'],
        "imbalance_price": ['timestamp_utc', 'settlement_date', 'settlement_period', 'imbalance_price'],
        "market_index": ['timestamp_utc', 'settlement_date', 'settlement_period', 'data_provider', 'price', 'volume'],
        "solar_total_production": ['timestamp_utc', 'generation_mw', 'installed_capacity_mwp', 'capacity_mwp'],
        "wind_total_production": ['timestamp_utc', 'settlement_date', 'settlement_period', 'boa', 'generation_mw'],
        "solar_wind_forecast": ['timestamp_utc', 'settlement_date', 'settlement_period', 'solar_mw', 'wind_offshore_mw', 'wind_onshore_mw']
    }

    try:
        min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
        today = datetime.today().date()
        csv_path = f"{value_to_update}.csv"
        
        # Initialize an empty DataFrame
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                min_date = df['settlement_date'].max()
                # df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
                print(f"Loaded existing CSV with {len(df)} rows")
            except Exception as e:
                print(f"Error reading existing CSV: {e}")
                df = pd.DataFrame(columns=mapping_value_to_columns[value_to_update])
        else:
            df = pd.DataFrame(columns=mapping_value_to_columns[value_to_update])
            print("Creating new CSV file")
        
        # Generate dates to update
        print(f"min_date: {min_date}, today: {today}")
        dates_to_update = pd.date_range(start=min_date, end=today, freq='D')
        try:
            for date in dates_to_update:
                date_str = date.strftime('%Y-%m-%d')
                if value_to_update == "day_ahead_price":
                    prices = rebase_api_client.get_day_ahead_price(date_str)
                elif value_to_update == "imbalance_price":
                    prices = rebase_api_client.get_imbalance_price(date_str)
                elif value_to_update == "market_index":
                    prices = rebase_api_client.get_market_index(date_str)
                elif value_to_update == "solar_total_production":
                    prices = rebase_api_client.get_solar_total_production(date_str)
                elif value_to_update == "wind_total_production":
                    prices = rebase_api_client.get_wind_total_production(date_str)
                elif value_to_update == "solar_wind_forecast":
                    prices = rebase_api_client.get_solar_wind_forecast(date_str)
                if len(prices):
                    new_df = pd.DataFrame(prices)
                    # new_df['timestamp_utc'] = pd.to_datetime(new_df['timestamp_utc'])
                    df = pd.concat([df, new_df], ignore_index=True)
                else:
                    print(f"No data returned for {date_str}")
            df.drop_duplicates(inplace=True)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error fetching data for {date_str}: {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    for value_to_update in different_values_to_update:
        Update(min_date, value_to_update)