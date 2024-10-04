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
import requests

if os.path.exists('proxies.txt'):
    with open('proxies.txt', 'r') as file:
        proxiy = file.read()
    proxies = {'http': proxiy, 'https': proxiy}
else:
    print("No proxies.txt file found.")
    proxies = None
api_key = open("team_key.txt").read()
api_key_stripped = api_key.strip()
rebase_api_client = comp_utils.RebaseAPI(api_key=api_key_stripped, proxy=proxies)

different_values_to_update = ["DWD_ICON-EU","NCEP_GFS"]

def Update(model_to_update):
    """
    Fetches value_to_update from the Rebase API and appends them to a CSV file.
    The CSV file is created if it does not exist.
    :param min_date: The earliest date to fetch data for, in the format 'YYYY-MM-DD'
    :param csv_path: The path to the CSV file to read/write
    :return: None
    """
    csv_path = f"weather_data/{model_to_update}.csv"
    basic_variables = "Temperature, WindSpeed, WindSpeed:100, WindDirection:100, CloudCover, RelativeHumidity, PressureReducedMSL, SolarDownwardRadiation, TotalPrecipitation"
    url = "https://api.rebase.energy/weather/v2/query"
    lats = [52.4872562, 52.8776682, 52.1354277, 52.4880497, 51.9563696, 52.2499177, 52.6416477, 52.2700912, 52.1960768, 52.7082618, 52.4043468, 52.0679429, 52.024023, 52.7681276, 51.8750506, 52.5582373, 52.4478922, 52.5214863, 52.8776682, 52.0780721,53.77, 53.77, 53.77, 53.77, 53.77, 53.77,
            53.84, 53.84, 53.84, 53.84, 53.84, 53.84,
            53.9, 53.9, 53.9, 53.9, 53.9, 53.9,
            53.97, 53.97, 53.97, 53.97, 53.97, 53.97,
            54.03, 54.03, 54.03, 54.03, 54.03, 54.03,
            54.1, 54.1, 54.1, 54.1, 54.1, 54.1]
    lons = [0.4012455, 0.7906532, -0.2640343, -0.1267052, 0.6588173, 1.3894081, 1.3509559, 0.7082557, 0.1534462, 0.7302284, 1.0762977, 1.1751747, 0.2962684, 0.1699257, 0.9115028, 0.7137489, 0.1204872, 1.5706825, 1.1916542, -0.0113488,1.702, 1.767, 1.832, 1.897, 1.962, 2.027,
            1.702, 1.767, 1.832, 1.897, 1.962, 2.027,
            1.702, 1.767, 1.832, 1.897, 1.962, 2.027,
            1.702, 1.767, 1.832, 1.897, 1.962, 2.027,
            1.702, 1.767, 1.832, 1.897, 1.962, 2.027,
            1.702, 1.767, 1.832, 1.897, 1.962, 2.027]
    body= {'model': model_to_update, 'latitude': lats,
        'longitude': lons,
          'variables': basic_variables, 'output-format': 'json', 'forecast-horizon': "latest"}
    try:
        # Initialize an empty DataFrame
        if os.path.exists(csv_path):   
            df = pd.read_csv(csv_path)
            print(f"Loaded existing CSV with {len(df)} rows")
        else:
            df = pd.DataFrame(columns=["ref_datetime","valid_datetime","latitude","longitude"])
            print("Creating new CSV file")
        
        response = requests.post(url, json=body, headers={"Authorization": f"{api_key_stripped}"})
        for point in range(len(response.json())):
            new_df = pd.DataFrame(response.json()[point])
            new_df['latitude'] = lats[point]
            new_df['longitude'] = lons[point]
            df = pd.concat([df, new_df])
        df.drop_duplicates(inplace=True)
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error updating {model_to_update}: {e}")

if __name__ == "__main__":
    for model in different_values_to_update:
        Update(model)