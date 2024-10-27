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
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor  # Dies ist nur für die Typisierung notwendig
import pickle
import lightgbm as lgb
from sklearn.exceptions import InconsistentVersionWarning
import json
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
#pd.set_option('display.max_columns', None)
import math


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)
def round_timedelta_to_days(td):
    return timedelta(days=math.ceil(td.total_seconds() / (24 * 3600)))

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(LSTMPredictor, self).__init__()
        
        # Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # Define the LSTM layer(s)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        
        # Fully connected layer to map LSTM output to the target size
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Cell state

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # We only need the output
        
        # Get the last output (many-to-one), out[:, -1, :] gives the last time step
        out = out[:, -1, :]
        
        # Pass the output through a fully connected layer
        out = self.fc(out)
        
        return out

def revenue(zb, DAP, Target_MW, imbalance_price):
    return zb * DAP + (Target_MW - zb) * (imbalance_price - 0.07 * (Target_MW - zb))

# Negative revenue function (for minimization)
def negative_revenue(zb, DAP, Target_MW, imbalance_price):
    return -revenue(zb, DAP, Target_MW, imbalance_price)

# Optimization function to compute the optimal bidding value for each row
def optimize_bidding(row):
    # Extract the values from the row
    DAP = row['predictions_day_ahead']
    Target_MW = row['1']
    imbalance_price = row['predictions_imbalance']
    
    # Initial guess for zb (midpoint between 0 and Target_MW)
    initial_zb = Target_MW / 2
    
    # Bounds for zb (as per KKT conditions)
    bounds = [(0, 1800)]
    
    # Perform the optimization
    result = minimize(negative_revenue, initial_zb, args=(DAP, Target_MW, imbalance_price), bounds=bounds)
    
    # Optimal trade value (zb)
    return result.x[0]

def get_predictions(model, X_tensor):
    input_size = 15  # Number of features
    hidden_size = 64              # Number of LSTM units
    num_layers = 3                 # Number of LSTM layers
    output_size = 1                # Always 9 for 9 quantiles
    dropout = 0.1  
    model_imbalance = LSTMPredictor(input_size, hidden_size, num_layers, output_size, dropout=dropout)
    model_imbalance.load_state_dict(torch.load(model))
    # Modell in den Evaluierungsmodus versetzen
    model_imbalance.eval()
    with torch.no_grad():
        predictions = model_imbalance(X_tensor)
    predictions = predictions.numpy()
    return predictions

def custom_pinball_loss(y_true, y_pred):
        y_true = y_true.get_label() if hasattr(y_true, 'get_label') else y_true
        delta = y_pred - y_true
        grad = np.where(delta >= 0, quantile, quantile -1)
        hess = np.ones_like(y_true)  # Hessian is 1 for pinball loss
        return grad, hess

def load_pickle1(path):
    return joblib.load(path)

def predict_wind_power(wind_df, model):   
    # Make predictions for all quantiles at once
    predictions = model.predict(wind_df)
    return predictions
    
def Set_up_features_wind(wind_df,submission_data):
    #features = ['wind_power_density_100_dwd','WindSpeed:100_dwd','wind_speed_model_diff','wind_direction_model_diff','hour','RelativeHumidity_dwd']
    wind_df = wind_df.copy()  # Ensure we are working with a copy
    wind_df.sort_values(by='ref_datetime', inplace=True)
    wind_df = wind_df.groupby(["valid_datetime","latitude","longitude"]).last().reset_index()
    wind_df.reset_index(inplace=True)
    wind_df.valid_datetime = pd.to_datetime(wind_df.valid_datetime)
    wind_df = wind_df.set_index(["valid_datetime","latitude","longitude"])
    wind_df = wind_df.groupby(['latitude', 'longitude'], group_keys=False).apply(resample_and_interpolate)
    wind_df = wind_df.reset_index()
    R_d = 287.05  # Specific gas constant for dry air (J/(kg·K))
    R_v = 461.5   # Specific gas constant for water vapor (J/(kg·K))
    p = 101325    # Standard atmospheric pressure in Pa
    # Calculate saturation vapor pressure (using temperature in Celsius), Tetens formula
    wind_df['Temperature_K'] = wind_df['Temperature'] + 273.15
    e_s = 0.61078 * np.exp((17.27 * (wind_df['Temperature'])) / (wind_df['Temperature'] +237.3))
    # in pa
    e_s = 1000 * e_s
    # Calculate actual vapor pressure
    e = wind_df['RelativeHumidity'] / 100 * e_s
    wind_df['AirDensity'] = (p - e) / (R_d * wind_df['Temperature_K']) + (e / (R_v * wind_df['Temperature_K']))
    # Turbine stats
    rotor_diameter = 154  # in meters
    approximated_total_efficiency = 0.348
    limiter = 0.94
    minimum_wind_speed = 3  # in m/s
    maximum_wind_speed_for_power_curve = 12.5  # in m/s
    maximum_wind_speed_for_operation = 25  # in m/s
    rotor_area = np.pi * (rotor_diameter / 2) ** 2  # in m²
    # turbine requires 3m/s to start rotating
    const_internal_friction_coefficient = 0.5 * 1.240 * np.pi * 77**2 * 3**3 * approximated_total_efficiency * 174 / 1000000
    maximum_power_per_turbine = 7 # in MW

    # Same for full
    wind_df.loc[:, 'WindSpeed_full_avg'] = (wind_df['WindSpeed'] + wind_df['WindSpeed:100']) / 2
    wind_df.loc[:, 'WindPower_full'] = 0.5 * wind_df['AirDensity'] * rotor_area * wind_df['WindSpeed:100'] ** 3 * 174 / 1000000
    wind_df.loc[:, 'UsableWindPower_full'] = np.minimum(wind_df['WindPower_full'], maximum_power_per_turbine * 174 * limiter / approximated_total_efficiency)
    wind_df['PowerOutput_full'] = np.where((wind_df['WindSpeed:100'] >= minimum_wind_speed) & (wind_df['WindSpeed:100'] <= maximum_wind_speed_for_operation), wind_df['UsableWindPower_full'] * approximated_total_efficiency - const_internal_friction_coefficient, 0)

    # wind_df["Temperature_avg"] = (wind_df["Temperature"] + wind_df["Temperature:100"]) / 2
    # wind_df["RelativeHumidity_avg"] = (wind_df["RelativeHumidity"] + wind_df["RelativeHumidity:100"]) / 2
    wind_df.loc[:, "Temperature_avg"] = wind_df["Temperature"]
    wind_df.loc[:, "RelativeHumidity_avg"] = wind_df["RelativeHumidity"]
    wind_df.loc[:, "WindSpeed:100_dwd_lag1"] = wind_df["WindSpeed:100"].shift(1)
    wind_df.loc[:, "WindSpeed:100_dwd_lag2"] = wind_df["WindSpeed:100"].shift(2)
    wind_df.loc[:, "WindSpeed:100_dwd_lag3"] = wind_df["WindSpeed:100"].shift(3)
    wind_df.loc[:, "UsableWindPower_opt"] = wind_df["UsableWindPower_full"]
    wind_df.loc[:, "WindSpeed:100_dwd"] = wind_df["WindSpeed:100"].shift(1)
    # print(wind_df)
    # print(pd.merge(wind_df, submission_data, left_on='valid_datetime',right_on="datetime", how='inner'))

    return pd.merge(wind_df, submission_data, left_on='valid_datetime',right_on="datetime", how='inner')

def resample_and_interpolate(group):
    return group.reset_index(level=[1, 2]).resample('30T').asfreq().interpolate()

def pv_temperature_efficiency(irradiance, ambient_temp, NOCT=45, wind_speed=1, eta_0=0.18, beta=0.004):
    # Calculate cell temperature using the simplified NOCT model
    Tc = ambient_temp + (NOCT - 20) * (irradiance / 800)
    
    # Calculate the efficiency loss due to increased cell temperature
    efficiency = eta_0 * (1 - beta * (Tc - 25))
    
    return Tc, efficiency

def Set_up_features_solar(solar_df,submission_data):
    #features = ['Mean_SolarRadiation_dwd','cos_day','Solar_installedcapacity_mwp','cos_hour','Temperature_dwd_Mean','Mean_CloudCover_dwd']
    solar_df = solar_df.copy()  # Ensure we are working with a copy
    solar_total_production = pd.read_csv("basic_files/solar_total_production.csv")
    solar_total_production.generation_mw = solar_total_production.generation_mw * 0.5 #fixed
    solar_df.sort_values(by='ref_datetime', inplace=True)
    solar_df = solar_df.groupby(["valid_datetime","latitude","longitude"]).last().reset_index()
    solar_df.reset_index(inplace=True)
    solar_df.valid_datetime = pd.to_datetime(solar_df.valid_datetime)
    solar_df = solar_df.set_index(["valid_datetime","latitude","longitude"])
    solar_df = solar_df.groupby(['latitude', 'longitude'], group_keys=False).apply(resample_and_interpolate)
    solar_df = solar_df.reset_index()
    solar_total_production.timestamp_utc = pd.to_datetime(solar_total_production.timestamp_utc)
    solar_df.drop(columns=['index','ref_datetime'], inplace=True)
    max_submitted_date = submission_data.datetime.max()
    max_total_production_date = solar_total_production.timestamp_utc.max()
    diff = max_submitted_date - max_total_production_date
    rounded_diff = round_timedelta_to_days(diff)
    time_to_delude = rounded_diff.days * 48
    solar_df_merged = pd.merge(solar_df, solar_total_production, how='left', left_on='valid_datetime', right_on='timestamp_utc')
    solar_df_merged = solar_df_merged.groupby("valid_datetime").mean().reset_index()
    distinct_lat_lon_pairs = solar_df[['latitude', 'longitude']].drop_duplicates()

    # Use .loc to safely modify DataFrame columns without warnings
    solar_df_merged.loc[:, "hour"] = solar_df_merged.valid_datetime.dt.hour
    solar_df_merged.loc[:, "day_of_year"] = solar_df_merged.valid_datetime.dt.dayofyear
    solar_df_merged.loc[:, "cos_day"] = np.cos(2 * np.pi * solar_df_merged["day_of_year"] / 365)
    solar_df_merged.loc[:, "cos_hour"] = np.cos(2 * np.pi * solar_df_merged["hour"] / 24)
    solar_df_merged.loc[:, "SolarDownwardRadiation_Mean"] = solar_df_merged["SolarDownwardRadiation"]
    solar_df_merged.loc[:, "Temperature_dwd_Mean"] = solar_df_merged["Temperature"]
    solar_df_merged["Temperature_dwd_Std"] = solar_df.groupby("valid_datetime").std().reset_index().Temperature
    # Rolling mean and lagged columns
    solar_df_merged.loc[:, "SolarDownwardRadiation_RW_dwd_Mean_30min"] = solar_df_merged["SolarDownwardRadiation_Mean"].rolling(window=1, min_periods=1).mean()
    solar_df_merged.loc[:, "SolarDownwardRadiation_RW_Mean_1h"] = solar_df_merged["SolarDownwardRadiation_Mean"].rolling(window=2, min_periods=1).mean()
    solar_df_merged.loc[:, "SolarDownwardRadiation_dwd_Mean_Lag_30min"] = solar_df_merged["SolarDownwardRadiation_Mean"].shift(1)
    solar_df_merged.loc[:, "SolarDownwardRadiation_Mean_Lag_1h"] = solar_df_merged["SolarDownwardRadiation_Mean"].shift(2)
    solar_df_merged.loc[:, "SolarDownwardRadiation_Mean_Lag_24h"] = solar_df_merged["SolarDownwardRadiation_Mean"].shift(48)
    for i in range(len(distinct_lat_lon_pairs)):
        lat = distinct_lat_lon_pairs.latitude.iloc[i]
        lon = distinct_lat_lon_pairs.longitude.iloc[i]
        mask = (solar_df.latitude == lat) & (solar_df.longitude == lon)
        solar_df_merged[f"Temperature_{i}"] = pd.Series(solar_df.Temperature[mask].values)[:len(solar_df_merged)]  # Fill gaps with NaN
        solar_df_merged[f"SolarDownwardRadiation_{i}"] = pd.Series(solar_df.SolarDownwardRadiation[mask].values)[:len(solar_df_merged)]  # Fill gaps with NaN
    for i in range(20):
        temp_col = f'Temperature_{i}'
        irradiance_col = f'SolarDownwardRadiation_{i}'
        panel_temp_col = f'Panel_Temperature_Point{i}'
        panel_eff_col = f'Panel_Efficiency_Point{i}'
        solar_df_merged[panel_temp_col], solar_df_merged[panel_eff_col] = pv_temperature_efficiency(solar_df_merged[irradiance_col], solar_df_merged[temp_col])
    

    solar_df_merged.loc[:, "Panel_Temperature_Mean"] = solar_df_merged.filter(regex=r"Panel_Temperature.*").mean(axis=1)
    solar_df_merged.loc[:, "Panel_Efficiency_Mean"] = solar_df_merged.filter(regex=r"Panel_Efficiency.*").mean(axis=1)
    solar_df_merged.loc[:, "Panel_Temperature_Std"] = solar_df_merged.filter(regex=r"Panel_Temperature.*").std(axis=1)
    solar_df_merged.loc[:, "Panel_Efficiency_Std"] = solar_df_merged.filter(regex=r"Panel_Efficiency.*").std(axis=1)
    solar_df_merged.loc[:, "Solar_MWh_Lag_48h"] = solar_df_merged["generation_mw"].shift(periods=time_to_delude)
    solar_df_merged.loc[:, "Capacity_MWP_Lag_48h"] = solar_df_merged["capacity_mwp"].shift(periods=time_to_delude)
    solar_df_merged.loc[:, "Target_Capacity_MWP%"] = solar_df_merged["generation_mw"] / solar_df_merged["capacity_mwp"]
    solar_df_merged.loc[:, "Target_Capacity_MWP_%_Lag_48"] = solar_df_merged["Target_Capacity_MWP%"].shift(periods=time_to_delude)
    
    solar_df_merged = pd.merge(solar_df_merged, submission_data, left_on='valid_datetime',right_on="datetime", how='inner')
    return solar_df_merged

def Set_up_features_bid(submission_data):
    df_imbalance_price = pd.read_csv("basic_files/imbalance_price.csv")
    df_day_ahead_price = pd.read_csv("basic_files/day_ahead_price.csv")
    df_market_price = pd.read_csv("basic_files/market_index.csv")
    df_day_ahead_price.timestamp_utc = pd.to_datetime(df_day_ahead_price.timestamp_utc)
    df_market_price.timestamp_utc = pd.to_datetime(df_market_price.timestamp_utc)
    df_imbalance_price.timestamp_utc = pd.to_datetime(df_imbalance_price.timestamp_utc)
    min_date = submission_data.datetime.min() - timedelta(minutes=30)
    datetimes = pd.date_range(end=min_date, periods=336, freq='30min')
    df_half_hourly = pd.DataFrame({"datetime": datetimes})
    df_half_hourly["datetime"] = pd.to_datetime(df_half_hourly["datetime"])
    df_submission_combined = pd.merge(df_half_hourly, submission_data, left_on='datetime', right_on='datetime', how='outer')
    df_submission_combined = pd.merge(df_submission_combined, df_day_ahead_price, left_on='datetime', right_on='timestamp_utc', how='left')
    df_submission_combined = pd.merge(df_submission_combined, df_imbalance_price, left_on='datetime', right_on='timestamp_utc', how='left')
    df_submission_combined = pd.merge(df_submission_combined, df_market_price, left_on='datetime', right_on='timestamp_utc', how='left')
    df_submission_combined["day_ahead_price"] = df_submission_combined["price_x"].rename("day_ahead_price")
    df_submission_combined["market_price"] = df_submission_combined["price_y"].rename("market_price")
    df_submission_combined["settlement_period"] = df_submission_combined["settlement_period_x"].rename("settlement_period")
    df_submission_combined["cos_hour"] = np.cos(2*np.pi*df_submission_combined["datetime"].dt.hour/24)
    df_submission_combined["cos_day"] = np.cos(2*np.pi*df_submission_combined["datetime"].dt.day/7)
    df_api_new_merged1 = df_submission_combined[["datetime","market_price","day_ahead_price","volume","settlement_period","cos_hour","cos_day","q10","q20","q30","q40","q50","q60","q70","q80","q90","imbalance_price"]].copy()
    df_api_new_merged1.loc[:,"market_price_lag96h"] = df_api_new_merged1["market_price"].shift(192)
    df_api_new_merged1.loc[:,"imbalance_price_lag96h"] = df_api_new_merged1["imbalance_price"].shift(192)
    df_api_new_merged1.loc[:,"day_ahead_price_lag1week"] = df_api_new_merged1["day_ahead_price"].shift(336)
    df_api_new_merged1.loc[:,"volume_lag96h"] = df_api_new_merged1["volume"].shift(192)
    df_api_new_merged1 = df_api_new_merged1.rename(columns={
    "q10": "1",
    "q20": "2"
    ,"q30": "3"
    ,"q40": "4"
    ,"q50": "5"
    ,"q60": "6"
    ,"q70": "7"
    ,"q80": "8"
    ,"q90": "9"
    })
    df_api_new_merged2 = df_api_new_merged1[["datetime","market_price_lag96h","imbalance_price_lag96h","day_ahead_price_lag1week","volume_lag96h",
                    "cos_hour","cos_day","1","2","3","4","5","6","7","8","9"]]
    df_api_new_merged2.dropna(inplace=True)
    scaler_path = "paul_analyse/LSTM_imbalance_scaler.pkl"
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    df_api_new_merged2_X = scaler.transform(df_api_new_merged2.drop(columns=["datetime"]))
    X_tensor = torch.tensor(df_api_new_merged2_X, dtype=torch.float32)
    X_tensor = X_tensor.unsqueeze(1)  # Adds a sequence length dimension
    model_imbalance = "paul_analyse/LSTM_imbalance_price.pth"
    predictions_imbalance = get_predictions(model_imbalance, X_tensor)
    model_day_ahead = "paul_analyse/LSTM_day_ahead_price.pth"
    predictions_day_ahead = get_predictions(model_day_ahead, X_tensor)
    df_api_new_merged2["predictions_imbalance"] = predictions_imbalance
    df_api_new_merged2["predictions_day_ahead"] = predictions_day_ahead
    df_api_new_merged2["market_bid"] = df_api_new_merged2.apply(optimize_bidding, axis=1)
    with open(f"paul_analyse/lightgbm_model.joblib", "rb") as f:
        model_bid_residual = load_pickle1(f)
    X_residual = df_api_new_merged2[["predictions_day_ahead","predictions_imbalance","market_bid","cos_hour","cos_day",'1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9']]
    residuals = model_bid_residual.predict(X_residual.values)
    df_api_new_merged2["market_bid"] = df_api_new_merged2["market_bid"] + residuals
    df_api_new_merged2 = df_api_new_merged2.rename(columns={
    "1": "q10",
    "2": "q20"
    ,"3": "q30"
    ,"4": "q40"
    ,"5": "q50"
    ,"6": "q60"
    ,"7": "q70"
    ,"8": "q80"
    ,"9": "q90"    })
    df_api_new_merged2 = df_api_new_merged2[["datetime","q10","q20","q30","q40","q50","q60","q70","q80","q90","market_bid"]].reset_index(drop=True)
    return df_api_new_merged2

def Update(model_wind_stom=None,model_solar_strom=None,model_bid=None):
    #create df with times
    api_key = open("team_key.txt").read()
    api_key_stripped = api_key.strip()
    rebase_api_client = comp_utils.RebaseAPI(api_key = api_key_stripped)
    submission_data=pd.DataFrame({"datetime":comp_utils.day_ahead_market_times()})
    #get all other dfs
    weather_data = pd.read_csv("weather_data/DWD_ICON-EU.csv")
    weather_data["valid_datetime"] = pd.to_datetime(weather_data["valid_datetime"])
    solar_df = weather_data.loc[~((weather_data.latitude==53.935) & (weather_data.longitude==1.8645))]
    wind_df = weather_data.loc[(weather_data.latitude==53.935) & (weather_data.longitude==1.8645)]
    solar_df = Set_up_features_solar(solar_df,submission_data)
    wind_df = Set_up_features_wind(wind_df,submission_data)

    #get wind power
    if model_wind_stom is not None:
        #load xgboost model based on pickle path model_wind_stom
        quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        for i,quantile in enumerate(quantiles):
            path = f"{model_wind_stom}{i+1}_res-True_calc-False.pkl"
            with open(f"{model_wind_stom}{i+1}_boa_v3_res-True_calc-False.pkl", "rb") as f:
                model = load_pickle1(f)
            # print(f"\nModell für Quantil {quantile}:")
            # print(f"Modelltyp: {type(model).__name__}")
            
            # if hasattr(model, 'feature_importances_'):
            #     feature_importances = model.feature_importances_
            #     if hasattr(model, 'feature_names_in_'):
            #         feature_names = model.feature_names_in_
            #     elif hasattr(model, 'feature_name_'):
            #         feature_names = model.feature_name_
            #     else:
            #         print("Warnung: Keine Feature-Namen gefunden.")
            #         continue
                
            #     sorted_features = sorted(zip(feature_names, feature_importances), 
            #                             key=lambda x: x[1], reverse=True)
                
            #     print("Feature-Reihenfolge:")
            #     for name, importance in sorted_features:
            #         print(f"{name}: {importance}")
            # else:
            #     print("Das Modell hat kein 'feature_importances_' Attribut.")
            #     if hasattr(model, 'feature_names_in_'):
            #         print("Feature-Namen:")
            #         for name in model.feature_names_in_:
            #             print(name)
            #     elif hasattr(model, 'feature_name_'):
            #         print("Feature-Namen:")
            #         for name in model.feature_name_:
            #             print(name)

            if not hasattr(model, '_preprocessor'):
                model._preprocessor = None
            df_to_predict = wind_df[['WindSpeed:100_dwd', 'Temperature_avg', 'RelativeHumidity_avg', 'AirDensity', 'WindSpeed:100_dwd_lag1', 'WindSpeed:100_dwd_lag2', 'WindSpeed:100_dwd_lag3','UsableWindPower_opt']]
            residuals = model.predict(df_to_predict)
            predictions = wind_df.PowerOutput_full / 2 + residuals
            wind_df.loc[:, quantile] = predictions
    else:
        quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        for quantile in quantiles:
            submission_data[quantile] = np.random.randint(300,1600,size=len(submission_data))

    if model_solar_strom is not None:
        quantiles_strom = [1,2,3,4,5,6,7,8,9]
        solar_df_to_predict = solar_df[[ 
            "SolarDownwardRadiation_Mean",
            "SolarDownwardRadiation_RW_Mean_1h",
            "SolarDownwardRadiation_RW_dwd_Mean_30min",
            "SolarDownwardRadiation_dwd_Mean_Lag_30min",
            "SolarDownwardRadiation_Mean_Lag_1h",
            "SolarDownwardRadiation_Mean_Lag_24h",
            "Panel_Efficiency_Mean",
            "Panel_Efficiency_Std",
            "Panel_Temperature_Mean",
            "Panel_Temperature_Std",
            "Temperature_dwd_Std",
            "Temperature_dwd_Mean",
            "cos_hour",
            "cos_day",
            "Solar_MWh_Lag_48h",
            "Capacity_MWP_Lag_48h",
            "Target_Capacity_MWP_%_Lag_48",
            ]]
        mean_to_multiply = solar_df.Capacity_MWP_Lag_48h.mean()
        for i in quantiles_strom:
            with open(f"{model_solar_strom}{i}.pkl", "rb") as f:
                model_light = load_pickle1(f)
            predictions_solar = model_light.predict(solar_df_to_predict)
            solar_df[f"q{i}0"] = predictions_solar*mean_to_multiply

    #sort
    wind_df = wind_df[["datetime","q10","q20","q30","q40","q50","q60","q70","q80","q90"]]
    for col in wind_df.columns:
        if wind_df[col].dtype == 'float32':
            wind_df[col] = wind_df[col].astype(float)
    wind_df[quantiles] = wind_df[quantiles].applymap(lambda x: max(x, 0))
    wind_df[quantiles] = wind_df[quantiles].apply(lambda row: sorted(row), axis=1, result_type='expand')

    solar_df = solar_df[["datetime","q10","q20","q30","q40","q50","q60","q70","q80","q90"]]
    for col in solar_df.columns:
        if solar_df[col].dtype == 'float32':
            solar_df[col] = solar_df[col].astype(float)
    solar_df[quantiles] = solar_df[quantiles].applymap(lambda x: max(x, 0))
    solar_df[quantiles] = solar_df[quantiles].apply(lambda row: sorted(row), axis=1, result_type='expand')

    #adding
    quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
    for i,col in enumerate(quantiles):
        submission_data[col] = wind_df[col].reset_index(drop=True) + solar_df[col].reset_index(drop=True)
    
    #add market_bid
    if model_bid is not None:
        # code for quantile regression
        submission_data=Set_up_features_bid(submission_data)
    else:
        submission_data["market_bid"]= submission_data["q50"]
    
    submission_data = submission_data[["datetime","q10","q20","q30","q40","q50","q60","q70","q80","q90","market_bid"]]

    print(submission_data)
    submission_data = comp_utils.prep_submission_in_json_format(submission_data)
    # #submit data
    rebase_api_client.submit(submission_data)
    print("Submitted data")

if __name__ == "__main__":
    Update(model_wind_stom="Generation_forecast/Wind_forecast/models/gbr_quantile_0.",model_solar_strom="Generation_forecast/Solar_forecast/models/lgbr_model/models/i8_models/lgbr_q",model_bid=("paul_analyse/LSTM_imbalance_price.pth","paul_analyse/LSTM_day_ahead_price.pth"))
