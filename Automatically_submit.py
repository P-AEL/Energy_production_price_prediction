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

    # Calculating the Generated power
    wind_df['WindPower'] = 0.5 * wind_df['AirDensity'] * rotor_area * wind_df['WindSpeed'] ** 3 * 174 / 1000000
    wind_df['UsableWindPower'] = np.minimum(wind_df['WindPower'], maximum_power_per_turbine * 174 * limiter / approximated_total_efficiency)
    # depending on the wind speed, the power output is limited to the maximum power output of the turbine or 0
    wind_df['PowerOutput'] = np.where((wind_df['WindSpeed'] >= minimum_wind_speed) & (wind_df['WindSpeed'] <= maximum_wind_speed_for_operation), wind_df['UsableWindPower'] * approximated_total_efficiency - const_internal_friction_coefficient, 0)

    # Same for 100m
    wind_df['WindPower:100'] = 0.5 * wind_df['AirDensity'] * rotor_area * wind_df['WindSpeed:100'] ** 3 * 174 / 1000000
    wind_df['UsableWindPower:100'] = np.minimum(wind_df['WindPower:100'], maximum_power_per_turbine * 174 * limiter / approximated_total_efficiency)
    wind_df['PowerOutput:100'] = np.where((wind_df['WindSpeed:100'] >= minimum_wind_speed) & (wind_df['WindSpeed:100'] <= maximum_wind_speed_for_operation), wind_df['UsableWindPower:100'] * approximated_total_efficiency - const_internal_friction_coefficient, 0)

    # Same for full
    wind_df['WindSpeed_full_avg'] = (wind_df['WindSpeed'] + wind_df['WindSpeed:100']) / 2
    wind_df['WindPower_full'] = 0.5 * wind_df['AirDensity'] * rotor_area * wind_df['WindSpeed:100'] ** 3 * 174 / 1000000
    wind_df['UsableWindPower_full'] = np.minimum(wind_df['WindPower_full'], maximum_power_per_turbine * 174 * limiter / approximated_total_efficiency)
    wind_df['PowerOutput_full'] = np.where((wind_df['WindSpeed:100'] >= minimum_wind_speed) & (wind_df['WindSpeed:100'] <= maximum_wind_speed_for_operation), wind_df['UsableWindPower_full'] * approximated_total_efficiency - const_internal_friction_coefficient, 0)

    # wind_df["Temperature_avg"] = (wind_df["Temperature"] + wind_df["Temperature:100"]) / 2
    # wind_df["RelativeHumidity_avg"] = (wind_df["RelativeHumidity"] + wind_df["RelativeHumidity:100"]) / 2
    wind_df["Temperature_avg"] = wind_df["Temperature"]
    wind_df["RelativeHumidity_avg"] = wind_df["RelativeHumidity"]   
    wind_df["WindSpeed:100_dwd_lag1"] = wind_df["WindSpeed:100"].shift(1)
    wind_df["WindSpeed:100_dwd_lag2"] = wind_df["WindSpeed:100"].shift(2)
    wind_df["WindSpeed:100_dwd_lag3"] = wind_df["WindSpeed:100"].shift(3)
    wind_df["UsableWindPower_opt"] = wind_df.UsableWindPower_full
    wind_df["WindSpeed:100_dwd"] = wind_df["WindSpeed:100"].shift(1)
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
    #features = ['Mean_SolarRadiation_dwd','cos_day','Solar_installedcapacity_mwp','cos_hour','Mean_Temperature_dwd','Mean_CloudCover_dwd']
    solar_df = solar_df.copy()  # Ensure we are working with a copy
    solar_total_production = pd.read_csv("basic_files/solar_total_production.csv")
    solar_df.sort_values(by='ref_datetime', inplace=True)
    solar_df = solar_df.groupby(["valid_datetime","latitude","longitude"]).last().reset_index()
    solar_df.reset_index(inplace=True)
    solar_df.valid_datetime = pd.to_datetime(solar_df.valid_datetime)
    solar_df = solar_df.set_index(["valid_datetime","latitude","longitude"])
    solar_df = solar_df.groupby(['latitude', 'longitude'], group_keys=False).apply(resample_and_interpolate)
    solar_df = solar_df.reset_index()
    solar_total_production.timestamp_utc = pd.to_datetime(solar_total_production.timestamp_utc)
    solar_df.drop(columns=['index','ref_datetime'], inplace=True)
    solar_df_merged = pd.merge(solar_df, solar_total_production, how='left', left_on='valid_datetime', right_on='timestamp_utc')
    solar_df_merged = solar_df_merged.groupby("valid_datetime").mean().reset_index()
    distinct_lat_lon_pairs = solar_df[['latitude', 'longitude']].drop_duplicates()

    # df.loc[:, 'ref_datetime'] = pd.to_datetime(df['ref_datetime'])
    solar_df_merged["hour"] = solar_df_merged.valid_datetime.dt.hour
    solar_df_merged["day_of_year"] = solar_df_merged.valid_datetime.dt.dayofyear
    solar_df_merged["cos_day_of_year"] = np.cos(2 * np.pi * solar_df_merged.day_of_year / 365)
    solar_df_merged["cos_hour"] = np.cos(2 * np.pi * solar_df_merged.hour / 24)
    solar_df_merged["Mean_SolarDownwardRadiation"] = solar_df_merged.SolarDownwardRadiation
    solar_df_merged["Mean_Temperature"] = solar_df_merged.Temperature
    solar_df_merged["Std_Temperature"] = solar_df.groupby("valid_datetime").std().reset_index().Temperature
    solar_df_merged["SolarDownwardRadiation_RW_Mean_30min"] = solar_df_merged.Mean_SolarDownwardRadiation.rolling(window=1, min_periods=1).mean()
    solar_df_merged["SolarDownwardRadiation_RW_Mean_1hour"] = solar_df_merged.Mean_SolarDownwardRadiation.rolling(window=2, min_periods=1).mean()
    solar_df_merged["SolarDownwardRadiation_dwd_Mean_Lag_30min"] = solar_df_merged.Mean_SolarDownwardRadiation.shift(1)
    solar_df_merged["SolarDownwardRadiation_dwd_Mean_Lag_1h"] = solar_df_merged.Mean_SolarDownwardRadiation.shift(2)
    solar_df_merged["SolarDownwardRadiation_dwd_Mean_Lag_24h"] = solar_df_merged.Mean_SolarDownwardRadiation.shift(48)
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
    
    solar_df_merged["Panel_Temperature_dwd_mean"] = solar_df_merged.filter(regex= r"Panel_Temperature.*").mean(axis= 1)
    solar_df_merged["Panel_Efficiency_dwd_mean"] = solar_df_merged.filter(regex= r"Panel_Efficiency.*").mean(axis= 1)
    solar_df_merged["Panel_Temperature_dwd_std"] = solar_df_merged.filter(regex= r"Panel_Temperature.*").std(axis= 1)
    solar_df_merged["Panel_Efficiency_dwd_std"] = solar_df_merged.filter(regex= r"Panel_Efficiency.*").std(axis= 1)
    solar_df_merged["solar_mw_lag_48h"] = solar_df_merged.generation_mw.shift(periods= 96)
    solar_df_merged["capacity_mwp_lag_48h"] = solar_df_merged.capacity_mwp.shift(periods= 96)
    solar_df_merged = pd.merge(solar_df_merged, submission_data, left_on='valid_datetime',right_on="datetime", how='inner')
    return solar_df_merged

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
            with open(f"{model_wind_stom}{i+1}.pkl", "rb") as f:
                model = load_pickle1(f)
            if not hasattr(model, '_preprocessor'):
                model._preprocessor = None
            df_to_predict = wind_df[["WindSpeed:100_dwd","Temperature_avg","RelativeHumidity_avg","AirDensity","UsableWindPower_opt","WindSpeed:100_dwd_lag1","WindSpeed:100_dwd_lag2","WindSpeed:100_dwd_lag3"]]
            residuals = model.predict(df_to_predict)
            predictions = wind_df.PowerOutput_full + residuals
            wind_df.loc[:, quantile] = predictions
            print(f"Predicted {predictions} ")
    else:
        quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        for quantile in quantiles:
            submission_data[quantile] = np.random.randint(300,1600,size=len(submission_data))

    if model_solar_strom is not None:
        #load xgboost model based on pickle path model_wind_stom
        quantiles_strom = [1,2,3,4,5,6,7,8,9]
        solar_df_to_predict = solar_df[[ 
            "Mean_SolarDownwardRadiation",
            "SolarDownwardRadiation_RW_Mean_1hour",
            "SolarDownwardRadiation_RW_Mean_30min",
            "SolarDownwardRadiation_dwd_Mean_Lag_30min",
            "SolarDownwardRadiation_dwd_Mean_Lag_1h",
            "SolarDownwardRadiation_dwd_Mean_Lag_24h",
            "Panel_Efficiency_dwd_mean",
            "Panel_Efficiency_dwd_std",
            "Panel_Temperature_dwd_mean",
            "Panel_Temperature_dwd_std",
            "Std_Temperature",
            "Mean_Temperature",
            "cos_hour",
            "cos_day_of_year",
            "capacity_mwp_lag_48h",
            "solar_mw_lag_48h"]]
        for i in quantiles_strom:
            with open(f"{model_solar_strom}{i}.pkl", "rb") as f:
                model_light = load_pickle1(f)
            predictions_solar = model_light.predict(solar_df_to_predict)
            solar_df[f"q{i}0"] = predictions_solar

        quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        for quantile in quantiles:
            submission_data[quantile] = np.random.randint(300,1600,size=len(submission_data))

    #join wind_df and solar_df on datetime and add the respective quantiles columns
    # submission_data = submission_data.groupby(["datetime"]).first().reset_index()
    quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
    for i,col in enumerate(quantiles):
        submission_data[col] = wind_df[col].reset_index(drop=True) + solar_df[col].reset_index(drop=True)
    
    #add market_bid
    if model_bid is not None:
        # code for quantile regression
        pass
    else:
        submission_data["market_bid"]= submission_data["q50"]
    
    submission_data = submission_data[["datetime","q10","q20","q30","q40","q50","q60","q70","q80","q90","market_bid"]]
    for col in submission_data.columns:
        if submission_data[col].dtype == 'float32':
            submission_data[col] = submission_data[col].astype(float)
    submission_data = comp_utils.prep_submission_in_json_format(submission_data)
    # #submit data
    rebase_api_client.submit(submission_data)
    print("Submitted data")

if __name__ == "__main__":
    Update(model_wind_stom="Generation_forecast/Wind_forecast/models/gbr_quantile_0.",model_solar_strom="Generation_forecast/Solar_forecast/models/lgbr_model/models/i4_models/lgbr_q",model_bid=None)