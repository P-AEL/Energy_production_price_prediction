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
from xgboost import XGBRegressor
# pd.set_option('display.max_columns', None)


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
    
def Set_up_features_wind(df):
    #features = ['wind_power_density_100_dwd','WindSpeed:100_dwd','wind_speed_model_diff','wind_direction_model_diff','hour','RelativeHumidity_dwd']
    df = df.copy()  # Ensure we are working with a copy
    df.loc[:, 'WindSpeed:100_dwd'] = df['WindSpeed:100']
    df.loc[:, 'RelativeHumidity_dwd'] = df['RelativeHumidity']
    df.loc[:, 'wind_power_density_100_dwd'] = 0.5 * 1.225 * df['WindSpeed:100_dwd']**3
    df.loc[:, 'ref_datetime'] = pd.to_datetime(df['ref_datetime'])
    df.loc[:, 'hour'] = df['datetime'].dt.hour
    return df

def Set_up_features_solar(df):
    #features = ['Mean_SolarRadiation_dwd','cos_day','Solar_installedcapacity_mwp','cos_hour','Mean_Temperature_dwd','Mean_CloudCover_dwd']
    df = df.copy()  # Ensure we are working with a copy
    df.loc[:, 'ref_datetime'] = pd.to_datetime(df['ref_datetime'])
    df.loc[:, "Mean_SolarRadiation_dwd"] = df["SolarDownwardRadiation"]
    df.loc[:, "Mean_Temperature_dwd"] = df["Temperature"]
    df.loc[:, "Mean_CloudCover_dwd"] = df["CloudCover"]
    df.loc[:, "Solar_installedcapacity_mwp"] = 2228.208777
    df.loc[:, 'hour'] = df['datetime'].dt.hour
    df.loc[:, 'day_of_year'] = df['datetime'].dt.dayofyear
    df.loc[:, 'cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df.loc[:, 'cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df = df.groupby(['datetime']).mean().reset_index()
    return df

def Update(model_wind_stom=None,model_solar_strom=None,model_bid=None):
    #create df with times
    rebase_api_client = comp_utils.RebaseAPI(api_key = open("A-Team_key.txt").read())
    submission_data=pd.DataFrame({"datetime":comp_utils.day_ahead_market_times()})
    #get all other dfs
    weather_data = pd.read_csv("weather_data/DWD_ICON-EU.csv")
    weather_data["valid_datetime"] = pd.to_datetime(weather_data["valid_datetime"])
    #get electricity production from model
    lat_lon_pairs = weather_data[['latitude', 'longitude']].drop_duplicates()
    temp_df = pd.DataFrame(
    [(ts, lat, lon) for ts in submission_data.datetime for _, lat, lon in lat_lon_pairs.itertuples()],
    columns=['datetime', 'latitude', 'longitude']
    )
    temp_df["datetime_hourly"] = temp_df["datetime"].dt.floor("H")
    weather_data = weather_data.sort_values('valid_datetime')
    #join in valid_date
    submission_data = temp_df.merge(
            weather_data, 
            left_on=['latitude', 'longitude', 'datetime_hourly'],
            right_on=['latitude', 'longitude', 'valid_datetime'], 
            how="left"
        )
    submission_data = submission_data.sort_values('ref_datetime')
    submission_data = submission_data.groupby(["latitude","longitude","datetime"]).last().reset_index()


    #take the already averaged wind data     53.935,1.8645
    wind_df = submission_data.loc[(submission_data.latitude==53.935) & (submission_data.longitude==1.8645)]
    wind_df = Set_up_features_wind(wind_df)
    solar_df = submission_data.loc[~((submission_data.latitude==53.935) & (submission_data.longitude==1.8645))]
    solar_df = Set_up_features_solar(solar_df)

    #get wind power
    if model_wind_stom is not None:
        #load xgboost model based on pickle path model_wind_stom
        quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        for quantile in quantiles:
            model = load_pickle1(model_wind_stom+quantile+".pkl")
            predictions = predict_wind_power(wind_df[['wind_power_density_100_dwd','WindSpeed:100_dwd','hour','RelativeHumidity_dwd']], model)
            wind_df.loc[:, quantile] = predictions
    else:
        quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        for quantile in quantiles:
            submission_data[quantile] = np.random.randint(300,1600,size=len(submission_data))

    if model_solar_strom is not None:
        #load xgboost model based on pickle path model_wind_stom
        quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        for quantile in quantiles:
            model = load_pickle1(model_solar_strom+quantile+".pkl")
            predictions = predict_wind_power(solar_df[['Mean_SolarRadiation_dwd','cos_day','Solar_installedcapacity_mwp','cos_hour','Mean_Temperature_dwd','Mean_CloudCover_dwd']], model)
            solar_df.loc[:, quantile] = predictions
    else:
        quantiles = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
        for quantile in quantiles:
            submission_data[quantile] = np.random.randint(300,1600,size=len(submission_data))

    #join wind_df and solar_df on datetime and add the respective quantiles columns
    submission_data = submission_data.groupby(["datetime"]).first().reset_index()

    for col in quantiles:
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
    Update(model_wind_stom="paul_analyse/xgboost_regressor_model_",model_solar_strom="paul_analyse/xgboost_regressor_model_solar_")