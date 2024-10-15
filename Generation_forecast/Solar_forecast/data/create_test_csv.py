import pandas as pd
import numpy as np

# %%
def resample_and_interpolate(group):
    return group.reset_index(level=[1, 2]).resample('30T').asfreq().interpolate()

# %%
weather_df = pd.read_csv("/Users/florian/Documents/github/DP2/Energy_production_price_prediction/weather_data/DWD_ICON-EU.csv")
solar_total = pd.read_csv("/Users/florian/Documents/github/DP2/Energy_production_price_prediction/basic_files/solar_total_production.csv")
weather_df.sort_values(by='ref_datetime', inplace=True)
weather_df = weather_df.groupby(["valid_datetime","latitude","longitude"]).last().reset_index()
weather_df.reset_index(inplace=True)

# %%
weather_df.valid_datetime = pd.to_datetime(weather_df.valid_datetime)
weather_df = weather_df.set_index(["valid_datetime","latitude","longitude"])
df_resampled = weather_df.groupby(['latitude', 'longitude'], group_keys=False).apply(resample_and_interpolate)
df_resampled = df_resampled.reset_index()

# %%
solar_total.timestamp_utc = pd.to_datetime(solar_total.timestamp_utc)
df_resampled.drop(columns=['index','ref_datetime'], inplace=True)
df_resampled_merged = pd.merge(df_resampled, solar_total, how='left', left_on='valid_datetime', right_on='timestamp_utc')
df_resampled_merged_solar = df_resampled_merged.loc[~(df_resampled_merged.latitude == 53.935) & ~(df_resampled_merged.longitude == 1.8645)]

# %%
df_resampled_merged_solar1 = df_resampled_merged_solar.groupby("valid_datetime").mean().reset_index()

# %%
distinct_lat_lon_pairs = df_resampled_merged_solar[['latitude', 'longitude']].drop_duplicates()

# %%
df_resampled_merged_solar1.head()

# %%
def set_up_solar_features(df):
    df["cos_day"] = np.cos(2 * np.pi * df.valid_datetime.dt.dayofyear / 365)
    df["cos_hour"] = np.cos(2 * np.pi * df.valid_datetime.dt.hour / 24)
    df["SolarDownwardRadiation_Mean"] = df.SolarDownwardRadiation
    df["Temperature_dwd_Mean"] = df.Temperature
    df["Temperature_dwd_Std"] = df_resampled_merged_solar.groupby("valid_datetime").std().reset_index().Temperature
    df["SolarDownwardRadiation_RW_dwd_Mean_30min"] = df.SolarDownwardRadiation_Mean.rolling(window=1, min_periods=1).mean()
    df["SolarDownwardRadiation_RW_Mean_1h"] = df.SolarDownwardRadiation_Mean.rolling(window=2, min_periods=1).mean()
    df["SolarDownwardRadiation_dwd_Mean_Lag_30min"] = df.SolarDownwardRadiation_Mean.shift(1)
    df["SolarDownwardRadiation_Mean_Lag_1h"] = df.SolarDownwardRadiation_Mean.shift(2)
    df["SolarDownwardRadiation_Mean_Lag_24h"] = df.SolarDownwardRadiation_Mean.shift(48)
    for i in range(len(distinct_lat_lon_pairs)):
        lat = distinct_lat_lon_pairs.latitude.iloc[i]
        lon = distinct_lat_lon_pairs.longitude.iloc[i]
        mask = (df_resampled_merged_solar.latitude == lat) & (df_resampled_merged_solar.longitude == lon)
        df[f"Temperature_{i}"] = pd.Series(df_resampled_merged_solar.Temperature[mask].values)[:len(df)]  # Fill gaps with NaN
        df[f"SolarDownwardRadiation_{i}"] = pd.Series(df_resampled_merged_solar.SolarDownwardRadiation[mask].values)[:len(df)]  # Fill gaps with NaN
    return df
df_resampled_merged_solar2 = set_up_solar_features(df_resampled_merged_solar1)

# %%
def pv_temperature_efficiency(irradiance, ambient_temp, NOCT=45, wind_speed=1, eta_0=0.18, beta=0.004):
    Tc = ambient_temp + (NOCT - 20) * (irradiance / 800)
    efficiency = eta_0 * (1 - beta * (Tc - 25))
    
    return Tc, efficiency

# %%
for i in range(20):
    temp_col = f'Temperature_{i}'
    irradiance_col = f'SolarDownwardRadiation_{i}'
    panel_temp_col = f'Panel_Temperature_Point{i}'
    panel_eff_col = f'Panel_Efficiency_Point{i}'
    df_resampled_merged_solar2[panel_temp_col], df_resampled_merged_solar2[panel_eff_col] = pv_temperature_efficiency(df_resampled_merged_solar2[irradiance_col], df_resampled_merged_solar2[temp_col])

# %%
df_resampled_merged_solar2["Panel_Temperature_Mean"] = df_resampled_merged_solar2.filter(regex= r"Panel_Temperature.*").mean(axis= 1)
df_resampled_merged_solar2["Panel_Efficiency_Mean"] = df_resampled_merged_solar2.filter(regex= r"Panel_Efficiency.*").mean(axis= 1)
df_resampled_merged_solar2["Panel_Temperature_Std"] = df_resampled_merged_solar2.filter(regex= r"Panel_Temperature.*").std(axis= 1)
df_resampled_merged_solar2["Panel_Efficiency_Std"] = df_resampled_merged_solar2.filter(regex= r"Panel_Efficiency.*").std(axis= 1)

# %%
df_resampled_merged_solar2["Solar_MWh_Lag_48h"] = df_resampled_merged_solar2.generation_mw.shift(periods= 96) * 0.5
df_resampled_merged_solar2["Solar_MWh_credit"] = df_resampled_merged_solar2.generation_mw * 0.5
df_resampled_merged_solar2["Capacity_MWP_Lag_48h"] = df_resampled_merged_solar2.capacity_mwp.shift(periods= 96)

# %%
df_resampled_merged_solar3 = df_resampled_merged_solar2[[ 
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
    "Capacity_MWP_Lag_48h",
    "Solar_MWh_Lag_48h",
    "Solar_MWh_credit"
    ]]
df_resampled_merged_solar3.dropna(inplace=True)

# %%
df_resampled_merged_solar3.columns

# %%
df_resampled_merged_solar3.to_csv("/Users/florian/Documents/github/DP2/Energy_production_price_prediction/Generation_forecast/Solar_forecast/data/test.csv", index=False)


