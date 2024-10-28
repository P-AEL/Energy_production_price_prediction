# Solar forecast

## Feat

| Features                                    | Description                              | 
|--------------------------------------------|-------------------------------------------|
| SolarDownwardRadiation_Mean                | Mean solar downward radiation (NCEP, DWD)| 
| SolarDownwardRadiation_RW_Mean_1h          | Mean solar downward radiation (NCEP, DWD) 1h |
| SolarDownwardRadiation_RW_dwd_Mean_30min   | Mean solar downward radiation (DWD, 30min) |
| SolarDownwardRadiation_dwd_Mean_Lag_30min  | Mean solar downward radiation (DWD, Lag 30min) |
| SolarDownwardRadiation_Mean_Lag_1h         | Mean solar downward radiation (Lag 1h) | 
| SolarDownwardRadiation_Mean_Lag_24h        | Mean solar downward radiation (Lag 24h) |
| Panel_Efficiency_Mean                      | Mean panel efficiency     |
| Panel_Efficiency_Std                       | Standard deviation of panel efficiency |
| Panel_Temperature_Mean                     | Mean panel temperature    | 
| Panel_Temperature_Std                      | Standard deviation of panel temperature|
| Temperature_dwd_Std                        | Standard deviation of temperature (DWD)    |
| Temperature_dwd_Mean                       | Mean temperature (DWD)         |
| cos_hour                                   | Cosine of the hour value                   |
| cos_day                                    | Cosine of the day value                    |
| Capacity_MWP_Lag_48h                       | Capacity MWP (Lag 48h)                     |
| Target_Capacity_MWP_%_Lag_48h              | Solar MWh / Capacity  (Lag 48h)       |
| Target_Capactity_MWP_%                     | Solar MWh / Capacity                   |



## Models

- MPL Train: Mean Pinball Loss over all 9 Quantiles on train data
- MPL API: Mean Pinabll Loss over all 9 Quantiles on test api data of october

| Model                                      | MPL Train  | MPL API |
|--------------------------------------------|------|------|
| HistGradientBoostingRegressor              |  39.337    |  29.032    |
| XGBoost                                    |7.457 |  7.6    |
| LGBMRegressor                              |  9.217    |   6.989   |
| MLP                                        |      |      |
| LSTM                                       |      |      |
| RNN                                        |      |      |
