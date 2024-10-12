# Solar forecast

## Feat

| Features                                    | Desc                              | Range  |
|--------------------------------------------|-------------------------------------------|---------------|
| SolarDownwardRadiation_Mean                | Durchschnittliche solare Abwärtsstrahlung | [0, ∞)        |
| SolarDownwardRadiation_ncep_Mean           | Durchschnittliche solare Abwärtsstrahlung (NCEP) | [0, ∞)        |
| SolarDownwardRadiation_dwd_Mean            | Durchschnittliche solare Abwärtsstrahlung (DWD) | [0, ∞)        |
| SolarDownwardRadiation_RW_Mean_1h          | Durchschnittliche solare Abwärtsstrahlung (1h) | [0, ∞)        |
| SolarDownwardRadiation_RW_dwd_Mean_30min   | Durchschnittliche solare Abwärtsstrahlung (DWD, 30min) | [0, ∞)        |
| SolarDownwardRadiation_RW_dwd_Mean_1h      | Durchschnittliche solare Abwärtsstrahlung (DWD, 1h) | [0, ∞)        |
| SolarDownwardRadiation_dwd_Mean_Lag_30min  | Durchschnittliche solare Abwärtsstrahlung (DWD, Lag 30min) | [0, ∞)        |
| SolarDownwardRadiation_Mean_Lag_1h         | Durchschnittliche solare Abwärtsstrahlung (Lag 1h) | [0, ∞)        |
| SolarDownwardRadiation_Mean_Lag_24h        | Durchschnittliche solare Abwärtsstrahlung (Lag 24h) | [0, ∞)        |
| Panel_Efficiency_Mean                      | Durchschnittliche Effizienz des Panels     | [0, 1]        |
| Panel_Efficiency_std                       | Standardabweichung der Effizienz des Panels | [0, 1]        |
| Panel_Temperature_Mean                     | Durchschnittliche Temperatur des Panels    | [-∞, ∞)       |
| Panel_Temperature_std                      | Standardabweichung der Temperatur des Panels | [0, ∞)        |
| Temperature_dwd_std                        | Standardabweichung der Temperatur (DWD)    | [0, ∞)        |
| Temperature_ncep_std                       | Standardabweichung der Temperatur (NCEP)   | [0, ∞)        |
| Temperature_dwd_Mean                       | Durchschnittliche Temperatur (DWD)         | [-∞, ∞)       |
| sin_hour                                   | Sinus des Stundenwerts                     | [-1, 1]       |
| cos_hour                                   | Kosinus des Stundenwerts                   | [-1, 1]       |


## Models

| Model                                      | Getestet | Loss |
|---------------------------------------------|----------|------|
| GARCH+ARIMA                                 |hat bei mir nix gerissen|   -   |
| GradientBoostingRegressor (sklearn)         | gerade dabei|       |
| XGBoost / LightGBM                          |          |      |
| Prophet                                     |          |      |
| Random Forest                               |          |      |
| Kalman Filter / Bayesian State Space Model / Exponential Smoothing State Space Model (ETS) |          |      |
| MLP / RNN / LSTM / TFT                         |          |      |

