# Docs

## Files

### Data Files
| Name                         | Type                                        | Description                                                                                                                                                                             |
|------------------------------|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dwd_icon_eu_hornsea_1*.nc     | Weather forecasts from DWD’s ICON-EU model in netCDF format. | Gridded weather forecast data surrounding the Hornsea 1 wind farm.                                                                                                                      |
| dwd_icon_eu_pes10*.nc         | Weather forecasts from DWD’s ICON-EU model in netCDF format. | Multiple weather forecast points spanning PES region 10 (East England).                                                                                                                 |
| dwd_icon_eu_demand*.nc        | Weather forecasts from DWD’s ICON-EU model in netCDF format. | Multiple weather forecast points spanning major population centres in GB (relevant for demand and price forecasting).                                                                    |
| ncep_gfs_hornsea_1*.nc        | Weather forecasts from NCEP’s GFS model in netCDF format.    | Gridded weather forecast data surrounding the Hornsea 1 wind farm.                                                                                                                      |
| ncep_gfs_pes10*.nc            | Weather forecasts from NCEP’s GFS model in netCDF format.    | Multiple weather forecast points spanning PES region 10 (East England).                                                                                                                 |
| ncep_gfs_demand*.nc           | Weather forecasts from NCEP’s GFS model in netCDF format.    | Multiple weather forecasts spanning major population centres in GB (relevant for demand and price forecasting).                                                                          |
| Energy_Data*.csv              | Energy market data in CSV format.                          | dtm: UTC timestamp corresponding to the beginning of each half-hour period MIP¹: Market Index Price, a volume weighted average of intraday trades, £/MWh DA_Price²: Day-ahead auction price, specifically the “Intermittent Market Reference Price”, £/MWh SS_Price³: Single System Price, £/MWh Solar_MW²: Solar generation in units of MW Solar_capacity_mwp²: Estimate of total installed PV capacity in PES region 10 including estimated performance degradation (MW-peak) Solar_installedcapacity_mwp²: Estimate of total installed PV capacity in PES region 10 (MW-peak) Wind_MW¹: Power production at Hornsea 1 wind farm in units of MW. boa_MWh¹: Net-volume of bid and offer acceptance volumes at Hornsea 1 in units of MWh |

¹ GFS: Global Forecast System
² PV: Photovoltaic
³ MW: Megawatt

### Python etc. Files

#### comp_utils.py
Functions of the RebaseAPI Class:
- get_variable()
- get_day_ahead_price()
- get_market_index()
- get_imbalance_price()
- get_wind_total_production()
- get_solar_total_production()
- get_solar_wind_forecast()
- get_day_ahead_demand_forecast()
- get_margin_forecast()
- query_weather_latest()
- query_weather_latest_grid()
- get_hornsea_dwd()
- get_hornsea_gfs()
- get_pes10_nwp()
- get_demand_nwp()
- submit()
- -----------
Other functions:
- weather_df_to_xr()
- day_ahead_market_times()
- prep_submission_in_json_format()

## API
[API docs](https://api.rebase.energy/challenges/redoc)


