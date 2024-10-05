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
[API energy docs](https://api.rebase.energy/challenges/redoc)

[API weather docs](https://api.rebase.energy/weather/docs/v2/#general-usage)

## Power generation prediction
## Hornsea 1
### Windfarm stats
174  Wind turbines with 7 MW generation capabilities each (Model: Siemens Gamesa SWT-7.0-154)

### Wind turbine stats:
- Turbine is able to yaw to always face into the wind direction
- rotor diameter: 154m
- rotor area: 18600m^2
- rotor blade count: 3
- direct drive gearbox
- synchronous permanent generator, 690V, 50Hz
- minimum required windspeed: 3m/s
- rated wind speed: 13m/s
- cut-out windspeed: 25m/s
- survival wind speed: 70m/s
- Power density 376.3W/m^2

### Power generation calculation using maths
Problems
- Turbine efficieny unclear
- Gearbox efficiency unknown
- Generator efficiency unknown
- Inverter and Transformer efficiency unknown
- Electrical losses unknown (cables etc)
- Wake losses unknown (loss from a turbine behind a turbine)
- frictional losses unknown

Idea:

lets just calculate the power generation using some industry standards and compare to the real data. Then try to calculate better values or some additional terms to implement in the power calculation process.

For calculating a first approximation for the power output per turbine $P_{\text{turbine}}$, we used the Power available in the wind $P_{\text{wind}}$:


$$
P_{\text{wind}} = \frac{1}{2} \rho A v^3
$$

Where:

- $P_{\text{wind}}$ = Power available in the wind (Watts)
- $\rho$ = Air density (kg/m³), typically **1.225 kg/m³** at sea level, but can be approximated more accurately by using Tetens formula
- $A$ = Rotor swept area (m²), calculated as $\pi r^2$, where $r$ is the rotor radius
- $v$ = Wind speed (m/s)

----
Approximating the air density using temperatur and relative humidity data

Tetens formula for Saturation Vapor Pressure (approximation)

$$
e_s\text{(kPa)} = 0.61078 \times \exp\left(\frac{17.27T}{T+237.3}\right)
$$
Where:

- $T$ = Temperature in degrees Celsius (°C)

The resulting $e_s$ is in **kilopascals (kPa)**. To convert it to **Pascals (Pa)**:

$$
e_s (\text{Pa}) = 1000 \times e_s (\text{kPa})
$$

Calculate Actual Vapor Pressure ($e$)

The **actual vapor pressure** is calculated by considering the **relative humidity** ($RH$):

$$
e = \left( \frac{RH}{100} \right) \times e_s
$$

Where:

- $RH$ = Relative humidity in percentage (0-100%)

#### Step 3: Calculate Air Density ($\rho$)

Finally, the **air density** ($\rho$) can be calculated by combining the contributions from dry air and water vapor using the **ideal gas law**:

$$
\rho = \frac{(p - e)}{R_d \cdot T_K} + \frac{e}{R_v \cdot T_K}
$$

Where:

- $p$ = Atmospheric pressure (Pa), typically **101325 Pa** at sea level
- $R_d$ = Specific gas constant for dry air ($287.05 \, J/(kg \cdot K)$)
- $R_v$ = Specific gas constant for water vapor ($461.5 \, J/(kg \cdot K)$)
- $T_K$ = Temperature in **Kelvin (K)**, calculated as:

$$
T_K = T_C + 273.15
$$

Where $T_C$ is the temperature in Celsius.

----

Back to previous calculated the power in the wind, the turbine can only capture a fraction of it:

$$
P_{\text{turbine}} = P_{\text{wind}} \times C_p \times \eta_{\text{generator}} \times \eta_{\text{conversion}}
$$

Where:

- $C_p$ = Power coefficient (typically **0.3-0.4**, considering the Betz limit and real-world conditions)
- $\eta_{\text{generator}}$ = Generator efficiency (typically **90-95%**)
- $\eta_{\text{conversion}}$ = Efficiency of inverters/converters (typically **95-98%**)

#### Example Power Calculation:

1. **Assumptions**:
   - Air density ($\rho$): 1.225 kg/m³
   - Rotor radius ($r$):77 m (using the Siemens Gamesa 7 MW turbine with a rotor diameter of 174 m)
   - Wind speed ($v$): 13 m/s (rated wind speed)
   - Power coefficient ($C_p$): 0.4 (realistic for modern turbines)
   - Generator efficiency: 95%
   - Power conversion efficiency: 98%

2. **Calculating Rotor Area**:

$$
A = \pi \times (77)^2 = 18,626 \, \text{m}^2
$$

3. **Power Available in the Wind**:

$$
P_{\text{wind}} = \frac{1}{2} \times 1.225 \times 18,626 \times 12^3 = 25.064 \times 10^6 \, \text{W} = 25.064 \, \text{MW}
$$

4. **Approximating the real efficiency of the Turbine**:

    From the statssheet we know that the rated power of the turbine at a windspeed of 13 m/s is 7MW. Based on that, we can approximate the total efiiciency

$$
C_p \times \eta_{\text{generator}} \times  \eta_{\text{conversion}} \times \cdots ≈\frac{7\text{MW}}{25.064\text{MW}} ≈ 0.28 
$$

So, at a wind speed of **13 m/s**, this turbine would run at approximately **28%** efficiency, accounting for turbine, generator, conversion, and other efficiencies.

#### Summary of Key Losses to Consider:

1. **Aerodynamic efficiency** (power coefficient $C_p$): Typically **30-40%**.
2. **Generator efficiency**: Around **90-95%**.
3. **Power conversion losses**: Around **2-5%** (inverter/converter and transformer losses).
4. **Wake effect losses**: Between **5-20%** depending on the farm layout.
5. **Transmission losses**: Typically **1-2%**.
6. **Mechanical losses**: Around **1-2%** (friction, bearings, etc.).

Using these numbers, the following windspeed-generation plot can be created based on an average air density of 1.225kg/m³
​
![image info](..\Generation_forecast\Wind_forecast\images\first_power_curve_approximation_plot.png)

Its easy to see that max power is achieved earlier at approximately 11m/s. Based on that, a new approximation for the system efficiency is calculated.

$$
P_{\text{wind}} = \frac{1}{2} \times 1.225 \times 18,626 \times 11^3 = 25.064 \times 10^6 \, \text{W} = 15.184 \, \text{MW}
$$

$$
C_p \times \eta_{\text{generator}} \times  \eta_{\text{conversion}} \times \cdots ≈\frac{7\text{MW}}{15.184\text{MW}} ≈ 0.46 
$$

Although this efficiency number is very high and at an high end of the industry efficiency standards, we will use it because it describes our data the best

![image info](..\Generation_forecast\Wind_forecast\images\second_power_curve_approximation_plot.png)

This curve fits the real data much better than the previous one but it is still easily visible that it requires some refinement.

Idea:

Combine this baseline approximation of the generated power with a machine learning model for the quantiles based on additional features

### Power generation forecast using machine learning models


