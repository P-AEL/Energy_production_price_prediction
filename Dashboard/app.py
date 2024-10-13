import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px

def pinball_loss(probabilistic_forecast, real_value):
    """
    Calculate the pinball loss for a given probabilistic forecast and real value.
    
    Args:
        probabilistic_forecast (dict): A dictionary of quantiles and forecasted values.
                                       Example: {'10': 19, '20': 17, ..., '90': 1}
        real_value (float): The observed real value.
        
    Returns:
        float: The calculated pinball loss.
    """
    total_loss = 0
    
    # Iterate through each quantile and its corresponding forecast value
    for quantile_str, forecast_value in probabilistic_forecast.items():
        quantile = int(quantile_str) / 100  # Convert quantile to decimal (e.g., 10 -> 0.1)
        
        # Calculate pinball loss for the current quantile
        if real_value >= forecast_value:
            loss = quantile * (real_value - forecast_value)
        else:
            loss = (1 - quantile) * (forecast_value - real_value)
        
        # Accumulate the loss
        total_loss += loss
    
    return total_loss

current_dir = os.getcwd()

path_df = os.path.abspath(os.path.join(current_dir, '..', 'basic_files'))
df_total_solar = pd.read_csv(os.path.join(path_df, 'solar_total_production.csv'))
df_total_solar.generation_mw = df_total_solar.generation_mw *0.5
df_total_wind = pd.read_csv(os.path.join(path_df, 'wind_total_production.csv'))
df_total_wind.generation_mw = df_total_wind.generation_mw *0.5 - df_total_wind.boa
df_imbalance_price = pd.read_csv(os.path.join(path_df, 'imbalance_price.csv'))
df_day_ahead_price = pd.read_csv(os.path.join(path_df, 'day_ahead_price.csv'))

# Get the path to the 'logs' directory in the parent directory
path = os.path.abspath(os.path.join(current_dir, '..', 'logs'))
files = os.listdir(path)
txt_files = [file for file in files if file.endswith('.txt')]
data = []
for file in txt_files:
    with open(os.path.join(path, file), 'r') as f:
        try:
            json_data = json.load(f)
            data.append(json_data)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from file: {file}")
date_name = []
for i in range(len(data)):
    date_name.append(data[i]["prediction_date"])

# Set page title
st.set_page_config(page_title="My Streamlit Dashboard")

# Add a title
st.title("Energy Forecasting Dashboard")

# Add some text
st.write("Quantile forecast for energy production")


# Use enumerate to create index-value pairs
options = list(enumerate(date_name))

# Extract the indices and the date values into a new selectbox
selected_option = st.selectbox("Select a date", options, format_func=lambda x: x[1])
selected_index, selected_date = selected_option

submissions = []
for i in range(len(data[selected_index]["solution"]["submission"])):
    submissions.append(data[selected_index]["solution"]["submission"][i]["timestamp"])
# submissions.append("All")

options_submissions = list(enumerate(submissions))
selected_option_submissions = st.selectbox("Select a submission", options_submissions, format_func=lambda x: x[1]) 
selected_index_submissions, selected_date_submissions = selected_option_submissions

market_bid = data[selected_index]["solution"]["submission"][selected_index_submissions]["market_bid"]
probabilistic_forecast = data[selected_index]["solution"]["submission"][selected_index_submissions]["probabilistic_forecast"]
df = pd.DataFrame(list(probabilistic_forecast.items()), columns=['Quantile', 'Value'])
fig = px.bar(df, x='Quantile', y='Value', title='Probabilistic Forecast', labels={'Quantile': 'Quantile', 'Value': 'Value'})
st.plotly_chart(fig)

selected_date_submissions = pd.to_datetime(selected_date_submissions)
df_total_wind['timestamp_utc'] = pd.to_datetime(df_total_wind['timestamp_utc'])
df_total_solar['timestamp_utc'] = pd.to_datetime(df_total_solar['timestamp_utc'])
df_day_ahead_price['timestamp_utc'] = pd.to_datetime(df_day_ahead_price['timestamp_utc'])
df_imbalance_price['timestamp_utc'] = pd.to_datetime(df_imbalance_price['timestamp_utc'])

real_value = None
try:
    solar_production = df_total_solar.loc[df_total_solar['timestamp_utc'] == selected_date_submissions]
    st.write(f"solar_production: {solar_production['generation_mw'].values[0]}")
    wind_production = df_total_wind.loc[df_total_wind['timestamp_utc'] == selected_date_submissions]
    st.write(f"wind_production: {wind_production['generation_mw'].values[0]}")
    st.write(f"total_production: {wind_production['generation_mw'].values[0] + solar_production['generation_mw'].values[0]}")
    real_value = wind_production['generation_mw'].values[0] + solar_production['generation_mw'].values[0]
except:
    st.write("No data available for selected date")

# Calculate the pinball loss
if real_value is not None:
    loss = pinball_loss(probabilistic_forecast, real_value)
    st.write(f"Pinball loss: {loss}")

try:
    day_ahead_price = df_day_ahead_price.loc[df_day_ahead_price['timestamp_utc'] == selected_date_submissions]
    imbalance_price = df_imbalance_price.loc[df_imbalance_price['timestamp_utc'] == selected_date_submissions]
    Revenue = market_bid * day_ahead_price["price"].values[0] +(real_value - market_bid) * imbalance_price["imbalance_price"].values[0]
    st.write(f"Revenue: {Revenue}")
except:
    st.write("No data available for selected date")

st.write(f"market_bid: {market_bid}")

#plot for revenue
try:
    revenues = []
    for date in options_submissions:
        selected_index_submissions_rev, selected_date_submissions_rev = date
        market_bid = data[selected_index]["solution"]["submission"][selected_index_submissions_rev]["market_bid"]
        selected_date_submissions_rev_datetime = pd.to_datetime(selected_date_submissions_rev)
        solar_production = df_total_solar.loc[df_total_solar['timestamp_utc'] == selected_date_submissions_rev_datetime]
        wind_production = df_total_wind.loc[df_total_wind['timestamp_utc'] == selected_date_submissions_rev_datetime]
        real_value = wind_production['generation_mw'].values[0] + solar_production['generation_mw'].values[0]
        day_ahead_price = df_day_ahead_price.loc[df_day_ahead_price['timestamp_utc'] == selected_date_submissions_rev_datetime]
        imbalance_price = df_imbalance_price.loc[df_imbalance_price['timestamp_utc'] == selected_date_submissions_rev_datetime]
        Revenue = market_bid * day_ahead_price["price"].values[0] +(real_value - market_bid) * imbalance_price["imbalance_price"].values[0]
        revenues.append(Revenue)
    df_revenue = pd.DataFrame(revenues, columns=['Revenue'])
    fig_revenue = px.line(df_revenue,x=submissions, y='Revenue', title='Revenue_over_day')
    st.plotly_chart(fig_revenue)
    #day average
    st.write(f"Average Revenue: {df_revenue['Revenue'].mean()}")
    

except:
    st.write("No data available for selected date")

#plot for revenue using different bidding strategies
try:
    st.write("Revenue using different bidding strategies 40% qunatile")
    revenues = []
    for date in options_submissions:
        selected_index_submissions_rev, selected_date_submissions_rev = date
        market_bid = data[selected_index]["solution"]["submission"][selected_index_submissions_rev]["probabilistic_forecast"]["40"]
        selected_date_submissions_rev_datetime = pd.to_datetime(selected_date_submissions_rev)
        solar_production = df_total_solar.loc[df_total_solar['timestamp_utc'] == selected_date_submissions_rev_datetime]
        wind_production = df_total_wind.loc[df_total_wind['timestamp_utc'] == selected_date_submissions_rev_datetime]
        real_value = wind_production['generation_mw'].values[0] + solar_production['generation_mw'].values[0]
        day_ahead_price = df_day_ahead_price.loc[df_day_ahead_price['timestamp_utc'] == selected_date_submissions_rev_datetime]
        imbalance_price = df_imbalance_price.loc[df_imbalance_price['timestamp_utc'] == selected_date_submissions_rev_datetime]
        Revenue = market_bid * day_ahead_price["price"].values[0] +(real_value - market_bid) * imbalance_price["imbalance_price"].values[0]
        revenues.append(Revenue)
    df_revenue = pd.DataFrame(revenues, columns=['Revenue'])
    fig_revenue = px.line(df_revenue,x=submissions, y='Revenue', title='Revenue_over_day')
    st.plotly_chart(fig_revenue)
    #day average
    st.write(f"Average Revenue: {df_revenue['Revenue'].mean()}")

except:
    st.write("No data available for selected date")

#plot for revenue using different bidding strategies
try:
    st.write("Revenue using different bidding strategies 60% qunatile")
    revenues = []
    for date in options_submissions:
        selected_index_submissions_rev, selected_date_submissions_rev = date
        market_bid = data[selected_index]["solution"]["submission"][selected_index_submissions_rev]["probabilistic_forecast"]["60"]
        selected_date_submissions_rev_datetime = pd.to_datetime(selected_date_submissions_rev)
        solar_production = df_total_solar.loc[df_total_solar['timestamp_utc'] == selected_date_submissions_rev_datetime]
        wind_production = df_total_wind.loc[df_total_wind['timestamp_utc'] == selected_date_submissions_rev_datetime]
        real_value = wind_production['generation_mw'].values[0] + solar_production['generation_mw'].values[0]
        day_ahead_price = df_day_ahead_price.loc[df_day_ahead_price['timestamp_utc'] == selected_date_submissions_rev_datetime]
        imbalance_price = df_imbalance_price.loc[df_imbalance_price['timestamp_utc'] == selected_date_submissions_rev_datetime]
        Revenue = market_bid * day_ahead_price["price"].values[0] +(real_value - market_bid) * imbalance_price["imbalance_price"].values[0]
        revenues.append(Revenue)
    df_revenue = pd.DataFrame(revenues, columns=['Revenue'])
    fig_revenue = px.line(df_revenue,x=submissions, y='Revenue', title='Revenue_over_day')
    st.plotly_chart(fig_revenue)
    #day average
    st.write(f"Average Revenue: {df_revenue['Revenue'].mean()}")

except:
    st.write("No data available for selected date")

#plot for loss
try:
    losses = []
    for date in options_submissions:
        selected_index_submissions_rev, selected_date_submissions_rev = date
        market_bid = data[selected_index]["solution"]["submission"][selected_index_submissions_rev]["market_bid"]
        selected_date_submissions_rev_datetime = pd.to_datetime(selected_date_submissions_rev)
        solar_production = df_total_solar.loc[df_total_solar['timestamp_utc'] == selected_date_submissions_rev_datetime]
        wind_production = df_total_wind.loc[df_total_wind['timestamp_utc'] == selected_date_submissions_rev_datetime]
        real_value = wind_production['generation_mw'].values[0] + solar_production['generation_mw'].values[0]
        day_ahead_price = df_day_ahead_price.loc[df_day_ahead_price['timestamp_utc'] == selected_date_submissions_rev_datetime]
        imbalance_price = df_imbalance_price.loc[df_imbalance_price['timestamp_utc'] == selected_date_submissions_rev_datetime]
        Revenue = market_bid * day_ahead_price["price"].values[0] +(real_value - market_bid) * imbalance_price["imbalance_price"].values[0]
        loss = pinball_loss(probabilistic_forecast, real_value)
        losses.append(loss)
    df_loss = pd.DataFrame(losses, columns=['Loss'])
    fig_loss = px.line(df_loss,x=submissions, y='Loss', title='Pinnball_Loss_over_day')
    st.plotly_chart(fig_loss)
    #day average
    st.write(f"Average Loss: {df_loss['Loss'].mean()}")

except:
    st.write("No data available for selected date")

#plot for market_bid
try:
    market_bids = []
    for date in options_submissions:
        selected_index_submissions_rev, selected_date_submissions_rev = date
        market_bid = data[selected_index]["solution"]["submission"][selected_index_submissions_rev]["market_bid"]
        market_bids.append(market_bid)
    df_market_bid = pd.DataFrame(market_bids, columns=['Market_bid'])
    fig_market_bid = px.line(df_market_bid,x=submissions, y='Market_bid', title='Market_bid_over_day')
    st.plotly_chart(fig_market_bid)

except:
    st.write("No data available for selected date")

#plot for real_value
try:
    real_values = []
    for date in options_submissions:
        selected_index_submissions_rev, selected_date_submissions_rev = date
        selected_date_submissions_rev_datetime = pd.to_datetime(selected_date_submissions_rev)
        solar_production = df_total_solar.loc[df_total_solar['timestamp_utc'] == selected_date_submissions_rev_datetime]
        wind_production = df_total_wind.loc[df_total_wind['timestamp_utc'] == selected_date_submissions_rev_datetime]
        real_value = wind_production['generation_mw'].values[0] + solar_production['generation_mw'].values[0]
        real_values.append(real_value)
    df_real_value = pd.DataFrame(real_values, columns=['Real_value'])
    fig_real_value = px.line(df_real_value,x=submissions, y='Real_value', title='Real_total_production_over_day')
    st.plotly_chart(fig_real_value)

except:
    st.write("No data available for selected date")
