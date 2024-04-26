
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('C:/Users/nprit/Documents/Rising-Eagle-company-work/Map_data/waterjob2017to2023.csv')

print(df.head)

# %%
# Convert 'Call Date' to datetime format
df['Call Date'] = pd.to_datetime(df['Call Date'])

# Function to get season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Create 'Season' column
df['Season'] = df['Call Date'].dt.month.apply(get_season)

# Extract year from 'Call Date'
df['Year'] = df['Call Date'].dt.year

# Count total number of jobs for each season and each year
job_counts = df.groupby(['Year', 'Season']).size().reset_index(name='Job Count')

# Create DataFrame after counting jobs
job_counts_df = pd.pivot_table(job_counts, values='Job Count', index='Year', columns='Season', fill_value=0)


# Display the DataFrame after counting the jobs by season
print(job_counts_df)

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have already calculated the job_counts DataFrame as per the previous code

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot for each season
job_counts_df.plot(kind='bar', ax=ax, stacked=True)

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Number of Jobs')
ax.set_title('Total Number of Jobs by Season Over the Years')

# Displaying the plot
plt.grid(axis='y')
plt.legend(title='Season')
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have already calculated the job_counts_df DataFrame as per the previous code

# Calculate percentage change from one season to another
winter_to_spring_change = job_counts_df['Spring'] / job_counts_df['Winter'] * 100 - 100
spring_to_summer_change = job_counts_df['Summer'] / job_counts_df['Spring'] * 100 - 100
summer_to_fall_change = job_counts_df['Fall'] / job_counts_df['Summer'] * 100 - 100
fall_to_winter_change = job_counts_df['Winter'].shift(-1) / job_counts_df['Fall'] * 100 - 100

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Define the number of bars
num_years = len(job_counts_df.index)
index = range(num_years)

# Define bar width
bar_width = 0.2

# Plotting each seasonal change as a grouped bar
ax.bar(index, winter_to_spring_change, width=bar_width, label='Winter to Spring', align='center')
ax.bar([i + bar_width for i in index], spring_to_summer_change, width=bar_width, label='Spring to Summer', align='center')
ax.bar([i + 2 * bar_width for i in index], summer_to_fall_change, width=bar_width, label='Summer to Fall', align='center')
ax.bar([i + 3 * bar_width for i in index], fall_to_winter_change, width=bar_width, label='Fall to Winter', align='center')

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Percentage Change (%)')
ax.set_title('Percentage Change of Jobs from One Season to Another')
ax.set_xticks([i + 1.5 * bar_width for i in index])
ax.set_xticklabels(job_counts_df.index)
ax.legend()

# Displaying the plot
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# %%
import pandas as pd

# Assuming you have the original DataFrame named df

# Convert 'Call Date' to datetime format
df['Call Date'] = pd.to_datetime(df['Call Date'])

# Function to get season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Create 'Season' column
df['Season'] = df['Call Date'].dt.month.apply(get_season)

# Extract year from 'Call Date'
df['Year'] = df['Call Date'].dt.year

# Group by loss city, year, and season, and count the number of jobs
city_year_season_counts = df.groupby(['Loss City', 'Year', 'Season']).size().reset_index(name='Job Count')

# Display the DataFrame containing the total number of jobs for each city in each year and season
print(city_year_season_counts)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have the DataFrame city_year_season_counts

# Pivot the DataFrame for visualization
pivot_counts = city_year_season_counts.pivot_table(index='Loss City', columns=['Year', 'Season'], values='Job Count', aggfunc='sum', fill_value=0)

# Plotting the heatmap with custom line color
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_counts, cmap='YlGnBu', linewidths=0.5, linecolor='black')
plt.title('Count of Jobs by City, Year, and Season')
plt.xlabel('Year - Season')
plt.ylabel('City')
plt.show()

#%%
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Assuming you have the DataFrame city_year_season_counts

# Pivot the DataFrame for visualization
pivot_counts = city_year_season_counts.pivot_table(index='Year', columns='Season', values='Job Count', aggfunc='sum', fill_value=0)

# Prepare the time series data
time_series_data = pivot_counts.stack()

# Seasonal differencing to make the data stationary
seasonal_differenced_data = time_series_data.diff(4).dropna()

# Visualize the differenced time series data
plt.figure(figsize=(12, 6))
seasonal_differenced_data.plot()
plt.title('Seasonally Differenced Time Series Data')
plt.xlabel('Date')
plt.ylabel('Differenced Job Count')
plt.show()

# Perform Augmented Dickey-Fuller Test for stationarity on seasonally differenced data
adf_result = adfuller(seasonal_differenced_data)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:', adf_result[4])

# Train the SARIMA model if seasonally differenced data is stationary
if adf_result[1] < 0.05:
    # Train the SARIMA model
    model = SARIMAX(seasonal_differenced_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))  # Adjust the order as needed
    result = model.fit()
    
    # Evaluate the model
    print(result.summary())
    
    # Plot diagnostic plots
    result.plot_diagnostics(figsize=(12, 8))
    plt.show()
else:
    print('The seasonally differenced data is still not stationary. Try additional differencing or other transformations.')

#%%

from statsmodels.tsa.arima.model import ARIMA
# Prepare data for forecasting
winter_data = job_counts_df.loc[:, 'Winter'].values
spring_data = job_counts_df.loc[:, 'Spring'].values
summer_data = job_counts_df.loc[:, 'Summer'].values

# Fit ARIMA model for Winter
winter_model = ARIMA(winter_data, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
winter_forecast = winter_model.forecast(steps=len(winter_data))  # Forecast for the same length as the original data

# Fit ARIMA model for Spring
spring_model = ARIMA(spring_data, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
spring_forecast = spring_model.forecast(steps=len(spring_data))

# Fit ARIMA model for Summer
summer_model = ARIMA(summer_data, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
summer_forecast = summer_model.forecast(steps=len(summer_data))

print("Forecasted number of jobs for Winter 2024:", winter_forecast)
print("Forecasted number of jobs for Spring 2024:", spring_forecast)
print("Forecasted number of jobs for Summer 2024:", summer_forecast)

#Evaluate the model
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Function to calculate evaluation metrics and plot forecasts
def evaluate_model(forecast, actual, season):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    
    print(f"Evaluation metrics for {season}:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    plt.plot(actual, label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.title(f"Actual vs Forecast for {season}")
    plt.xlabel("Time")
    plt.ylabel("Number of Jobs")
    plt.legend()
    plt.show()

# Evaluate and plot forecasts for Winter
evaluate_model(winter_forecast, winter_data, 'Winter')

# Evaluate and plot forecasts for Spring
evaluate_model(spring_forecast, spring_data, 'Spring')

# Evaluate and plot forecasts for Summer
evaluate_model(summer_forecast, summer_data, 'Summer')

#Sarima Model
# %%
# Fit SARIMA model for Winter
winter_model = SARIMAX(job_counts_df['Winter'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

# Forecast jobs for Winter 2024
winter_forecast = winter_model.forecast(steps=1)

# Fit SARIMA model for Spring
spring_model = SARIMAX(job_counts_df['Spring'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

# Forecast jobs for Spring 2024
spring_forecast = spring_model.forecast(steps=1)

# Fit SARIMA model for Summer
summer_model = SARIMAX(job_counts_df['Summer'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

# Forecast jobs for Summer 2024
summer_forecast = summer_model.forecast(steps=1)

# Print forecasted number of jobs
print("Forecasted number of jobs for Winter 2024:", winter_forecast.values[0])
print("Forecasted number of jobs for Spring 2024:", spring_forecast.values[0])
print("Forecasted number of jobs for Summer 2024:", summer_forecast.values[0])