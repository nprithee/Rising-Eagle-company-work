

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load the dataset
#Waterjob dataset
df1 = pd.read_csv('C:/Users/nprit/Documents/Rising-Eagle-company-work/JobReportData/waterjobreport2017-winter2024.csv')
print(df1.head)

#Moldjob dataset
df2 = pd.read_csv('C:/Users/nprit/Documents/Rising-Eagle-company-work/JobReportData/Moldjob2017-winter2024.csv')
print(df2.head)

#Firejob dataset
df3 = pd.read_csv('C:/Users/nprit/Documents/Rising-Eagle-company-work/JobReportData/Firejob2017-Winter2024.csv')
print(df3.head)

# %%

# Define a function to map months to seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Define a function to format season for a given dataset
def format_season(df):
    # Convert 'Call Date' to datetime format
    df['Call Date'] = pd.to_datetime(df['Call Date'])
    # Extract month from 'Call Date'
    df['Month'] = df['Call Date'].dt.month
    # Map months to seasons
    df['Season'] = df['Month'].apply(get_season)
    return df

# Load and format Waterjob dataset
df1 = format_season(df1)

# Load and format Moldjob dataset
df2 = format_season(df2)

# Load and format Firejob dataset

df3 = format_season(df3)

# %%
# Count total number of jobs for each season and each year
waterjob_counts = df1.groupby(['Season', df1['Call Date'].dt.year]).size().reset_index(name='Job Count')
moldjob_counts = df2.groupby(['Season', df2['Call Date'].dt.year]).size().reset_index(name='Job Count')
firejob_counts = df3.groupby(['Season', df3['Call Date'].dt.year]).size().reset_index(name='Job Count')

# Create DataFrame after counting jobs
waterjob_counts_df = pd.DataFrame(waterjob_counts)
moldjob_counts_df = pd.DataFrame(moldjob_counts)
firejob_counts_df = pd.DataFrame(firejob_counts)

# Pivot the waterjob counts DataFrame
waterjob_pivot = waterjob_counts_df.pivot(index='Season', columns='Call Date', values='Job Count')

# Pivot the moldjob counts DataFrame
moldjob_pivot = moldjob_counts_df.pivot(index='Season', columns='Call Date', values='Job Count')

# Pivot the firejob counts DataFrame
firejob_pivot = firejob_counts_df.pivot(index='Season', columns='Call Date', values='Job Count')

# Display the pivoted DataFrames
print("Waterjob Counts")
print(waterjob_pivot)
print("\nMoldjob Counts")
print(moldjob_pivot)
print("\nFirejob Counts ")
print(firejob_pivot)

#%%

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you already have df1, df2, and df3 loaded

# Define a function to process and plot each dataset
def plot_seasonal_job_counts(df, title):
    # Convert 'Call Date' to datetime format
    df['Call Date'] = pd.to_datetime(df['Call Date'])

    # Extract year from 'Call Date'
    df['Year'] = df['Call Date'].dt.year

    # Count total number of jobs for each season and each year
    seasonal_job_counts = df.groupby(['Year', 'Season']).size().unstack(fill_value=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot for each season
    seasonal_job_counts.plot(kind='bar', ax=ax, stacked=True)

    # Adding labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Jobs')
    ax.set_title(title)

    # Displaying the plot
    plt.grid(axis='y')
    plt.legend(title='Season')
    plt.tight_layout()
    plt.show()

# Plotting for water jobs
plot_seasonal_job_counts(df1, 'Water Jobs by Season Over the Years')

# Plotting for mold jobs
plot_seasonal_job_counts(df2, 'Mold Jobs by Season Over the Years')

# Plotting for fire jobs
plot_seasonal_job_counts(df3, 'Fire Jobs by Season Over the Years')


# %%
# Plot waterjob counts
plt.figure(figsize=(10, 6))
plt.title("Water Jobs Performance by Season and Year")
sns.heatmap(waterjob_pivot, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.5, linecolor='black')
plt.xlabel("Season")
plt.ylabel("Year")
plt.show()

# Plot moldjob counts
plt.figure(figsize=(10, 6))
plt.title("Mold Jobs Performance by Season and Year")
sns.heatmap(moldjob_pivot, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.5, linecolor='black')
plt.xlabel("Season")
plt.ylabel("Year")
plt.show()

# Plot firejob counts
plt.figure(figsize=(10, 6))
plt.title("Fire Jobs Performance by Season and Year")
sns.heatmap(firejob_pivot, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=0.5, linecolor='black')
plt.xlabel("Season")
plt.ylabel("Year")
plt.show()

# %%
# Define a function to plot seasonal sale amounts with annotations
def plot_seasonal_sale_amount(df, title):
    # Extract year and season from 'Call Date'
    df['Year'] = df['Call Date'].dt.year
    df['Season'] = df['Month'].apply(get_season)
    
    # Group by year and season, and sum the sale amounts
    seasonal_sale_amount = df.groupby(['Year', 'Season'])['Sale Amount'].sum().unstack(fill_value=0)
    
    # Plotting
    ax = seasonal_sale_amount.plot(kind='line', marker='o', figsize=(10, 6))
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Sale Amount')
    plt.grid(True)
    plt.legend(title='Season')
    plt.tight_layout()
    
    # Adding annotations
    for year in seasonal_sale_amount.index:
        for season in seasonal_sale_amount.columns:
            value = seasonal_sale_amount.loc[year, season]
            ax.annotate(f'{value}', xy=(year, value), xytext=(-10, 5), textcoords='offset points', ha='center', fontsize=8)
    
    plt.show()

# Plot seasonal sale amounts with annotations for water jobs
plot_seasonal_sale_amount(df1, 'Water Jobs Seasonal Sale Amount Over the Years')

# Plot seasonal sale amounts with annotations for mold jobs
plot_seasonal_sale_amount(df2, 'Mold Jobs Seasonal Sale Amount Over the Years')

# Plot seasonal sale amounts with annotations for fire jobs
plot_seasonal_sale_amount(df3, 'Fire Jobs Seasonal Sale Amount Over the Years')





#%%

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Prepare data for forecasting
winter_data_water = waterjob_pivot.loc['Winter'].values
summer_data_water = waterjob_pivot.loc['Summer'].values

winter_data_mold = moldjob_pivot.loc['Winter'].values
summer_data_mold = moldjob_pivot.loc['Summer'].values

winter_data_fire = firejob_pivot.loc['Winter'].values
summer_data_fire = firejob_pivot.loc['Summer'].values

# Fit ARIMA model for Water jobs
winter_model_water = ARIMA(winter_data_water, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
winter_forecast_water = winter_model_water.forecast(steps=len(winter_data_water))  # Forecast for the same length as the original data

summer_model_water = ARIMA(summer_data_water, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
summer_forecast_water = summer_model_water.forecast(steps=len(summer_data_water))

# Fit ARIMA model for Mold jobs
winter_model_mold = ARIMA(winter_data_mold, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
winter_forecast_mold = winter_model_mold.forecast(steps=len(winter_data_mold))

summer_model_mold = ARIMA(summer_data_mold, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
summer_forecast_mold = summer_model_mold.forecast(steps=len(summer_data_mold))

# Fit ARIMA model for Fire jobs
winter_model_fire = ARIMA(winter_data_fire, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
winter_forecast_fire = winter_model_fire.forecast(steps=len(winter_data_fire))

summer_model_fire = ARIMA(summer_data_fire, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
summer_forecast_fire = summer_model_fire.forecast(steps=len(summer_data_fire))



#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame for the forecasted values
forecast_df = pd.DataFrame({
    'Season': ['Winter 2024', 'Summer 2024'],
    'Water Jobs': [winter_forecast_water[0], summer_forecast_water[0]],
    'Mold Jobs': [winter_forecast_mold[0], summer_forecast_mold[0]],
    'Fire Jobs': [winter_forecast_fire[0], summer_forecast_fire[0]]
})

# Set the color palette
colors = sns.color_palette("pastel")

# Display the forecasted values table
print("Forecasted Number of Jobs:")
print(forecast_df.set_index('Season'))

# Visualization: Create a bar plot to show forecasted values as minimal targets for each season
plt.figure(figsize=(12, 8))
sns.barplot(data=forecast_df.melt(id_vars='Season', var_name='Job Type', value_name='Forecasted Jobs'), x='Season', y='Forecasted Jobs', hue='Job Type', palette=colors)
plt.title('Forecasted Number of Jobs as Minimal Targets for Each Season', fontsize=16)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Number of Jobs', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Job Type', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Visualization: Create a line chart to show forecasted values as minimal targets for each season
plt.figure(figsize=(12, 8))
for column in forecast_df.columns[1:]:
    plt.plot(forecast_df['Season'], forecast_df[column], marker='o', label=column, linestyle='-')
plt.title('Forecasted Number of Jobs as Minimal Targets for Each Season', fontsize=16)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Number of Jobs', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Job Type', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#%%

# Extract sale amount data for each type of job and each season
winter_data_water_sales = df1[df1['Season'] == 'Winter']['Sale Amount'].values
summer_data_water_sales = df1[df1['Season'] == 'Summer']['Sale Amount'].values

winter_data_mold_sales = df2[df2['Season'] == 'Winter']['Sale Amount'].values
summer_data_mold_sales = df2[df2['Season'] == 'Summer']['Sale Amount'].values

winter_data_fire_sales = df3[df3['Season'] == 'Winter']['Sale Amount'].values
summer_data_fire_sales = df3[df3['Season'] == 'Summer']['Sale Amount'].values

#%%

# Fit ARIMA model for Water jobs' sale amount
winter_model_water_sale = ARIMA(winter_data_water_sales, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
winter_forecast_water_sales = winter_model_water_sale.forecast(steps=len(winter_data_water_sales))  

summer_model_water_sale = ARIMA(summer_data_water_sales, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
summer_forecast_water_sales = summer_model_water_sale.forecast(steps=len(summer_data_water_sales))

# Fit ARIMA model for Mold jobs' sale amount
winter_model_mold_sale = ARIMA(winter_data_mold_sales, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
winter_forecast_mold_sales = winter_model_mold_sale.forecast(steps=len(winter_data_mold_sales))

summer_model_mold_sale = ARIMA(summer_data_mold_sales, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
summer_forecast_mold_sales = summer_model_mold_sale.forecast(steps=len(summer_data_mold_sales))

# Fit ARIMA model for Fire jobs' sale amount
winter_model_fire_sale = ARIMA(winter_data_fire_sales, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
winter_forecast_fire_sales = winter_model_fire_sale.forecast(steps=len(winter_data_fire_sales))

summer_model_fire_sale = ARIMA(summer_data_fire_sales, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
summer_forecast_fire_sales = summer_model_fire_sale.forecast(steps=len(summer_data_fire_sales))
# Create a DataFrame for the forecasted sale amounts
forecast_sales_df = pd.DataFrame({
    'Season': ['Winter 2024', 'Summer 2024'],
    'Water Jobs': [winter_forecast_water_sales[0], summer_forecast_water_sales[0]],
    'Mold Jobs': [winter_forecast_mold_sales[0], summer_forecast_mold_sales[0]],
    'Fire Jobs': [winter_forecast_fire_sales[0], summer_forecast_fire_sales[0]]
})
#%%
# Set the color palette
colors = sns.color_palette("pastel")

# Display the forecasted sale amounts table
print("Forecasted Sale Amounts:")
print(forecast_sales_df.set_index('Season'))

# Visualization: Create a bar plot to show forecasted sale amounts as minimal targets for each season
plt.figure(figsize=(12, 8))
sns.barplot(data=forecast_sales_df.melt(id_vars='Season', var_name='Job Type', value_name='Forecasted Sales'), x='Season', y='Forecasted Sales', hue='Job Type', palette=colors)
plt.title('Forecasted Sale Amounts as Minimal Targets for Each Season', fontsize=16)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Sale Amount', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Job Type', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()




#%%
