

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your financial data (replace 'data.csv' with your data file)
try:
    df = pd.read_csv('/Users/nusratprithee/Documents/Rising-Eagle-company-work/RisingEagle_profitloss.csv')
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)

# Check the column names to verify the 'Date' column
print(df.columns)

# Data Preprocessing
# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')

# Handle missing values in the 'Amount' column
df['Amount'] = df['Amount'].str.replace(',', '').astype(float)  # Remove commas and convert to float
df['Amount'].fillna(0, inplace=True)  # Fill missing values with 0

# Filter the data for the two kinds of amounts you want to compare
amount1_df = df[df['Subcatagory'] == 'Plumbing']
amount2_df = df[df['Subcatagory'] == 'Sales']

# Plot the scatter plot to compare the two kinds of amounts
plt.figure(figsize=(10, 6))
plt.scatter(amount1_df['Date'], amount1_df['Amount'], label='Plumbing', alpha=0.7)
plt.scatter(amount2_df['Date'], amount2_df['Amount'], label='Sales', alpha=0.7)
plt.title("Comparison between Plumbing and Sales Amount")
plt.xlabel("Date")
plt.ylabel("Amount")
plt.legend()
plt.grid(True)
plt.show()

# Filter the data for "Income" category and the specific subcategories
income_df = df[(df['Catagory'] == 'Income') & (df['Subcatagory'].isin(['Plumbing', 'Sales']))]


# Pivot the data to have subcategories as columns and months as index
pivot_df = income_df.pivot(index='Date', columns='Subcatagory', values='Amount')

# Create a bar chart with stacked bars
plt.figure(figsize=(12, 6))
pivot_df.plot(kind='bar', stacked=True)
plt.title("Income Comparison between Plumbing and Sales")
plt.xlabel("Month")
plt.ylabel("Income Amount")
plt.legend(title='Subcategory')
plt.grid(True)
plt.show()


# Create a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Income Heatmap between Plumbing and Sales")
plt.xlabel("Subcategory")
plt.ylabel("Month")
plt.show()
# %%

# Convert the 'Date' column to a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Group the data by 'Date' and 'Catagory', then sum the 'Amount' within each group
grouped_cogs_expenses_df = df.groupby(['Date', 'Catagory'])['Amount'].sum().unstack(fill_value=0)

# Calculate the total COGS and Expenses for each month
grouped_cogs_expenses_df['Total COGS'] = grouped_cogs_expenses_df['COGS']
grouped_cogs_expenses_df['Total Expenses'] = grouped_cogs_expenses_df['Expenses']

# Calculate the Total Income as the difference between Total COGS and Total Expenses
grouped_cogs_expenses_df['Total Income'] = grouped_cogs_expenses_df['Total COGS'] - grouped_cogs_expenses_df['Total Expenses']

# Calculate the total for each month
grouped_cogs_expenses_df['Total'] = grouped_cogs_expenses_df.sum(axis=1)

# Calculate the percentages
percentage_df = grouped_cogs_expenses_df[['Total COGS', 'Total Expenses', 'Total Income']].div(grouped_cogs_expenses_df['Total'], axis=0) * 100

# Create a stacked bar chart to compare Total COGS, Total Expenses, and Income as percentages
ax = percentage_df.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Percentage Comparison between Total COGS, Total Expenses, and Income")
plt.xlabel("Month")
plt.ylabel("Percentage")
plt.legend(loc='upper right', labels=['Total COGS', 'Total Expenses', 'Total Income'])
plt.grid(True)

# Add percentage labels on top of each bar segment
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2f}%', (x + width/2, y + height), ha='center')

plt.show()

# Profit and loss Analysis
#%%

# Calculate net profit (Income - Expenses) over time
grouped_cogs_expenses_df['Net Profit'] = grouped_cogs_expenses_df['Total Income'] - grouped_cogs_expenses_df['Total Expenses']

# Filter data for the year 2023
net_profit_2023 = grouped_cogs_expenses_df['Net Profit']['2023']

# Create a line plot to visualize the trend of net profit for 2023
plt.figure(figsize=(12, 6))
plt.plot(net_profit_2023.index, net_profit_2023.values, marker='o', linestyle='-', color='b')
plt.xlabel('Date')
plt.ylabel('Net Profit')
plt.title('Net Profit Over Time in 2023')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Calculate month-to-month changes
month_to_month_changes = net_profit_2023.diff()

# Create a bar plot to show the month-to-month changes
plt.figure(figsize=(12, 6))
month_to_month_changes.plot(kind='bar', color='c')
plt.xlabel('Month (2023)')
plt.ylabel('Net Profit Change')
plt.title('Month-to-Month Changes in Net Profit (2023)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate month-to-month percentage changes
percentage_changes = (net_profit_2023 / net_profit_2023.shift(1) - 1) * 100

# Create a bar plot to show the month-to-month percentage changes with labels
plt.figure(figsize=(12, 6))
bars = percentage_changes.plot(kind='bar', color='m')
plt.xlabel('Month (2023)')
plt.ylabel('Net Profit % Change')
plt.title('Month-to-Month Percentage Changes in Net Profit (2023)')
plt.grid(True)
plt.xticks(rotation=45)

# Add percentage labels on top of each bar
for bar in bars.patches:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}%', (bar.get_x() + bar.get_width() / 2, height),
                 ha='center', va='bottom')

plt.tight_layout()
plt.show()


#Mathematical Expression
# %%
# Filter data for the year 2023
net_profit_2023 = grouped_cogs_expenses_df['Net Profit']['2023']

# Display the net profit for 2023
print(net_profit_2023)

# Filter data for the year 2023
net_profit_2023 = grouped_cogs_expenses_df['Net Profit']['2023']

# Calculate month-to-month changes
month_to_month_changes = net_profit_2023.diff()

# Display the month-to-month changes in net profit for 2023
print(month_to_month_changes)


#Month To Month Comparison

#%%
# Filter data for the year 2023
data_2023 = grouped_cogs_expenses_df[grouped_cogs_expenses_df.index.year == 2023]

# Create a line plot for Total Income, Total Expenses, and Net Profit in 2023
plt.figure(figsize=(12, 6))
plt.plot(data_2023.index, data_2023['Total Income'], marker='o', label='Total Income', linestyle='-', color='g')
plt.plot(data_2023.index, data_2023['Total Expenses'], marker='o', label='Total Expenses', linestyle='-', color='r')
plt.plot(data_2023.index, data_2023['Net Profit'], marker='o', label='Net Profit', linestyle='-', color='b')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.title('Month-to-Month Comparison of Income, Expenses, and Net Profit for 2023')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


#COGS Analysis of 2023 with the subcatagories
# %%

import pandas as pd
import matplotlib.pyplot as plt
import calendar

# Assuming you already have df and it contains the necessary data

# Filter Category
cogs_2023_df = df[(df['Catagory'] == 'COGS') & (df['Date'].dt.year == 2023)]

# Pivot the data
pivot_cogs_2023_df = cogs_2023_df.pivot_table(index='Date', columns='Subcatagory', values='Amount', aggfunc='sum', fill_value=0)

# Sort subcategories by total amount for each month in descending order
pivot_cogs_2023_df = pivot_cogs_2023_df[pivot_cogs_2023_df.sum().sort_values(ascending=False).index]

# Create a stacked bar chart
plt.figure(figsize=(12, 8))
width = 0.7  # Width of each bar

# Extract the subcategories and months
subcategories = pivot_cogs_2023_df.columns
months = sorted(pivot_cogs_2023_df.index.month.unique())  # Get unique months

# Calculate the position of bars on the x-axis
x = range(len(months))

# Initialize a list to keep track of the bottom position for each subcategory
bottom = [0] * len(months)

# Create stacked bars
for subcategory in subcategories:
    plt.bar(
        x,
        pivot_cogs_2023_df[subcategory],
        width=width,
        label=subcategory,
        bottom=bottom,
    )
    bottom = [bottom[i] + pivot_cogs_2023_df[subcategory][i] for i in range(len(months))]

# Customize the chart
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('COGS Analysis by Subcategory for 2023')
plt.xticks(x, [calendar.month_abbr[month] for month in months])
plt.legend(loc='upper right', title='Subcatagory', bbox_to_anchor=(1.25, 1.0))
plt.grid(True, axis='y')

# Show the chart
plt.tight_layout()
plt.show()
# %%

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you already have df and it contains the necessary data

# Filter Category
cogs_2023_df = df[(df['Catagory'] == 'COGS') & (df['Date'].dt.year == 2023)]

# Group and sum by Subcategory
subcategories_sum = cogs_2023_df.groupby('Subcatagory')['Amount'].sum().reset_index()

# Sort by Amount in descending order and take the top 8
top_8_subcategories = subcategories_sum.nlargest(8, 'Amount')

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_8_subcategories['Amount'], labels=top_8_subcategories['Subcatagory'], autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})

# Add a legend
plt.legend(labels=top_8_subcategories['Subcatagory'], title='Subcategory', loc='best')

# Add a title
plt.title('Top 8 COGS Subcategories for 2023')

# Show the chart
plt.tight_layout()
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#Expenses Analysis of 2023 with the subcatagories
# %%
import pandas as pd
import matplotlib.pyplot as plt
import calendar

# Assuming you already have df and it contains the necessary data

# Filter Category
cogs_2023_df = df[(df['Catagory'] == 'Expenses') & (df['Date'].dt.year == 2023)]

# Pivot the data
pivot_cogs_2023_df = cogs_2023_df.pivot_table(index='Date', columns='Subcatagory', values='Amount', aggfunc='sum', fill_value=0)

# Sort subcategories by total amount for each month in descending order
pivot_cogs_2023_df = pivot_cogs_2023_df[pivot_cogs_2023_df.sum().sort_values(ascending=False).index]

# Create a stacked bar chart
plt.figure(figsize=(12, 8))
width = 0.7  # Width of each bar

# Extract the subcategories and months
subcategories = pivot_cogs_2023_df.columns
months = sorted(pivot_cogs_2023_df.index.month.unique())  # Get unique months

# Calculate the position of bars on the x-axis
x = range(len(months))

# Initialize a list to keep track of the bottom position for each subcategory
bottom = [0] * len(months)

# Create stacked bars
for subcategory in subcategories:
    plt.bar(
        x,
        pivot_cogs_2023_df[subcategory],
        width=width,
        label=subcategory,
        bottom=bottom,
    )
    bottom = [bottom[i] + pivot_cogs_2023_df[subcategory][i] for i in range(len(months))]

# Customize the chart
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('Expenses Analysis by Subcategory for 2023')
plt.xticks(x, [calendar.month_abbr[month] for month in months])
plt.legend(loc='upper right', title='Subcatagory', bbox_to_anchor=(1.25, 1.0))
plt.grid(True, axis='y')

# Show the chart
plt.tight_layout()
plt.show()
# %%

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you already have df and it contains the necessary data

# Filter Category
cogs_2023_df = df[(df['Catagory'] == 'Expenses') & (df['Date'].dt.year == 2023)]

# Group and sum by Subcategory
subcategories_sum = cogs_2023_df.groupby('Subcatagory')['Amount'].sum().reset_index()

# Sort by Amount in descending order and take the top 8
top_8_subcategories = subcategories_sum.nlargest(8, 'Amount')

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_8_subcategories['Amount'], labels=top_8_subcategories['Subcatagory'], autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})

# Add a legend
plt.legend(labels=top_8_subcategories['Subcatagory'], title='Subcategory', loc='best')

# Add a title
plt.title('Top 8 Expenses Subcategories for 2023')

# Show the chart
plt.tight_layout()
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


#Random Forest 
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')

# Handle missing values in the 'Amount' column
df['Amount'].fillna(0, inplace=True)  # Fill missing values with 0




# Encode categorical features (Category and Subcategory) using one-hot encoding
df = pd.get_dummies(df, columns=['Catagory', 'Subcatagory'], drop_first=True)

# Step 2: Split the Data
# Split the dataset into a training set (80%) and a testing set (20%)
X = df.drop(columns=['Amount'])
y = df['Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest Model
# Create and train a Random Forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Evaluate the Model
# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Visualize predicted vs. actual 'Amount' values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Amount")
plt.ylabel("Predicted Amount")
plt.title("Actual vs. Predicted Amount (Random Forest)")
plt.grid(True)
plt.show()

# Step 6: Interpret the Results
# Visualize feature importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Step 7: Anomaly Detection and Business Strategy
# You can identify anomalies by comparing predicted and actual 'Amount' values.
# Investigate instances where the model's predictions significantly deviate from actuals.

# Step 8: Forecasting
# To make future predictions, prepare input data for upcoming periods and use the trained model.
# Example:
# future_data = pd.DataFrame(...)  # Prepare data for future periods
# future_predictions = rf_model.predict(future_data)

# Optionally, you can save the trained model for future use:
# import joblib
# joblib.dump(rf_model, 'financial_model.pkl')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load your financial data (replace 'data.csv' with your data file)
try:
    df = pd.read_csv('/Users/nusratprithee/Documents/Rising-Eagle-company-work/RisingEagle_profitloss.csv')
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)

# Check the column names to verify the 'Date' column
print(df.columns)

# Data Preprocessing
# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')

# Handle missing values in the 'Amount' column
df['Amount'] = df['Amount'].str.replace(',', '').astype(float)  # Remove commas and convert to float
df['Amount'].fillna(0, inplace=True)  # Fill missing values with 0

# Filter the data for the period of January 2023 to July 2023
start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2023-07-31')
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Calculate Total Income, Total COGS, Total Expenses for the specified period
total_income = filtered_df[filtered_df['Catagory'] == 'Income']['Amount'].sum()
total_cogs = filtered_df[filtered_df['Catagory'] == 'COGS']['Amount'].sum()
total_expenses = filtered_df[filtered_df['Catagory'] == 'Expenses']['Amount'].sum()

# Calculate Gross Profit and Net Operating Income
gross_profit = total_income - total_cogs
net_operating_income = gross_profit - total_expenses

# Create a DataFrame to display the calculated metrics
period_metrics = pd.DataFrame({
    'Metric': ['Total Income', 'Total COGS', 'Total Expenses', 'Gross Profit', 'Net Operating Income'],
    'Value': [total_income, total_cogs, total_expenses, gross_profit, net_operating_income]
})

# Set the index to 'Metric'
period_metrics.set_index('Metric', inplace=True)

# Display the DataFrame
print(period_metrics)