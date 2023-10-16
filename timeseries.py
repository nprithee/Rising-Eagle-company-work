
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load your financial data (replace 'data.csv' with your data file)
try:
    df = pd.read_csv('/Users/nusratprithee/Documents/Rising-Eagle-company-work/RisingEagle_profitloss.csv')
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)

# Data Preprocessing
# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
# Set date column as index
df.set_index('Date', inplace=True)

# Handle missing values in the 'Amount' column
df['Amount'] = df['Amount'].str.replace(',', '').astype(float)  # Remove commas and convert to float
df['Amount'].fillna(0, inplace=True)  # Fill missing values with 0

# Create a dictionary to map categories to colors for better visualization
category_colors = {
    'Income': 'green',
    'COGS': 'red',
    'Expenses': 'blue'
}

# Visualize income, COGS, and expenses over time with improved colors
plt.figure(figsize=(12, 6))
for category, color in category_colors.items():
    category_data = df[df['Catagory'] == category]
    plt.plot(category_data.index, category_data['Amount'], label=category, color=color, linewidth=2)

plt.xlabel('Date')
plt.ylabel('Amount')
plt.title('Income, COGS, and Expenses Over Time')
plt.legend()
plt.grid(True)  # Add grid lines for better readability
plt.tight_layout()  # Ensure labels are not cut off
plt.show()

# Time Series Analysis (Decomposition) - Use additive model
result = seasonal_decompose(df['Amount'], model='additive', period=12)  # Assuming monthly data
result.plot()
plt.suptitle('Time Series Decomposition', fontsize=16)  # Add a title to the decomposition plots
plt.show()

# Simple Time Series Modeling (SARIMA)
# Assuming you want to forecast Amount
train_size = int(0.8 * len(df))
train, test = df['Amount'][:train_size], df['Amount'][train_size:]

model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot predictions vs. actual values with improved styling
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='green', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.title('Amount Forecasting')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
