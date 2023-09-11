

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('financial_data.csv')

# Calculate net profit for each month
df['Net Profit'] = df['Income'] - (df['Expenses'] + df['COGS'])

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(df['Date'], df['Net Profit'], color='blue')
plt.xlabel('Month')
plt.ylabel('Net Profit')
plt.title('Net Profit for Each Month')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


#%%
