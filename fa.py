

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your financial data (replace 'data.csv' with your data file)
try:
    df = pd.read_csv(r'C:/Users/nprit\Documents/Rising-Eagle-company-work/RisingEagle-profitloss.csv')
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

# Group the data by month
monthly_df = df.groupby(df['Date'].dt.to_period('M'))

# Calculate Total Income, Total COGS, Total Expenses, Gross Profit, and Net Operating Income for each month
monthly_metrics = []
for month, group in monthly_df:
    total_income = group[group['Catagory'] == 'Income']['Amount'].sum()
    total_cogs = group[group['Catagory'] == 'COGS']['Amount'].sum()
    total_expenses = group[group['Catagory'] == 'Expenses']['Amount'].sum()
    gross_profit = total_income - total_cogs
    net_operating_income = gross_profit - total_expenses
    
    monthly_metrics.append({
        'Month': month.strftime('%b %Y'),
        'Total Income': total_income,
        'Total COGS': total_cogs,
        'Total Expenses': total_expenses,
        'Gross Profit': gross_profit,
        'Net Operating Income': net_operating_income
    })

# Create a DataFrame to display the calculated metrics for each month
monthly_metrics_df = pd.DataFrame(monthly_metrics)
monthly_metrics_df.set_index('Month', inplace=True)

# Display the DataFrame
print(monthly_metrics_df)

# Create a line plot for month-to-month comparison
plt.figure(figsize=(12, 6))
plt.plot(monthly_metrics_df.index, monthly_metrics_df['Total Income'], label='Total Income', marker='o', linestyle='-')
plt.plot(monthly_metrics_df.index, monthly_metrics_df['Total COGS'], label='Total COGS', marker='o', linestyle='-')
plt.plot(monthly_metrics_df.index, monthly_metrics_df['Total Expenses'], label='Total Expenses', marker='o', linestyle='-')
plt.plot(monthly_metrics_df.index, monthly_metrics_df['Gross Profit'], label='Gross Profit', marker='o', linestyle='-')
plt.plot(monthly_metrics_df.index, monthly_metrics_df['Net Operating Income'], label='Net Operating Income', marker='o', linestyle='-')
plt.xlabel('Month')
plt.ylabel('Amount')
plt.title('Month-to-Month Financial Comparison')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate percentage changes in Net Operating Income
monthly_metrics_df['Net Operating Income Change'] = monthly_metrics_df['Net Operating Income'].pct_change() * 100

# Create a bar chart for percentage changes with annotations
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=monthly_metrics_df.index, y=monthly_metrics_df['Net Operating Income Change'])
plt.xlabel('Month')
plt.ylabel('Percentage Change (%)')
plt.title('Net Operating Income Percentage Change Month-to-Month')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Annotate the bars with their respective percentage values
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2f}%', (x + width/2, y + height), ha='center')

plt.show()

#%%
# total income change of month to month
# Calculate month-to-month changes in Total Income
monthly_metrics_df['Total Income Change'] = monthly_metrics_df['Total Income'].diff()

# Create a line plot for month-to-month changes in Total Income
plt.figure(figsize=(12, 6))
plt.plot(monthly_metrics_df.index, monthly_metrics_df['Total Income Change'], label='Total Income Change', marker='o', linestyle='-', color='b')
plt.xlabel('Month')
plt.ylabel('Change in Total Income')
plt.title('Month-to-Month Changes in Total Income')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# Add labels to the data points
for i, (month, change) in enumerate(zip(monthly_metrics_df.index, monthly_metrics_df['Total Income Change'])):
    plt.text(i, change, f'{change:.2f}', ha='right', va='bottom', fontsize=10)

# Add horizontal line at y=0
plt.axhline(0, color='gray', linestyle='--')

# Customize marker appearance
plt.scatter(monthly_metrics_df.index, monthly_metrics_df['Total Income Change'], c='b', marker='o', s=100, label='Change')

plt.tight_layout()
plt.show()


## Show the ration of Net operating income with dependent variables
#%%
import matplotlib.pyplot as plt
import pandas as pd

# Calculate the ratios
monthly_metrics_df['Net Operating Income to Total Income Ratio'] = monthly_metrics_df['Net Operating Income'] / monthly_metrics_df['Total Income']
monthly_metrics_df['Net Operating Income to Total Expenses Ratio'] = monthly_metrics_df['Net Operating Income'] / monthly_metrics_df['Total Expenses']
monthly_metrics_df['Net Operating Income to Total COGS Ratio'] = monthly_metrics_df['Net Operating Income'] / monthly_metrics_df['Total COGS']

# Create a table to display the mathematical expressions
ratios_table = pd.DataFrame({
    'Ratio Name': ['Net Operating Income to Total Income', 'Net Operating Income to Total Expenses', 'Net Operating Income to Total COGS'],
    'Mathematical Expression': ['Net Operating Income / Total Income', 'Net Operating Income / Total Expenses', 'Net Operating Income / Total COGS']
})

# Display the table
print(ratios_table)

# Create a detailed plot
plt.figure(figsize=(12, 6))

plt.plot(monthly_metrics_df.index, monthly_metrics_df['Net Operating Income to Total Income Ratio'], label='Net Operating Income to Total Income Ratio', marker='o', linestyle='-')
plt.plot(monthly_metrics_df.index, monthly_metrics_df['Net Operating Income to Total Expenses Ratio'], label='Net Operating Income to Total Expenses Ratio', marker='o', linestyle='-')
plt.plot(monthly_metrics_df.index, monthly_metrics_df['Net Operating Income to Total COGS Ratio'], label='Net Operating Income to Total COGS Ratio', marker='o', linestyle='-')

plt.xlabel('Month')
plt.ylabel('Ratio')
plt.title('Ratios of Net Operating Income to Financial Metrics Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# Annotate the data points with labels
for i, row in monthly_metrics_df.iterrows():
    plt.annotate(f'Income: {row["Total Income"]:.2f}\nExpenses: {row["Total Expenses"]:.2f}\nCOGS: {row["Total COGS"]:.2f}', (i, row['Net Operating Income to Total Income Ratio']))

plt.tight_layout()
plt.show()


# corelation between the all calculted variables
# %%

# Calculate the correlation matrix
correlation_matrix = monthly_metrics_df.corr()

# Create a heatmap to visualize the correlation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()




# %%
# Create a DataFrame to display the calculated metrics for each month
monthly_metrics_df = pd.DataFrame(monthly_metrics)

# Calculate the correlation matrix
correlation_matrix = monthly_metrics_df.corr()

# Create a heatmap to visualize the correlation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# Effets on dependent variables of dependent variables
# %%

import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for the plots
sns.set(style="whitegrid")

# Create subplots for better organization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter plot for Total Income vs. Net Operating Income
sns.scatterplot(x='Total Income', y='Net Operating Income', data=monthly_metrics_df, hue=monthly_metrics_df.index, ax=axes[0, 0], palette='viridis', legend=False)
sns.regplot(x='Total Income', y='Net Operating Income', data=monthly_metrics_df, scatter=False, ax=axes[0, 0], color='blue')
axes[0, 0].set_title('Scatter Plot: Total Income vs. Net Operating Income')
axes[0, 0].set_xlabel('Total Income')
axes[0, 0].set_ylabel('Net Operating Income')

# Scatter plot for Total COGS vs. Net Operating Income
sns.scatterplot(x='Total COGS', y='Net Operating Income', data=monthly_metrics_df, hue=monthly_metrics_df.index, ax=axes[0, 1], palette='viridis', legend=False)
sns.regplot(x='Total COGS', y='Net Operating Income', data=monthly_metrics_df, scatter=False, ax=axes[0, 1], color='red')
axes[0, 1].set_title('Scatter Plot: Total COGS vs. Net Operating Income')
axes[0, 1].set_xlabel('Total COGS')
axes[0, 1].set_ylabel('Net Operating Income')

# Scatter plot for Total Expenses vs. Net Operating Income
sns.scatterplot(x='Total Expenses', y='Net Operating Income', data=monthly_metrics_df, hue=monthly_metrics_df.index, ax=axes[1, 0], palette='viridis', legend=False)
sns.regplot(x='Total Expenses', y='Net Operating Income', data=monthly_metrics_df, scatter=False, ax=axes[1, 0], color='green')
axes[1, 0].set_title('Scatter Plot: Total Expenses vs. Net Operating Income')
axes[1, 0].set_xlabel('Total Expenses')
axes[1, 0].set_ylabel('Net Operating Income')

# Hide the empty subplot
axes[1, 1].axis('off')

# Add legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()



# %%
