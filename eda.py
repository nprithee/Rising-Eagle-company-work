
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


#%%



# Calculate Total Costing as the sum of Total COGS and Total Expenses
grouped_cogs_expenses_df['Total Costing'] = grouped_cogs_expenses_df['Total COGS'] + grouped_cogs_expenses_df['Total Expenses']

# Calculate the Total Income as the difference between Total Costing and Total Expenses
grouped_cogs_expenses_df['Total Income'] = grouped_cogs_expenses_df['Total Costing'] - grouped_cogs_expenses_df['Total Expenses']

# Create a histogram to compare Total Costing and Total Income over time
plt.figure(figsize=(12, 6))
plt.bar(grouped_cogs_expenses_df.index, grouped_cogs_expenses_df['Total Costing'], label='Total Costing', alpha=0.7)
plt.bar(grouped_cogs_expenses_df.index, grouped_cogs_expenses_df['Total Income'], label='Total Income', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Amount')
plt.title('Comparison of Total Costing and Total Income Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%%
import seaborn as sns
import matplotlib.pyplot as plt


# Pivot the data to have categories as columns and months as index
pivot_cogs_expenses_df = df.pivot_table(index='Date', columns='Catagory', values='Amount', aggfunc='sum', fill_value=0)

# Create a heatmap to visualize the data
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_cogs_expenses_df, cmap='coolwarm', annot=True, fmt=".1f", cbar=True)
plt.title("COGS and Expenses Heatmap")
plt.xlabel("Category")
plt.ylabel("Month")
plt.show()






# %%


# Convert the 'Date' column to a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Filter rows where 'Category' is either 'COGS' or 'Expenses'
cogs_expenses_df = df[df['Catagory'].isin(['COGS', 'Expenses'])]

# Group the data by 'Date' and 'Category', then sum the 'Amount' within each group
grouped_cogs_expenses_df = cogs_expenses_df.groupby(['Date', 'Catagory'])['Amount'].sum().unstack(fill_value=0)

# Calculate the total COGS and Expenses for each month
grouped_cogs_expenses_df['Total COGS'] = grouped_cogs_expenses_df['COGS']
grouped_cogs_expenses_df['Total Expenses'] = grouped_cogs_expenses_df['Expenses']

# Filter the data for "Income" category and the specific subcategories
income_df = df[(df['Catagory'] == 'Income') & (df['Subcatagory'].isin(['Plumbing', 'Sales']))]

grouped_cogs_expenses_df['Total Income'] = grouped_cogs_expenses_df[income_df].sum(axis=1)

# Calculate Total Costing as the sum of Total COGS and Total Expenses
grouped_cogs_expenses_df['Total Costing'] = grouped_cogs_expenses_df['Total COGS'] + grouped_cogs_expenses_df['Total Expenses']

# Create a stacked bar chart to compare Total COGS, Total Expenses, Total Income, and Total Costing
ax = grouped_cogs_expenses_df[['Total COGS', 'Total Expenses', 'Total Income', 'Total Costing']].plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Comparison of Total COGS, Total Expenses, Total Income, and Total Costing")
plt.xlabel("Month")
plt.ylabel("Amount")
plt.legend(loc='upper right', labels=['Total COGS', 'Total Expenses', 'Total Income', 'Total Costing'])
plt.grid(True)

# Add data labels on top of each bar segment
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if height > 0:
        ax.annotate(f'{height:.2f}', (x + width/2, y + height), ha='center')

plt.show()

# %%
