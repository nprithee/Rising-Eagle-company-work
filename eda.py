#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your financial data (replace 'data.csv' with your data file)
try:
    df = pd.read_csv(r'C:\Users\nprit\Documents\Rising-Eagle-company-work\RisingEagle-profitloss.csv')
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

# Filter the data for the two kinds of amounts you want to compare
amount1_df = df[df['Subcatagory'] == 'Plumbing']
amount2_df = df[df['Subcatagory'] == 'Sales']

# Plot the frequency polygons to compare the two kinds of amounts
plt.figure(figsize=(10, 6))
sns.lineplot(x=amount1_df['Date'], y=amount1_df['Amount'], label='Plumbing')
sns.lineplot(x=amount2_df['Date'], y=amount2_df['Amount'], label='Sales')

plt.title("Comparison between Plumbing and Sales Amount")
plt.xlabel("Date")
plt.ylabel("Amount")
plt.legend()
plt.grid(True)
plt.show()

# Group the data by month
monthly_df = df.groupby(df['Date'].dt.to_period('M'))

#%%
# Function to calculate total income, total cogs, total expenses, and total costing for each month
def calculate_monthly_totals(df):
    # Filter rows where 'Category' is either 'Income', 'COGS', or 'Expenses'
    filtered_df = df[df['Catagory'].isin(['Income', 'COGS', 'Expenses'])]

    # Group the data by 'Date' and 'Category' and sum the 'Amount' within each group
    grouped_df = filtered_df.groupby(['Date', 'Catagory'])['Amount'].sum().unstack(fill_value=0)

    # Extract the Total Income, Total COGS, and Total Expenses for each month
    total_income = grouped_df['Income']
    total_cogs = grouped_df['COGS']
    total_expenses = grouped_df['Expenses']

    # Calculate Total Costing for each month
    total_costing = total_cogs + total_expenses

    return total_income, total_cogs, total_expenses, total_costing

# Example usage of the function
total_income, total_cogs, total_expenses, total_costing = calculate_monthly_totals(df)

# Print the results for each month
for month, income, cogs, expenses, costing in zip(total_income.index, total_income, total_cogs, total_expenses, total_costing):
    print(f'Month: {month}, Total Income: {income}, Total COGS: {cogs}, Total Expenses: {expenses}, Total Costing: {costing}')

# %%

# Example usage of the function
total_income, total_cogs, total_expenses, total_costing = calculate_monthly_totals(df)

# Create a stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting the stacked bars
ax.bar(total_income.index, total_income, label='Total Income', color='green')
ax.bar(total_cogs.index, total_cogs, label='Total COGS', bottom=total_income, color='blue')
ax.bar(total_expenses.index, total_expenses, label='Total Expenses', bottom=total_costing, color='red')
ax.bar(total_costing.index, total_costing, label='Total Costing', color='orange')

# Adding labels and legend
ax.set_title('Comparison of Total COGS, Total Expenses, Total Income, and Total Costing')
ax.set_xlabel('Month')
ax.set_ylabel('Amount')
ax.legend()

plt.show()

