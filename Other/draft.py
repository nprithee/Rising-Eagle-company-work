
#%%

import pandas as pd

# Create a sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)

# Set 'Name' column as the index
df.set_index('Name', inplace=True)

# Using loc to select rows and columns by labels
selected_data_loc = df.loc[['Alice', 'Charlie'], ['Age', 'City']]
print("Using loc:")
print(selected_data_loc)

# Using iloc to select rows and columns by numerical index
selected_data_iloc = df.iloc[[0,2], [0, 1]]
print("\nUsing iloc:")
print(selected_data_iloc)
# %%
import pandas as pd

# Example of a Series
series_data = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'], name='MySeries')
print("Series:")
print(series_data)

# Example of a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)

# %%
import pandas as pd

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Salary': [50000, 60000, 75000]}
df = pd.DataFrame(data)

# Define a function to double the salary
def double_salary(salary):
    return salary * 2

# Applying the function to the 'Salary' column
df['DoubleSalary'] = df['Salary'].apply(double_salary)

# Displaying the updated DataFrame
print("Updated DataFrame with Double Salary:")
print(df)

# %%
import pandas as pd

# Creating two DataFrames
df1 = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
df2 = pd.DataFrame({'Name': ['Charlie', 'David'], 'Age': [35, 40]})

# Appending df2 to df1
combined_df = df1.append(df2, ignore_index=True)

# Displaying the combined DataFrame
print("Combined DataFrame:")
print(combined_df)

# %%
