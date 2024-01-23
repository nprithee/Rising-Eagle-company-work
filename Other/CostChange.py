

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%


df = pd.read_csv('/Users/nusratprithee/Documents/Rising-Eagle-company-work/Purchase-History-with-Average-Unit-Price-2020-vs.-2021.csv')
columns_to_drop = ["Type", "Qty Sold 01/01/21 to 12/31/21", "Sales $$ 01/01/21 to 12/31/21", "Qty Sold 01/01/20 to 12/31/20", "Sales $$ 01/01/20 to 12/31/20"]
df = df.drop(columns_to_drop, axis=1)

# Convert the average unit price columns from strings to numeric values
df['Average Unit Price 01/01/21 to 12/31/21'] = df['Average Unit Price 01/01/21 to 12/31/21'].str.replace('$', '').str.replace(',', '').astype(float)
df['Average Unit Price 01/01/20 to 12/31/20'] = df['Average Unit Price 01/01/20 to 12/31/20'].str.replace('$', '').str.replace(',', '').astype(float)

# Calculate the percentage change
df['Percentage Change'] = ((df['Average Unit Price 01/01/21 to 12/31/21'] - df['Average Unit Price 01/01/20 to 12/31/20']) / df['Average Unit Price 01/01/20 to 12/31/20']) * 100

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Item #'], df['Percentage Change'])

# Add labels and title
plt.xlabel('Item #')
plt.ylabel('Percentage Change')
plt.title('Percentage Change of Average Unit Price (2020 to 2021)')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Add percentage values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}%", ha='center', va='bottom')

# Display the chart
plt.tight_layout()
plt.show()

#%%
#Heatmap

# Create a heatmap
plt.figure(figsize=(10, 6))
heatmap = plt.imshow(heatmap_data, cmap='coolwarm', interpolation='nearest')

# Add percentage values in each cell
for i in range(len(df['Item #'])):
    for j in range(len(df['Description'])):
        text = "{:.1f}%".format(heatmap_data.iloc[i, j])
        plt.text(j, i, text, ha='center', va='center', color='white', fontsize=8)

# Add colorbar
cbar = plt.colorbar(heatmap)
cbar.set_label('Percentage Change')

# Add grid lines
plt.grid(True, which='both', color='lightgray', linewidth=0.5)

# Add labels and title
plt.xlabel('Description')
plt.ylabel('Item #')
plt.title('Percentage Change of Average Unit Price (2020 to 2021)')

# Adjust the layout
plt.tight_layout()

# Display the chart
plt.show()

## Barplot

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file and drop unnecessary columns
df = pd.read_csv('/Users/nusratprithee/Documents/Rising-Eagle-company-work/Purchase-History-with-Average-Unit-Price-2020-vs.-2021.csv')
columns_to_drop = ["Type", "Qty Sold 01/01/21 to 12/31/21", "Sales $$ 01/01/21 to 12/31/21", "Qty Sold 01/01/20 to 12/31/20", "Sales $$ 01/01/20 to 12/31/20"]
df = df.drop(columns_to_drop, axis=1)

# Convert the average unit price columns from strings to numeric values
df['Average Unit Price 01/01/21 to 12/31/21'] = df['Average Unit Price 01/01/21 to 12/31/21'].str.replace('$', '').str.replace(',', '').astype(float)
df['Average Unit Price 01/01/20 to 12/31/20'] = df['Average Unit Price 01/01/20 to 12/31/20'].str.replace('$', '').str.replace(',', '').astype(float)

# Calculate the percentage change of the average unit price
df['Percentage Change'] = ((df['Average Unit Price 01/01/21 to 12/31/21'] - df['Average Unit Price 01/01/20 to 12/31/20']) / df['Average Unit Price 01/01/20 to 12/31/20']) * 100

# Sort the dataframe by percentage change in descending order
df = df.sort_values('Percentage Change', ascending=False)

# Plot the bar chart
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Percentage Change', y='Item #', data=df, color='steelblue')
plt.xlabel('Percentage Change')
plt.ylabel('Item #')
plt.title('Percentage Change of Items')

# Annotate the bars with percentage change values
for i, v in enumerate(df['Percentage Change']):
    ax.text(v, i, f'{v:.2f}%', va='center')

plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('/Users/nusratprithee/Documents/Rising-Eagle-company-work/Purchase-History-with-Average-Unit-Price-2020-vs.-2021.csv')



# 2. Data Preprocessing
# Clean up column names
df.columns = df.columns.str.strip()

# 3. Calculate Percentage Change
df['Percentage Change'] = ((df['Qty Sold 01/01/21 to 12/31/21'] - df['Qty Sold 01/01/20 to 12/31/20']) / df['Qty Sold 01/01/20 to 12/31/20']) * 100

# 4. Bar Chart of Percentage Change
plt.figure(figsize=(12, 6))
plt.bar(df['Item #'], df['Percentage Change'])
plt.xlabel('Item #')
plt.ylabel('Percentage Change')
plt.title('Percentage Change of Quantity Sold')
plt.xticks(rotation=90)
plt.show()

# 5. Heatmap of Percentage Change
pivot_table = df.pivot_table(values='Percentage Change', index='Item #', columns='Description')
plt.figure(figsize=(10, 6))
plt.imshow(pivot_table, cmap='RdYlGn', interpolation='nearest', aspect='auto')
plt.colorbar(label='Percentage Change')
plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=90)
plt.yticks(range(len(pivot_table.index)), pivot_table.index)
plt.title('Heatmap of Percentage Change')
plt.xlabel('Description')
plt.ylabel('Item #')
plt.tight_layout()
plt.show()

# 6. Percentage Change Bar Chart for Each Item
plt.figure(figsize=(12, 6))
for index, row in df.iterrows():
    plt.bar(row['Description'], row['Percentage Change'], label=row['Item #'])
plt.xlabel('Description')
plt.ylabel('Percentage Change')
plt.title('Percentage Change of Quantity Sold for Each Item')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# 7. GPCA Table
gpca_table = df.groupby(['Item #', 'Description'])['Qty Sold 01/01/21 to 12/31/21', 'Qty Sold 01/01/20 to 12/31/20'].sum().reset_index()

# 8. Line Plot of Quantity Sold
plt.figure(figsize=(12, 6))
plt.plot(df['Item #'], df['Qty Sold 01/01/21 to 12/31/21'], marker='o', label='Qty Sold 2021')
plt.plot(df['Item #'], df['Qty Sold 01/01/20 to 12/31/20'], marker='o', label='Qty Sold 2020')
plt.xlabel('Item #')
plt.ylabel('Quantity Sold')
plt.title('Quantity Sold Comparison')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()



# %%
