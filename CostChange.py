

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
