

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#%%
# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/nusratprithee/Documents/Rising-Eagle-company-work/RisingEagle_profitloss.csv')
print(df)

## Data Preprocessing
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Assuming 'df' is your DataFrame
# Handle missing values if any
df.fillna(0, inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Catagory', 'Subcatagory']  # Update with your actual columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split the data into training and testing sets
X = df.drop(columns=['Amount'])
y = df['Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Exploratory Data Analysis (EDA)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Example EDA plots
sns.histplot(df['Amount'])
plt.show()

sns.boxplot(x='Category', y='Amount', data=df)
plt.show()

# Explore correlations
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

##  Model Selection and Training

#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


## Model Evaluation
#%%

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

## Interpretation and Visualization
#%%

# Feature importances
importances = model.feature_importances_
features = X.columns
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()

