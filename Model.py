# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Load the datasets
test_df = pd.read_csv('/Users/kevinharvey/Desktop/AI Project/test.csv')
train_df = pd.read_csv('/Users/kevinharvey/Desktop/AI Project/train.csv')
# Drop rows with missing values
train_df.dropna()
test_df.dropna()
# Separate target variable
y_train = train_df['SalePrice']
# Select numerical features excluding 'Id' and 'SalePrice'
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_features = [feature for feature in numerical_features if feature not in ['Id', 'SalePrice']]
# Prepare feature sets
X_train = train_df[numerical_features]
X_test = test_df[numerical_features]
# Output the number of rows and columns of the datasets and selected features
print("Datasets reloaded and features selected successfully.")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
## Handle Missing Values

# Handle missing values by imputing with mean
for col in X_train.columns:
    if X_train[col].isnull().any():
        mean_val = X_train[col].mean()
        X_train[col].fillna(mean_val, inplace=True)
        X_test[col].fillna(mean_val, inplace=True)

print("Missing values in X_train after imputation:")
print(X_train.isnull().sum().sum())
print("Missing values in X_test after imputation:")
print(X_test.isnull().sum().sum())

**Reasoning**:
The previous code produced SettingWithCopyWarning and failed to impute all missing values in X_test, leaving 10 NaNs


imputation_means = X_train.mean()

X_train = X_train.fillna(imputation_means)
X_test = X_test.fillna(imputation_means)

print("Missing values in X_train after imputation:")
print(X_train.isnull().sum().sum())
print("Missing values in X_test after imputation:")
print(X_test.isnull().sum().sum())
## Train Linear Regression Model

### Reasoning:
Initialize and train a Linear Regression model from scikit-learn using the prepared training data.

model = LinearRegression()
model.fit(X_train, y_train)

print("Linear Regression model trained successfully.")
## Predict and Display Results

### Reasoning:
Use the trained model to make predictions on the test dataset and display the first few predicted house prices.

## Summary:

### Data Analysis Key Findings

*   **Data Loading and Feature Selection**: `train.csv` (1460 rows) and `test.csv` (1459 rows) datasets were successfully loaded. `SalePrice` was designated as the target variable. A set of 36 numerical features, excluding 'Id' and 'SalePrice', was selected for both training and testing datasets.
*   **Missing Value Handling**: Missing values in both `X_train` and `X_test` were effectively imputed using the mean values computed from `X_train`. Post-imputation, both datasets reported 0 remaining missing values, ensuring data completeness for model training.
*   **Model Training**: A Linear Regression model was successfully initialized and trained using the prepared `X_train` (features) and `y_train` (target variable) datasets.

# Make predictions on the test set
y_pred = model.predict(X_test)
print("First 5 predicted house prices:")
print(y_pred[:5])
# Evaluating Performance
print(f"Mean Squared Error on training data: {mse_train:.2f}")
rmse=np.sqrt(mse_train)
print(f"Root Mean Squared Error (unfiltered model on training data): {rmse:.2f}")
