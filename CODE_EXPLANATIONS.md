# Machine Learning Code Explanations

This document provides a detailed, line-by-line explanation of the code used in `ML_Regression_Comparison.ipynb`.

## 1. Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ All libraries imported successfully!")
```

**Detailed Explanation:**

- `import pandas as pd`: Imports the Pandas library and aliases it as `pd`. Pandas is essential for data manipulation and analysis, offering data structures like DataFrames.
- `import numpy as np`: Imports the NumPy library as `np`. NumPy provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.
- `import matplotlib.pyplot as plt`: Imports the Pyplot interface from Matplotlib as `plt`. This is the standard library for creating static, animated, and interactive visualizations.
- `import seaborn as sns`: Imports Seaborn as `sns`. Seaborn is a visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics.
- `from sklearn.linear_model import LinearRegression`: Imports the `LinearRegression` class from Scikit-Learn. This algorithm models the relationship between two or more variables by fitting a linear equation to observed data.
- `from sklearn.model_selection import train_test_split`: Imports `train_test_split`, a utility function to split arrays or matrices into random train and test subsets. This is crucial for evaluating model performance.
- `from sklearn.preprocessing import StandardScaler`: Imports `StandardScaler`, which standardizes features by removing the mean and scaling to unit variance (z-score normalization).
- `from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error`: Imports metric functions to evaluate regression models:
    - `mean_squared_error`: The average squared difference between estimated values and the actual value.
    - `r2_score`: The coefficient of determination, basically how well the model predicts the variance in the data.
    - `mean_absolute_error`: The average of the absolute errors.
- `from sklearn.ensemble import RandomForestRegressor`: Imports `RandomForestRegressor`, an ensemble learning method that operates by constructing a multitude of decision trees at training time.
- `import xgboost as xgb`: Imports XGBoost (Extreme Gradient Boosting), a powerful and efficient gradient boosting library.
- `import lightgbm as lgb`: Imports LightGBM, another gradient boosting framework that uses tree-based learning algorithms, often faster than XGBoost.
- `from catboost import CatBoostRegressor`: Imports the regressor from CatBoost, a gradient boosting library that handles categorical variables well.
- `import tensorflow as tf`: Imports TensorFlow, an end-to-end open-source platform for machine learning.
- `from tensorflow.keras.models import Sequential`: Imports the `Sequential` model type from Keras (TF's high-level API), which allows building models layer by layer.
- `from tensorflow.keras.layers import Dense, Dropout`: Imports layer types:
    - `Dense`: A regular fully connected neural network layer.
    - `Dropout`: A regularization layer that randomly sets input units to 0 to prevent overfitting.
- `from tensorflow.keras.optimizers import Adam`: Imports the Adam optimization algorithm, which is an efficient stochastic gradient descent method.
- `import warnings`: Imports the standard Python `warnings` library.
- `warnings.filterwarnings('ignore')`: Configures the warning filter to ignore warning messages, keeping the notebook output clean.
- `sns.set_style("whitegrid")`: Sets the Seaborn plot style to "whitegrid", which adds a white background with grid lines for better readability.
- `plt.rcParams['figure.figsize'] = (12, 6)`: Sets the default figure size for Matplotlib plots to 12 inches wide by 6 inches tall.
- `print("✓ All libraries imported successfully!")`: Prints a confirmation message to the console indicating that all imports ran without error.

## 2. Load and Prepare Data

```python
# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Separate target variable
y = train_df['SalePrice']

# Select numerical features excluding 'Id' and 'SalePrice'
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_features = [feature for feature in numerical_features if feature not in ['Id', 'SalePrice']]

# Prepare feature sets
X = train_df[numerical_features].copy()
X_test_full = test_df[numerical_features].copy()

# Handle missing values - fill with mean
for col in X.columns:
    if X[col].isnull().sum() > 0:
        mean_val = X[col].mean()
        X[col] = X[col].fillna(mean_val)
        X_test_full[col] = X_test_full[col].fillna(mean_val)

print(f"\nFeatures selected: {len(numerical_features)}")
```

**Detailed Explanation:**

- `train_df = pd.read_csv('train.csv')`: Loads the training data from the CSV file named 'train.csv' into a Pandas DataFrame called `train_df`.
- `test_df = pd.read_csv('test.csv')`: Loads the test data from 'test.csv' into `test_df`.
- `print(f"Training data shape: {train_df.shape}")`: Prints the dimensions (rows, columns) of the training DataFrame.
- `print(f"Test data shape: {test_df.shape}")`: Prints the dimensions of the test DataFrame.
- `y = train_df['SalePrice']`: Extracts the 'SalePrice' column from the training data and assigns it to `y`. This is our target variable (label) we want to predict.
- `numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()`: Identifies all columns in the DataFrame that contain numerical data types (integers, floats) and converts their names to a list.
- `numerical_features = [feature for feature in numerical_features if feature not in ['Id', 'SalePrice']]`: List comprehension that filters the list of numerical feature names. It removes 'Id' (an identifier, not a predictor) and 'SalePrice' (the target, to avoid data leakage).
- `X = train_df[numerical_features].copy()`: Creates a new DataFrame `X` containing only the selected numerical features from the training data. `.copy()` ensures we are working on a new object, not a view of the original DataFrame.
- `X_test_full = test_df[numerical_features].copy()`: Creates a corresponding feature set `X_test_full` from the test data, ensuring it has the same columns.
- `for col in X.columns:`: Starts a loop iterating through each column name in the feature DataFrame `X`.
- `if X[col].isnull().sum() > 0:`: Checks if the current column has any missing values (NaNs). `isnull()` creates a boolean mask, and `sum()` counts the Trues.
- `mean_val = X[col].mean()`: Calculates the arithmetic mean (average) of the current column.
- `X[col] = X[col].fillna(mean_val)`: Fills (imputes) the missing values in the column with the calculated mean value.
- `X_test_full[col] = X_test_full[col].fillna(mean_val)`: Applies the *same* mean value from the training set to fill missing values in the test set.
- `print(f"\nFeatures selected: {len(numerical_features)}")`: Prints the total count of features that will be used for training.

## 3. Split and Scale Data

```python
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
```

**Detailed Explanation:**

- `X_train, X_val, y_train, y_val = train_test_split(...)`: Calls the `train_test_split` function to divide our data.
    - `X`: The features.
    - `y`: The target variable.
    - `test_size=0.2`: Specifies that 20% of the data should be reserved for validation (`X_val`, `y_val`).
    - `random_state=42`: Sets a seed for the random number generator, ensuring reproducibility.
- `scaler = StandardScaler()`: Initializes a `StandardScaler` object. This scaler will standardize features to have mean=0 and variance=1.
- `X_train_scaled = scaler.fit_transform(X_train)`: Performs two operations on the training data:
    1. `fit`: Computes the mean and standard deviation of each feature in `X_train`.
    2. `transform`: Uses those calculated parameters to scale `X_train`.
- `X_val_scaled = scaler.transform(X_val)`: Scales the validation set using the *already computed* mean and standard deviation from the training set.
- `print(f"Training set size: {X_train.shape[0]}")`: Prints the number of rows (samples) in the training set.
- `print(f"Validation set size: {X_val.shape[0]}")`: Prints the number of rows in the validation set.

## 4. Train Models

### 4.1 Random Forest Regressor

```python
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

# Predictions
rf_pred_train = rf_model.predict(X_train)
rf_pred_val = rf_model.predict(X_val)

# Calculate metrics
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_pred_train))
rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_pred_val))
rf_train_r2 = r2_score(y_train, rf_pred_train)
rf_val_r2 = r2_score(y_val, rf_pred_val)
```

**Detailed Explanation:**

- `rf_model = RandomForestRegressor(...)`: Initializes the Random Forest model with specific hyperparameters:
    - `n_estimators=200`: The number of trees in the forest.
    - `max_depth=15`: The maximum depth of each tree.
    - `min_samples_split=5`: The minimum number of samples required to split an internal node.
    - `min_samples_leaf=2`: The minimum number of samples required to be at a leaf node.
    - `random_state=42`: Seed for reproducibility.
    - `n_jobs=-1`: Uses all available CPU cores.
    - `verbose=1`: Output log messages during training.
- `rf_model.fit(X_train, y_train)`: Trains the Random Forest model on the training features and labels.
- `rf_pred_train = rf_model.predict(X_train)`: Predicts housing prices on the training set.
- `rf_pred_val = rf_model.predict(X_val)`: Predicts housing prices on the validation set.
- `rf_train_rmse = ...`: Calculates Root Mean Squared Error (RMSE) for the training set.
- `rf_val_rmse = ...`: Calculates RMSE for the validation set.
- `rf_train_r2 = ...`: Calculates R-squared for the training set.
- `rf_val_r2 = ...`: Calculates R-squared for the validation set.

### 4.2 XGBoost

```python
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

**Detailed Explanation:**

- `xgb_model = xgb.XGBRegressor(...)`: Initializes the XGBoost regressor.
    - `max_depth=6`: Maximum tree depth.
    - `learning_rate=0.1`: Step size shrinkage.
    - `subsample=0.8`, `colsample_bytree=0.8`: Fraction of data/columns used per tree.
    - `verbosity=1`: Logging level.
- `xgb_model.fit(...)`: Trains the model.
    - `eval_set=[(X_val, y_val)]`: Validation set for evaluation during training.
    - `verbose=False`: Suppresses training logs.

### 4.3 CatBoost

```python
cb_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='RMSE',
    random_seed=42,
    verbose=0
)

cb_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False
)
```

**Detailed Explanation:**

- `cb_model = CatBoostRegressor(...)`: Initializes the CatBoost regressor.
    - `iterations=500`: Max trees.
    - `learning_rate=0.05`: Learning speed.
    - `eval_metric='RMSE'`: Optimization metric.
- `cb_model.fit(...)`: Trains the model.
    - `early_stopping_rounds=50`: Stops early if validation score doesn't improve.

### 4.4 Artificial Neural Network (Keras)

```python
# Build model
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

ann_model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train
history = ann_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    verbose=0
)
```

**Detailed Explanation:**

- `ann_model = Sequential([...])`: Initializes a sequential neural network.
- `Dense(128, activation='relu', ...)`: First hidden layer with 128 neurons and ReLU activation.
- `Dropout(0.2)`: Regularization later, drops 20% of neurons.
- `Dense(64, ...)`: Second hidden layer.
- `Dense(32, ...)`: Third hidden layer.
- `Dense(1)`: Output layer with 1 neuron (for regression).
- `ann_model.compile(...)`: Configures training.
    - `optimizer=Adam`: Optimization algorithm.
    - `loss='mean_squared_error'`: Loss function.
- `history = ann_model.fit(...)`: Trains the model.
    - Uses `X_train_scaled` (important!).
    - `epochs=100`: Number of training passes.
    - `batch_size=32`: Samples per gradient update.

### 4.5 Linear Regression

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```

**Detailed Explanation:**

- `lr_model = LinearRegression()`: Initializes the Linear Regression model.
- `lr_model.fit(X_train, y_train)`: Fits the model to training data.

## 5. Model Comparison

```python
# Create comparison DataFrame
comparison_data = [
    {
        'Model': 'Random Forest',
        'Train RMSE': rf_train_rmse,
        # ...
    },
    # ...
]

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('Val RMSE')
```

**Detailed Explanation:**

- `comparison_data`: List of dictionaries storing model metrics.
- `df_comparison = pd.DataFrame(comparison_data)`: Creates a DataFrame from the metrics.
- `df_comparison.sort_values('Val RMSE')`: Sorts models by Validation RMSE (best at top).

## 6. Visualize Model Performance

```python
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Val RMSE', data=df_comparison, palette='viridis')
plt.title('Validation RMSE (Lower is Better)')
plt.xlabel('Model')
plt.ylabel('RMSE ($)')
plt.xticks(rotation=45)
plt.show()
```

**Detailed Explanation:**

- `plt.figure(...)`: Sets figure size.
- `sns.barplot(...)`: Plots a bar chart of Validation RMSE for each model.
- `plt.title`, `plt.xlabel`, etc.: Adds labels and title.
- `plt.show()`: Displays the plot.

## 7. Feature Importance Analysis

```python
models_dict = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'CatBoost': cb_model
}

for model_name, model in models_dict.items():
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': numerical_features,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(10)
    
    print(f"\n{model_name}:")
    for i, row in importance_df.iterrows():
        print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")
```

**Detailed Explanation:**

- `models_dict`: Dictionary of tree-based models.
- Loop iterates through models, extracts `feature_importances_`.
- Creates a DataFrame, sorts by importance, and prints the top 10 features for each model.

## 8. Summary Table

```python
# Display nice summary table
summary_df = df_comparison.copy()
summary_df['Val RMSE'] = summary_df['Val RMSE'].apply(lambda x: f"${x:,.2f}")
summary_df['Train RMSE'] = summary_df['Train RMSE'].apply(lambda x: f"${x:,.2f}")
# ... (formatting other columns)
print(summary_df)
```

**Detailed Explanation:**

- `summary_df = df_comparison.copy()`: Creates a copy of the comparison DataFrame to modify for display without affecting the original data.
- `summary_df['Val RMSE'] = ...`: Applies a lambda function to format the RMSE values as currency strings (e.g., "$25,000.00").
- `print(summary_df)`: Prints the formatted summary table to the console.
