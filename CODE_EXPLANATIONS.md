# Cell-by-Cell Explanation: Housing Prices Model

This document provides a line-by-line and cell-by-cell breakdown of the implementation in `FinalModel.ipynb`.

## Cell 3: Import Libraries
**Type**: Code
- `import pandas as pd`: Imports the Pandas library (as `pd`) for tabular data handling.
- `import numpy as np`: Imports NumPy for numerical computations and array operations.
- `import matplotlib.pyplot as plt` & `import seaborn as sns`: Standard tools for creating charts and heatmaps.
- `from sklearn.linear_model import LinearRegression`: Baseline model for simple regression.
- `from sklearn.model_selection import train_test_split`: Function to divide data into training and testing sets.
- `from sklearn.preprocessing import StandardScaler`: Tool to normalize features to a similar scale.
- `from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error`: Functions to evaluate how well the models perform.
- `from sklearn.ensemble import RandomForestRegressor`: Regression model based on a collection of decision trees.
- `import xgboost as xgb`: High-performance toolkit for tree boosting.
- `import tensorflow as tf`: Google's framework for building neural networks.
- `import warnings; warnings.filterwarnings('ignore')`: Suppresses non-critical warning messages for cleaner output.

---

## Cell 4: Load and Prepare Data
**Type**: Code
- `train_df = pd.read_csv('train.csv')`: Loads the training dataset.
- `test_df = pd.read_csv('test.csv')`: Loads the test dataset (used for final unseen predictions).
- `y = train_df['SalePrice']`: Extracts the target column (the price we want to predict).
- `numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()`: Identifies all columns that contain numbers (ignoring text/categories for now).
- `numerical_features = [feature for ... if feature not in ['Id', 'SalePrice']]`: Removes administrative IDs and the target label from the list of inputs.
- `X = train_df[numerical_features].copy()`: Creates the input feature matrix from training data.
- **Handling Missing Values Loop**:
    - `X[col].median()`: Calculates the middle value of a column.
    - `X[col].fillna(median_val)`: Replaces any "NaN" (Not a Number/Missing) values with that median so the model doesn't error out.

---

## Cell 5: Split and Scale Data
**Type**: Code
- `train_test_split(X, y, test_size=0.2, random_state=42)`: Sets aside 20% of the data to validate the model's accuracy. `random_state=42` ensures the split is the same every time you run it.
- `scaler = StandardScaler()`: Initializes the scaler.
- `scaler.fit_transform(X_train)`: Learns the mean/variance of the training data and adjusts it.
- `scaler.transform(X_val)`: Adjusts the validation data using the training data's "rules" to keep them consistent.

---

## Cell 6: Random Forest Regressor
**Type**: Code
- `RandomForestRegressor(...)`: Creates the model.
    - `n_estimators=200`: Builds 200 different trees.
    - `max_depth=15`: Limits how deep each tree can grow to prevent them from "memorizing" specific data (overfitting).
- `rf_model.fit(X_train, y_train)`: Starts the training process.
- `rf_model.predict(X_val)`: Asks the model to guess prices for the validation set.
- `np.sqrt(mean_squared_error(...))`: Calculates the error in the original unit (dollars).

---

## Cell 7: XGBoost
**Type**: Code
- `xgb.XGBRegressor(...)`: Sets up the Gradient Boosting model.
    - `learning_rate=0.1`: Controls how quickly the model updates its logic; smaller steps are often more stable.
    - `subsample=0.8`: Controls what percentage of data the model looks at for each tree, adding randomness to prevent overfitting.
- `xgb_model.fit(...)`: Trains the model. The `eval_set` allows the model to see how it's doing on validation data during the training process.

---

## Cell 8: Artificial Neural Network (Keras)
**Type**: Code
- `Sequential([...])`: Defines the layers of the brain in order.
    - `Dense(128, activation='relu')`: 128 "neurons" that look for patterns.
    - `Dropout(0.2)`: Randomly turns off 20% of neurons to make the model more robust.
- `ann_model.compile(...)`: Sets the "Adam" optimizer and loss function (MSE) before training.
- `ann_model.fit(..., epochs=100)`: Passes through the data 100 times to learn the patterns.

---

## Cell 9: Linear Regression
**Type**: Code
- `lr_model = LinearRegression()`: Fits a straight line to the data. This is the simplest possible model to see if our complex ones are actually better.

---

## Cell 10-13: Results & Visualization
**Type**: Code
- **Comparison Table**: Collects RMSE, MAE, and R² from all models into one Pandas table for easy reading.
- `df_comparison.sort_values('Val RMSE')`: Automatically sorts the models to show the winner at the top.
- **Bar Plot**: Uses `sns.barplot` to visually compare the error (RMSE) of each algorithm.
- **Feature Importance**: Uses `model.feature_importances_` to show which house features (like `OverallQual`) matter the most to the winning models.
