# 🏠 Cell-by-Cell Explanation: Housing Prices Model

This document provides a line-by-line and cell-by-cell breakdown of the implementation in `FinalModel.ipynb`.

## 📚 Cell 3: Import Libraries
**Type**: Code
- `import pandas as pd`: 🐼 Imports the Pandas library (as `pd`) for tabular data handling.
- `import numpy as np`: 🔢 Imports NumPy for numerical computations and array operations.
- `import matplotlib.pyplot as plt` & `import seaborn as sns`: 📊 Standard tools for creating charts and heatmaps.
- `from sklearn.linear_model import LinearRegression`: 📈 Baseline model for simple regression.
- `from sklearn.model_selection import train_test_split`: ✂️ Function to divide data into training and testing sets.
- `from sklearn.preprocessing import StandardScaler`: ⚖️ Tool to normalize features to a similar scale.
- `from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error`: 📏 Functions to evaluate how well the models perform.
- `from sklearn.ensemble import RandomForestRegressor`: 🌲 Regression model based on a collection of decision trees.
- `import xgboost as xgb`: 🚀 High-performance toolkit for tree boosting.
- `import tensorflow as tf`: 🤖 Google's framework for building neural networks.
- `from tensorflow.keras.models import Sequential`: 🏗️ Used to build the Artificial Neural Network (ANN).
- `from tensorflow.keras.layers import Dense, Dropout`: 🧱 Used to define the layers of the ANN.
- `from tensorflow.keras.optimizers import Adam`: ⚙️ Used to optimize the ANN.
- `import warnings; warnings.filterwarnings('ignore')`: ⚠️ Suppresses non-critical warning messages for cleaner output.

---

## 📥 Cell 4: Load and Prepare Data
**Type**: Code
- `train_df = pd.read_csv('train.csv')`: 📂 Loads the training dataset.
- `test_df = pd.read_csv('test.csv')`: 📄 Loads the test dataset (used for final unseen predictions).
- `y = train_df['SalePrice']`: 💰 Extracts the target column (the price we want to predict).
- `numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()`: 🔢 Identifies all columns that contain numbers (ignoring text/categories).
- `numerical_features = [feature for ... if feature not in ['Id', 'SalePrice']]`: 🧹 Removes administrative IDs and the target label from the list of inputs.
- `X = train_df[numerical_features].copy()`: 📋 Creates the input feature matrix from training data.
- **Handling Missing Values Loop**: 🔄
    - `for col in X.columns`: Iterates through each column in the dataset.
    - `if X[col].isnull().sum() > 0`: ❓ Checks if there are any missing values in the column.
    - `X[col].median()`: 🎯 Calculates the middle value of a column. its rule is $\frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
    - `X[col].fillna(median_val)`: 🛠️ Replaces any "NaN" (Not a Number/Missing) values with that median so the model doesn't error out.

---

## ⚖️ Cell 5: Split and Scale Data
**Type**: Code
- `train_test_split(X, y, test_size=0.2, random_state=42)`: Sets aside 20% of the data to validate (test) the model's accuracy. `random_state=42` ensures the split is the same every time you run it, the random state is a seed value that ensures the same split every time.
- `scaler = StandardScaler()`: Initializes the scaler, which normalizes the data to a similar scale.
- `scaler.fit_transform(X_train)`: Learns the mean/variance of the training data and adjusts it.
- `scaler.transform(X_val)`: Adjusts the validation data using the training data's "rules" to keep them consistent.

---

## Cell 6: Random Forest Regressor
**Type**: Code
- `RandomForestRegressor(n_estimators=200, max_depth=15,min_samples_split=5, min_samples_leaf=5)`: Creates the model.
    - `n_estimators=200`: Builds 200 different trees.
    - `max_depth=15`: Limits how deep each tree can grow to prevent them from "memorizing" specific data (overfitting).
    - `min_samples_split=5`: The minimum number of samples required to split an internal node.
    - `min_samples_leaf=5`: The minimum number of samples required to be at a leaf node.
    - `random_state=42`: Ensures the same split every time you run it, the random state is a seed value that ensures the same split every time.
- `rf_model.fit(X_train, y_train)`: Starts the training process.
- `rf_model.predict(X_val)`: Asks the model to guess prices for the validation set.
- `np.sqrt(mean_squared_error(...))`: Calculates the error in the original unit (dollars).

---

## Cell 7: XGBoost
**Type**: Code
- `xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)`: Sets up the Gradient Boosting model.
    - `n_estimators=200`: Builds 200 different trees.
    - `max_depth=6`: Limits how deep each tree can grow to prevent them from "memorizing" specific data (overfitting).
    - `learning_rate=0.1`: Controls how quickly the model updates its logic; smaller steps are often more stable.
    - `subsample=0.8`: Controls what percentage of data the model looks at for each tree, adding randomness to prevent overfitting.
    - `colsample_bytree=0.8`: Controls what percentage of features the model looks at for each tree, adding randomness to prevent overfitting.
    - `random_state=42`: Ensures the same split every time you run it, the random state is a seed value that ensures the same split every time.
    - `n_jobs=-1`: Uses all available CPU cores to speed up training.
    - `verbosity = 1`: Controls the level of detail in the output.
- `xgb_model.fit(X_train, y_train)`: Trains the model. The `eval_set` allows the model to see how it's doing on validation data during the training process.
- `xgb_model.predict(X_val)`: Asks the model to guess prices for the validation set.
- `np.sqrt(mean_squared_error(...))`: Calculates the error in the original unit (dollars).

---

## Cell 8: Artificial Neural Network (Keras)
**Type**: Code
- `Sequential([...])`: Defines the layers of the brain in order.
    - `Dense(128, activation='relu')`: 128 "neurons" that look for patterns, relu is a type of activation function that is used to introduce non-linearity into the model. its rule is if the input is greater than 0, the output is the input, otherwise the output is 0.
    - `Dropout(0.2)`: Randomly turns off 20% of neurons to make the model more robust.
    - `Dense(64, activation='relu')`: 64 "neurons" that look for patterns.
    - `Dropout(0.2)`: Randomly turns off 20% of neurons to make the model more robust.
    - `Dense(1)`: 1 "neuron" that looks for patterns.
    - `ann_model.compile(...)`: Sets the "Adam" optimizer and loss function (MSE) before training.
    - `learning_rate = 0.01`: Sets the learning rate to 0.01.
    -`loss = mean_squared_error`: Sets the loss function to mean squared error. its rule is $\frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
    - `ann_model.fit(..., epochs=100)`: Passes through the data 100 times to learn the patterns.
    - `batch_size = 32`: Sets the batch size to 32. a batch is a subset of the data that is used to train the model.
    - `verbose = 0`: Sets the verbosity to 0, which means it won't print anything during training.

---

## Cell 9: Linear Regression
**Type**: Code
- `lr_model = LinearRegression()`: Fits a straight line to the data. This is the simplest possible model to see if our complex ones are actually better.

---

## Cell 10-13: Results & Visualization
**Type**: Code
- **Comparison Table**: Collects RMSE, MAE, and $R^2$ from all models into one Pandas table for easy reading.
- `df_comparison.sort_values('Val RMSE')`: Automatically sorts the models to show the winner at the top.
- **Bar Plot**: Uses `sns.barplot` to visually compare the error (RMSE) of each algorithm.
- **Feature Importance**: Uses `model.feature_importances_` to show which house features (like `OverallQual`) matter the most to the winning models.
## Cell 15: top 10 features
- `for model_name, model in models_dict.items():` loops through each model in the dictionary.
    - `importance = model.feature_importances_`: Gets the - importance of each feature.
    - `importance_df = pd.DataFrame({` creates a DataFrame with the top 10 features and their importance.
        - `'Feature': numerical_features,`: The features.
        - `'Importance': importance`: The importance of each feature.
    }).sort_values('Importance', ascending=False).head(10)`: Sorts the DataFrame by importance and takes the top 10 features.
