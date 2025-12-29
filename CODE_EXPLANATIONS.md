# FinalModel.ipynb - Code Explanation

## 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
...
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
...
import tensorflow as tf
from tensorflow.keras.models import Sequential
...
```
*   **pandas & numpy**: Used for data manipulation (DataFrames) and numerical operations.
*   **matplotlib & seaborn**: Used for plotting graphs and visualizations.
*   **LinearRegression**: A baseline linear approach.
*   **RandomForestRegressor**: An ensemble method using multiple decision trees.
*   **XGBoost (XGBRegressor)**: A powerful gradient boosting algorithm known for high performance.
*   **TensorFlow/Keras**: Used to build and train the Artificial Neural Network (ANN).
*   **StandardScaler**: Used to normalize features (scale them to mean=0, variance=1) which is critical for Neural Networks.
*   **Metrics**: `mean_squared_error`, `r2_score`, etc., are used to evaluate how well the models perform.

## 2. Load and Prepare Data
```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
...
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_features = [feature for feature in ... if feature not in ['Id', 'SalePrice']]
X = train_df[numerical_features].copy()
```
*   **Data Loading**: Reads the training and testing CSV files.
*   **Feature Selection**: Automatically selects only numeric columns. It explicitly excludes `Id` (not predictive) and `SalePrice` (the target variable).
*   **Handling Missing Values**: Fills any missing data (`NaN`) with the median value of that column. This prevents errors during model training.

## 3. Split and Scale Data
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```
*   **train_test_split**: Splits the training data into two parts:
    *   **Training Set (80%)**: Used to teach the models.
    *   **Validation Set (20%)**: Used to test the models on unseen data.
*   **StandardScaler**: Standardizes the data. `fit_transform` calculates mean/std from training data and scales it. `transform` uses those same stats to scale validation data. This ensures no data leakage.

## 4. Train Models

### 4.1 Random Forest Regressor
```python
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, ...)
rf_model.fit(X_train, y_train)
```
*   **n_estimators=200**: Builds 200 decision trees.
*   **max_depth=15**: Limits tree depth to prevent overfitting.
*   **fit**: Trains the model.

### 4.2 XGBoost
```python
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, ...)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], ...)
```
*   **Gradient Boosting**: Builds trees sequentially, correcting errors of previous trees.
*   **eval_set**: Passes validation data during training to monitor performance (used later for visualization).

### 4.3 Artificial Neural Network (Keras ANN)
```python
ann_model = Sequential([
    Dense(128, activation='relu', ...),
    Dropout(0.2),
    ...
])
ann_model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
history = ann_model.fit(...)
```
*   **Structure**: A deep network with 3 hidden layers (128 -> 64 -> 32 neurons).
*   **Dropout**: Randomly turns off 20% of neurons during training to prevent overfitting.
*   **Optimizer**: Uses Adam, an adaptive learning rate optimization algorithm.
*   **History**: Stores the training metrics (loss) for every epoch, which we plot later.

### 4.4 Linear Regression
```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```
*   **Baseline**: Attempts to fit a straight line (hyperplane) through the data. Used as a baseline to see if complex models are actually adding value.

## 5. Model Comparison
This section gathers `RMSE`, `MAE`, and `R2 Score` for all models into a DataFrame (`df_comparison`) and prints a summary table to identify the best model.
*   **RMSE (Root Mean Squared Error)**: Measures average error magnitude (in dollars). Lower is better.
*   **R2 Score**: Explains variance (0 to 1). Closer to 1 is better.

## 6. Visualize Model Performance (New Dashboard)
This comprehensive visualization section helps diagnose model behavior.

### Subplot 1: Validation RMSE Comparison
*   **What it shows**: A bar chart of RMSE for each model.
*   **Interpretation**: The shortest bar represents the most accurate model on average errors.

### Subplot 2: Validation R² Score Comparison
*   **What it shows**: A bar chart of R-squared scores.
*   **Interpretation**: The tallest bar represents the model that best captures the data's variance.

### Subplot 3: XGBoost Performance Trend
*   **Code**: `xgb_model.evals_result()`
*   **What it shows**: How the RMSE decreased as XGBoost added more trees (iterations).
*   **Interpretation**: A curve that flattens out indicates the model has finished learning. If it goes back up, it's overfitting.

### Subplot 4: Keras ANN Training History
*   **Code**: `history.history['loss']` vs `val_loss`.
*   **What it shows**: The Mean Squared Error (Loss) for both Training and Validation sets over epochs.
*   **Interpretation**:
    *   Lines converging = Good training.
    *   Val loss going up while Train loss down = Overfitting.

### Subplot 5: Learning Efficiency (XGBoost)
*   **Code**: `learning_curve(...)`
*   **What it shows**: R2 scores as we provide the model with 10%, 30%, ..., 100% of the training data.
*   **Interpretation**:
    *   **Low Score on small data**: Model needs more data.
    *   **Gap between Train/Val**: Indicates variance/overfitting. A closing gap suggests more data helps.

### Subplot 6: Performance Ranking
*   **What it shows**: A sorting ranking of models based on Validation RMSE.
*   **Purpose**: Quick visual identifier for the "winner".

## 7. Feature Importance & Gradio App (Top 10 Features)
*   **Feature Importance**: Extracts the top 10 most influential features (e.g., `OverallQual`, `GrLivArea`) from the XGBoost model.
*   **Retraining**: Retrains a lightweight XGBoost model using *only* these 10 features.
*   **Gradio App**: Creates an interactive web UI where you can use sliders to adjust these 10 features and get a real-time price prediction.
