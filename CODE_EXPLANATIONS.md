# FinalModel.ipynb - Detailed Code Explanation

This document provides a comprehensive, line-by-line explanation of the code within `FinalModel.ipynb`. This notebook is designed to train, compare, and visualize multiple machine learning regression models for predicting housing prices, culminating in an interactive Grounded AI application.

---

## 1. Import Libraries
This section imports all necessary Python libraries for data handling, plotting, and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error , accuracy_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
```

**Line-by-Line Breakdown:**
*   `import pandas as pd`: Imports Pandas, the primary library for data manipulation and analysis, aliased as `pd`.
*   `import numpy as np`: Imports NumPy for numerical operations (arrays, math functions), aliased as `np`.
*   `import matplotlib.pyplot as plt`: Imports Matplotlib's plotting interface for creating static charts.
*   `import seaborn as sns`: Imports Seaborn, a high-level visualization library built on top of Matplotlib for more attractive statistical graphics.
*   `from sklearn.linear_model import LinearRegression`: Imports the standard Linear Regression algorithm from Scikit-Learn.
*   `from sklearn.model_selection import train_test_split`: Imports the function used to split datasets into training and validation sets.
*   `from sklearn.preprocessing import StandardScaler`: Imports the tool for standardizing features (scaling data to have mean=0 and variance=1).
*   `from sklearn.metrics import ...`: Imports metric functions to evaluate model performance (RMSE, R², MAE).
*   `from sklearn.ensemble import RandomForestRegressor`: Imports the Random Forest algorithm (an ensemble of decision trees).
*   `import xgboost as xgb` & `from xgboost import XGBRegressor`: Imports the XGBoost library, a powerful gradient boosting framework.
*   `import tensorflow as tf`: Imports TensorFlow, the backend for our neural network.
*   `from tensorflow.keras...`: Imports Keras components:
    *   `Sequential`: For building a linear stack of neural network layers.
    *   `Dense`: A standard fully-connected layer.
    *   `Dropout`: A regularization layer to prevent overfitting.
    *   `Adam`: An efficient optimization algorithm for training networks.
*   `warnings.filterwarnings('ignore')`: Suppresses non-critical warnings to keep the output clean.

---

## 2. Load and Prepare Data
This section loads the dataset and prepares it for machine learning by selecting relevant features and handling missing values.

```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
y = train_df['SalePrice']
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_features = [feature for feature in numerical_features if feature not in ['Id', 'SalePrice']]
X = train_df[numerical_features].copy()
X_test_full = test_df[numerical_features].copy()
```

**Line-by-Line Breakdown:**
*   `pd.read_csv('train.csv')`: Loads the training data from a CSV file into a DataFrame `train_df`.
*   `y = train_df['SalePrice']`: Isolates the target variable (the house price we want to predict) into `y`.
*   `select_dtypes(include=np.number)`: Automatically identifies all columns that contain numbers (floats/integers).
*   `numerical_features = [feature for feature in numerical_features if feature not in ['Id', 'SalePrice']]`: Filters the list of numerical columns to remove `Id` (useless for prediction) and `SalePrice` (the target itself, to prevent data leakage).
*   `X = train_df[numerical_features].copy()`: Creates the feature matrix `X` using only the selected numerical columns.

```python
for col in X.columns:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test_full[col] = X_test_full[col].fillna(median_val)
```
*   **Handling Missing Values**: Iterates through every column. If a column has missing values (`NaN`), it calculates the median of that column and fills the empty spots with it. This is a robust way to handle missing data without throwing rows away.

---

## 3. Split and Scale Data
Standardization is crucial for models like Neural Networks and often helps other models converge faster.

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

**Line-by-Line Breakdown:**
*   `train_test_split(X,y,test_size=0.2,random_state=42)`: randomly splits `X` and `y` into:
    *   `X_train`, `y_train`: 80% of data for training.
    *   `X_val`, `y_val`: 20% of data for validation (testing performance).
    *   `random_state=42`: Ensures the split is reproducible (same split every time).
*   `scaler.fit_transform(X_train)`: Computes the mean and std deviation of the *training set only* and scales it.
*   `scaler.transform(X_val)`: Scales the validation set using the *training set's* statistics. This prevents "data leakage" (peeking at the test answer).

---

## 4. Train Models
This section initializes, trains, and evaluates four different regression models.

### 4.1 Random Forest Regressor
```python
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
)
rf_model.fit(X_train, y_train)
```
*   `RandomForestRegressor`: Creates an ensemble model.
    *   `n_estimators=200`: Uses 200 different decision trees.
    *   `max_depth=15`: Limits how deep trees can grow (prevents memorizing the data).
    *   `min_samples_split = 15`: specifies the minimum sample before splitting
    *  `min_samples_leaf = 2`: specifies the minimum sample at the end of a tree
*   `.fit()`: The command that actually trains the model on the data.

### 4.2 XGBoost
```python
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```
*   `XGBRegressor`: Uses Gradient Boosting, where new trees try to correct errors made by previous trees.
    *   `n_estimators=200`: Uses 200 different decision trees.
    *   `learning_rate=0.1`: Controls the step size at each iteration.
    *   `max_depth=15`: Limits how deep trees can grow (prevents memorizing the data).
    *   `min_samples_split = 15`: specifies the minimum sample before splitting
    *  `min_samples_leaf = 2`: specifies the minimum sample at the end of a tree
*   `eval_set=[(X_val, y_val)]`: We pass the validation data during training so the model can track how well it's doing on unseen data after every tree is added.

### 4.3 Artificial Neural Network (Keras)
```python
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
ann_model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
history = ann_model.fit(...)
```
*   `Sequential`: Defines a linear stack of layers.
    *   `Dense(128, activation='relu')`: A hidden layer with 128 neurons using the ReLU activation function (allows learning non-linear patterns).
    *   `Dropout(0.2)`: Randomly deactivates 20% of neurons during each training step. This forces the network to be robust and redundant, preventing overfitting.
    *   `Dense(1)`: The final output layer with 1 neuron (the predicted price).
*   `compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')`: Configures the training process using the Adam optimizer and Mean Squared Error as the target to minimize.
*   `fit(...)` Trains the model on the training data.
*   `epochs=100`: Number of times the model will iterate over the entire training dataset.
*   `batch_size=32`: Number of samples processed before the model's internal parameters are updated.
*   `validation_data=(X_val_scaled, y_val)`: The validation set is used to monitor the model's performance and prevent overfitting.
### 4.4 Linear Regression
```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```
*   `LinearRegression`: Simple baseline model finding the best-fit straight line/hyperplane.

---

## 5. Model Comparison
This section aggregates the performance metrics (RMSE, R², MAE) into a Pandas DataFrame to easily compare "apples with apples".

```python
comparison_data = [ ... ] # List of dictionaries containing metrics for each model
df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('Val RMSE')
```
*   Creates a neat table ranking models by `Val RMSE` (Validation Root Mean Squared Error). The model with the lowest RMSE appears at the top.

---

## 6. Visualize Model Performance
This massive plotting block creates a 6-panel dashboard to diagnose model health.

*   `plt.subplot(3, 2, 1)`: **RMSE Comparison**. A bar chart showing error magnitude. Lower is better.
*   `plt.subplot(3, 2, 2)`: **R² Score Comparison**. Bar chart showing variance explained. Higher is better (closer to 1.0).
*   `plt.subplot(3, 2, 3)`: **XGBoost Performance Trend**. Plots the learning curve over iterations. Flattening curve = converged. Rising curve = overfitting.
*   `plt.subplot(3, 2, 4)`: **Keras ANN Training History**. Plots Train vs Validation Loss. Diverging lines indicate overfitting.
*   `plt.subplot(3, 2, 5)`: **Learning Efficiency**. Shows how model performance improves as we give it more data (0% to 100% of dataset).
*   `plt.subplot(3, 2, 6)`: **Performance Ranking**. Visual summary of the "winner".

---

## 7. Feature Importance & Retraining
To make the model usable in a real-time app, we simplify it by selecting only the most important features.

```python
# Extract feature importances
importance = model.feature_importances_
importance_df = pd.DataFrame({...}).sort_values('Importance', ascending=False).head(10)
```
*   Extracts which columns (e.g., `OveralQual`, `GrLivArea`) the XGBoost model relied on most.

```python
# Retrain on Top 10 Features
xgb_top10 = XGBRegressor(...)
xgb_top10.fit(X_train_top10, y_train)
```
*   Retrains a new, lighter XGBoost model using *only* the top 10 features. This performs almost as well as the full model but is much easier for a user to input data into (10 sliders instead of 80).

---

## 8. Gradio Interface (Interactive App)
This section builds the web-based UI.

```python
import gradio as gr

def predict_house_price(*args):
    # Converts slider inputs into a DataFrame
    input_data = pd.DataFrame([args], columns=top_10_xgb_features)
    # Predicts price
    prediction = xgb_top10.predict(input_data)[0]
    return f"${prediction:,.2f}..."
```
*   **Prediction Function**: Takes inputs from the sliders, formats them, and asks the model for a price.

```python
sliders = [
    gr.Slider(..., label=feature_labels.get(f, f)) for f in top_10_xgb_features
]

with gr.Blocks(...) as demo:
    # Defines the UI Layout (Title, Rows, Columns)
    ...
    predict_btn.click(predict_house_price, inputs=sliders, outputs=[...])

demo.launch(share=True)
```
*   **Gradio UI**:
    *   Automatically creates a slider for each of the top 10 features, scaled to that feature's min/max values in the dataset.
    *   **Buttons**: Connects the "Predict" button to the `predict_house_price` function.
    *   **Launch**: Starts a local web server (and a public share link) so you can actually interact with the model in a browser.
