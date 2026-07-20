# xgboost_regressor.py

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv("data/patient_visits.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# =====================================================
# Date Features
# =====================================================
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

# Drop date column
df.drop(columns=["date"], inplace=True)


# =====================================================
# Target Variable
# =====================================================
TARGET = "visits"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# =====================================================
# Train/Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# =====================================================
# XGBoost Regressor
# =====================================================
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# =====================================================
# Predictions
# =====================================================
y_pred = model.predict(X_test)


# =====================================================
# Evaluation Metrics
# =====================================================
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

mape = (
    np.mean(
        np.abs((y_test - y_pred) / y_test)
    ) * 100
)

print("\n===== XGBoost Regressor Results =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")


# =====================================================
# Feature Importance
# =====================================================
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features")
print(importance_df.head(15))


# =====================================================
# Save Model
# =====================================================
joblib.dump(
    model,
    "xgboost_regressor.pkl"
)

print("\nModel saved successfully.")


# =====================================================
# Visualization
# =====================================================
plt.figure(figsize=(14,6))

plt.plot(
    y_test.values,
    label="Actual"
)

plt.plot(
    y_pred,
    label="Predicted"
)

plt.title("XGBoost Patient Volume Forecast")
plt.xlabel("Observation")
plt.ylabel("Patient Count")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# =====================================================
# Future Forecast Example
# =====================================================
future_data = X.tail(30)

future_predictions = model.predict(
    future_data
)

forecast_df = pd.DataFrame({
    "Forecasted_visits": future_predictions
})

print("\nFuture Forecast")
print(forecast_df.head())

forecast_df.to_csv(
    "saved_artifacts/xgboost_forecast.csv",
    index=False
)

print("\nForecast saved as xgboost_forecast.csv")


"""
#Hyperparameter Tuning using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

param_grid = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3],
    "min_child_weight": [1, 3, 5, 7]
}

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=30,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

print("Best Parameters")
print(search.best_params_)

best_model = search.best_estimator_


"""

#Time-Series Version (Recommended)
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


#Plotly Visualization
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=y_test,
        mode="lines",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        y=y_pred,
        mode="lines",
        name="Predicted"
    )
)

fig.update_layout(
    title="XGBoost Forecast",
    xaxis_title="Observation",
    yaxis_title="Patient Count"
)

fig.show()