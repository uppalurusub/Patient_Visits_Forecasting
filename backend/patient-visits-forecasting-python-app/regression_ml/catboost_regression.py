# catboost_regressor.py

import pandas as pd
import numpy as np
import joblib

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv("data/patient_visits.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# =====================================================
# Date Feature Engineering
# =====================================================
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["dayofyear"] = df["date"].dt.dayofyear

df.drop(columns=["date"], inplace=True)


# =====================================================
# Target Variable
# =====================================================
TARGET = "visits"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# =====================================================
# Time Series Split
# =====================================================
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)


# =====================================================
# CatBoost Regressor
# =====================================================
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)


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

print("\n===== CatBoost Results =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")


# =====================================================
# Feature Importance
# =====================================================
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.get_feature_importance()
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
    "saved_artifacts/catboost_regressor.pkl"
)

print("\nModel Saved Successfully")


# =====================================================
# Visualization
# =====================================================
plt.figure(figsize=(15,6))

plt.plot(
    y_test.values,
    label="Actual",
    linewidth=2
)

plt.plot(
    y_pred,
    label="Predicted",
    linewidth=2
)

plt.title("CatBoost Patient Volume Forecast")
plt.xlabel("Observation")
plt.ylabel("Patient Count")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# =====================================================
# Future Forecast
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
    "saved_artifacts/catboost_forecast.csv",
    index=False
)

print("\nForecast saved successfully.")


#CatBoost with Early Stopping
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=42
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=200,
    verbose=100
)

"""
#Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor

param_grid = {
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "iterations": [500, 1000, 1500],
    "l2_leaf_reg": [1, 3, 5, 7, 9]
}

model = CatBoostRegressor(
    loss_function="RMSE",
    verbose=0
)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

print("Best Parameters:")
print(search.best_params_)

best_model = search.best_estimator_
"""

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
    title="CatBoost Forecast",
    xaxis_title="Observation",
    yaxis_title="Patient Count"
)

fig.show()

#Recommended Lag Features for Forecasting
# Lag Features
df["lag_1"] = df["visits"].shift(1)
df["lag_7"] = df["visits"].shift(7)
df["lag_14"] = df["visits"].shift(14)
df["lag_30"] = df["visits"].shift(30)

# Rolling Mean
df["rolling_mean_7"] = (
    df["visits"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["rolling_mean_30"] = (
    df["visits"]
    .shift(1)
    .rolling(30)
    .mean()
)

# Rolling Std
df["rolling_std_7"] = (
    df["visits"]
    .shift(1)
    .rolling(7)
    .std()
)

df.dropna(inplace=True)
print(df.head())