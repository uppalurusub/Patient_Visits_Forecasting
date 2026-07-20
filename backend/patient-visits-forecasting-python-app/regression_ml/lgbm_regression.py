# lightgbm_regressor.py

import pandas as pd
import numpy as np
import joblib

import lightgbm as lgb
#from lightgbm import LGBMRegressor

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

# Drop original date column
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

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)


# =====================================================
# LightGBM Regressor
# =====================================================
model = lgb.LGBMRegressor(
    objective="regression",
    boosting_type="gbdt",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose = -1
)

model.fit(X_train, y_train)


# =====================================================
# Predictions
# =====================================================
y_pred = model.predict(X_test)


# =====================================================
# Metrics
# =====================================================
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

mape = np.mean(
    np.abs((y_test - y_pred) / y_test)
) * 100

print("\n===== LightGBM Results =====")
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
    "lightgbm_regressor.pkl"
)

print("\nModel Saved Successfully")


# =====================================================
# Actual vs Predicted Plot
# =====================================================
plt.figure(figsize=(15, 6))

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

plt.title("LightGBM Forecast")
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
    "saved_artifacts/lightgbm_forecast.csv",
    index=False
)

print("\nForecast saved successfully.")

#LightGBM with Early Stopping (Recommended)
from lightgbm import LGBMRegressor

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    verbose = -1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ]
)


"""
#Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor

param_grid = {
    "n_estimators": [200, 500, 1000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63, 127],
    "max_depth": [4, 6, 8, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

search = RandomizedSearchCV(
    estimator=LGBMRegressor(
        objective="regression",
        random_state=42
    ),
    param_distributions=param_grid,
    n_iter=25,
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
    title="LightGBM Patient Volume Forecast",
    xaxis_title="Observation",
    yaxis_title="Patient Count"
)

fig.show()

#Advanced Time-Series Features (Highly Recommended)
# Lag Features
df["lag_1"] = df["visits"].shift(1)
df["lag_7"] = df["visits"].shift(7)
df["lag_30"] = df["visits"].shift(30)

# Rolling Statistics
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

df["rolling_std_7"] = (
    df["visits"]
    .shift(1)
    .rolling(7)
    .std()
)

df.dropna(inplace=True)

print(df.head())
