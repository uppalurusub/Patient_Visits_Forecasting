# adaboost_regressor.py

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
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
# Target Column
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
    test_size=0.2,
    shuffle=False
)

print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)


# =====================================================
# Base Estimator
# =====================================================
base_estimator = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)


# =====================================================
# AdaBoost Regressor
# =====================================================
model = AdaBoostRegressor(
    estimator=base_estimator,
    n_estimators=300,
    learning_rate=0.05,
    loss="square",
    random_state=42
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

print("\n===== AdaBoost Regressor Results =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


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
print(importance_df.head(10))


# =====================================================
# Save Model
# =====================================================
joblib.dump(
    model,
    "adaboost_regressor.pkl"
)

print("\nModel saved successfully.")


# =====================================================
# Visualization
# =====================================================
plt.figure(figsize=(14,6))

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

plt.title("AdaBoost Regressor Forecast")
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

print("\nFuture Forecast:")
print(forecast_df.head())

forecast_df.to_csv(
    "saved_artifacts/adaboost_forecast.csv",
    index=False
)

print("\nForecast saved successfully.")


"""
#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.5],
    "loss": ["linear", "square", "exponential"]
}

base_tree = DecisionTreeRegressor(max_depth=5)

grid = GridSearchCV(
    AdaBoostRegressor(
        estimator=base_tree,
        random_state=42
    ),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:")
print(grid.best_params_)

best_model = grid.best_estimator_

"""