# gradient_boosting_regressor.py

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
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
# Convert Date Column
# =====================================================
df["date"] = pd.to_datetime(df["date"])

# Time-based features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

# Drop original date
df = df.drop(columns=["date"])


# =====================================================
# Target Variable
# =====================================================
TARGET = "visits"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# =====================================================
# Train Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)


# =====================================================
# Gradient Boosting Regressor
# =====================================================
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
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

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\n===== Gradient Boosting Regressor Metrics =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


# =====================================================
# Feature Importance
# =====================================================
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features:")
print(feature_importance.head(10))


# =====================================================
# Visualization
# =====================================================
plt.figure(figsize=(14, 6))

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

plt.title("Gradient Boosting Regressor Forecast")
plt.xlabel("Time")
plt.ylabel("Visits")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig(
    "saved_artifacts/multiple_regression_model.png"
)
plt.close


# =====================================================
# Future Forecast Example
# =====================================================
future_rows = X.tail(30)

future_predictions = model.predict(future_rows)

future_df = pd.DataFrame({
    "Forecasted_visits": future_predictions
})

print("\nFuture Forecast:")
print(future_df.head())

future_df.to_csv(
    "saved_artifacts/gradient_boosting_forecast.csv",
    index=False
)

print("\nForecast saved as gradient_boosting_forecast.csv")

"""
#Hyperparameter Tuning Version
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
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

