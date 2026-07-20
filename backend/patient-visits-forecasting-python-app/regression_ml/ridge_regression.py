import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv("data/patient_visits.csv")

print("Dataset Shape:", df.shape)

# ==================================================
# Target Variable
# ==================================================

TARGET = "visits"

# ==================================================
# Date Feature Engineering
# ==================================================

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter

    df.drop(columns=["date"], inplace=True)

# ==================================================
# Features and Target
# ==================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeatures:")
print(X.columns.tolist())

# ==================================================
# Train Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# Train Ridge Model
# ==================================================

model = Ridge(
    alpha=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# ==================================================
# Predictions
# ==================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ==================================================
# Metrics Function
# ==================================================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }

# ==================================================
# Results
# ==================================================

train_metrics = evaluate(
    y_train,
    y_train_pred
)

test_metrics = evaluate(
    y_test,
    y_test_pred
)

print("\nTrain Metrics")
print(train_metrics)

print("\nTest Metrics")
print(test_metrics)

# ==================================================
# Coefficients
# ==================================================

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coef_df = coef_df.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Importance")
print(coef_df)

# ==================================================
# Save Model
# ==================================================

joblib.dump(
    model,
    "saved_artifacts/ridge_regression_model.pkl"
)

print("\nModel Saved Successfully")


#Hyperparameter Tuning (Recommended)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()

param_grid = {
    "alpha": [
        0.01,
        0.1,
        1,
        10,
        100,
        1000
    ]
}

grid_search = GridSearchCV(
    estimator=ridge,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(
    "Best Alpha:",
    grid_search.best_params_
)

best_model = grid_search.best_estimator_

#Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_test_pred,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Visits")
plt.ylabel("Predicted Visits")
plt.title("Ridge Regression Predictions")

plt.tight_layout()
plt.show()
plt.savefig(
    "saved_artifacts/ridge_regression_model.png"
)
plt.close


#Forecast New Data
import pandas as pd
import joblib

model = joblib.load(
    "saved_artifacts/ridge_regression_model.pkl"
)

new_data = pd.DataFrame({
    "patient_count": [150],
    "admissions": [35],
    "discharges": [30],
    "occupancy": [78],
    "year": [2026],
    "month": [6],
    "day": [10],
    "day_of_week": [2],
    "week_of_year": [24],
    "quarter": [2]
})

prediction = model.predict(
    new_data
)

print(
    "Predicted Visits:",
    round(prediction[0], 2)
)

#Ridge Regression with Lag Features (Best for Forecasting)
df["lag_1"] = df["visits"].shift(1)
df["lag_7"] = df["visits"].shift(7)
df["lag_30"] = df["visits"].shift(30)

df["rolling_mean_7"] = (
    df["visits"]
    .rolling(7)
    .mean()
)

df["rolling_mean_30"] = (
    df["visits"]
    .rolling(30)
    .mean()
)

df.dropna(inplace=True)

print(df.head())