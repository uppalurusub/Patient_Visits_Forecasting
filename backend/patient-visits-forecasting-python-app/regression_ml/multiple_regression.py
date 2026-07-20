import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/patient_visits.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =====================================================
# Define Target Variable
# =====================================================

TARGET = "visits"

# Drop date if present
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

    # Extract useful date features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek

    df.drop(columns=["date"], inplace=True)

# =====================================================
# Features and Target
# =====================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeature Columns:")
print(X.columns.tolist())

# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# =====================================================
# Train Model
# =====================================================

model = LinearRegression()

model.fit(X_train, y_train)

# =====================================================
# Predictions
# =====================================================

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# =====================================================
# Metrics
# =====================================================

def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

train_metrics = calculate_metrics(
    y_train,
    y_pred_train
)

test_metrics = calculate_metrics(
    y_test,
    y_pred_test
)

print("\nTrain Metrics")
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Metrics")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# =====================================================
# Feature Importance (Coefficients)
# =====================================================

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coef_df = coef_df.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nModel Coefficients")
print(coef_df)

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    model,
    "saved_artifacts/multiple_regression_model.pkl"
)

print("\nModel saved successfully.")


#Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred_test,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Patient Count")
plt.ylabel("Predicted Patient Count")
plt.title("Linear Regression Predictions")

plt.tight_layout()

plt.show()
plt.savefig(
    "saved_artifacts/multiple_regression_model.png"
)
plt.close


#Forecast New Data
import joblib
import pandas as pd

model = joblib.load(
    "saved_artifacts/multiple_regression_model.pkl"
)

new_data = pd.DataFrame({
    "patient_count": [180],
    "admissions": [45],
    "discharges": [40],
    "occupancy": [82],
    "year": [2026],
    "month": [6],
    "day": [15],
    "dayofweek": [0]
})

prediction = model.predict(new_data)

print(
    "Predicted Patient Visits:",
    round(prediction[0], 2)
)

#For Time-Series Forecasting
df["lag_1"] = df["visits"].shift(1)
df["lag_7"] = df["visits"].shift(7)
df["rolling_mean_7"] = df["visits"].rolling(7).mean()

df.dropna(inplace=True)

print(df.head())