import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/patient_visits.csv")

print("Dataset Shape:", df.shape)

# =====================================================
# Target Variable
# =====================================================

TARGET = "visits"

# =====================================================
# Date Feature Engineering
# =====================================================

if "date" in df.columns:

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter

    df.drop(columns=["date"], inplace=True)

# =====================================================
# Features and Target
# =====================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeatures:")
print(X.columns.tolist())

# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# Feature Scaling
# Lasso requires scaling
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# Train Model
# =====================================================

model = Lasso(
    alpha=0.1,
    max_iter=10000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

# =====================================================
# Predictions
# =====================================================

y_train_pred = model.predict(
    X_train_scaled
)

y_test_pred = model.predict(
    X_test_scaled
)

# =====================================================
# Evaluation Function
# =====================================================

def evaluate_model(y_true, y_pred):

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

# =====================================================
# Metrics
# =====================================================

train_metrics = evaluate_model(
    y_train,
    y_train_pred
)

test_metrics = evaluate_model(
    y_test,
    y_test_pred
)

print("\nTrain Metrics")
print(train_metrics)

print("\nTest Metrics")
print(test_metrics)

# =====================================================
# Feature Coefficients
# =====================================================

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coef_df = coef_df.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Coefficients")
print(coef_df)

# Features selected by Lasso
selected_features = coef_df[
    coef_df["Coefficient"] != 0
]

print("\nSelected Features")
print(selected_features)

# =====================================================
# Save Model and Scaler
# =====================================================

joblib.dump(
    model,
    "saved_artifacts/lasso_regression_model.pkl"
)

joblib.dump(
    scaler,
    "saved_artifacts/lasso_scaler.pkl"
)

print("\nModel Saved Successfully")

#Hyperparameter Tuning
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

param_grid = {
    "alpha": [
        0.0001,
        0.001,
        0.01,
        0.1,
        1,
        10
    ]
}

grid_search = GridSearchCV(
    estimator=Lasso(max_iter=10000),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(
    X_train_scaled,
    y_train
)

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
    "r--"
)

plt.xlabel("Actual Visits")
plt.ylabel("Predicted Visits")
plt.title("Lasso Regression Predictions")

plt.tight_layout()
plt.show()
plt.savefig(
    "saved_artifacts/lasso_regression_model.png"
)
plt.close

#Predict New Data
import pandas as pd
import joblib

model = joblib.load(
    "saved_artifacts/lasso_regression_model.pkl"
)

scaler = joblib.load(
    "saved_artifacts/lasso_scaler.pkl"
)

new_data = pd.DataFrame({
    "patient_count": [150],
    "admissions": [40],
    "discharges": [35],
    "occupancy": [80],
    "year": [2026],
    "month": [6],
    "day": [10],
    "day_of_week": [2],
    "week_of_year": [24],
    "quarter": [2]
})

new_data_scaled = scaler.transform(
    new_data
)

prediction = model.predict(
    new_data_scaled
)

print(
    f"Predicted Visits: {prediction[0]:.2f}"
)

#Time-Series Version (Recommended)
df["lag_1"] = df["visits"].shift(1)
df["lag_7"] = df["visits"].shift(7)
df["lag_14"] = df["visits"].shift(14)

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