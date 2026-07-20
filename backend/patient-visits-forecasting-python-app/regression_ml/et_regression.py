import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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
    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["quarter"] = df["date"].dt.quarter

    df.drop(columns=["date"], inplace=True)

# =====================================================
# Features & Target
# =====================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeatures:")
print(X.columns.tolist())

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# Extra Trees Model
# =====================================================

model = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

# =====================================================
# Train Model
# =====================================================

model.fit(
    X_train,
    y_train
)

# =====================================================
# Predictions
# =====================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

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

    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred) / y_true
            )
        ) * 100
    )

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4)
    }

# =====================================================
# Results
# =====================================================

print("\nTrain Metrics")
print(
    evaluate_model(
        y_train,
        y_train_pred
    )
)

print("\nTest Metrics")
print(
    evaluate_model(
        y_test,
        y_test_pred
    )
)

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

print("\nFeature Importance")
print(importance_df)

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    model,
    "saved_artifacts/extra_trees_regressor.pkl"
)

print("\nModel Saved Successfully")

"""
#Hyperparameter Tuning
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [
        "sqrt",
        "log2",
        None
    ]
}

search = RandomizedSearchCV(
    estimator=ExtraTreesRegressor(
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=param_grid,
    n_iter=25,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

search.fit(
    X_train,
    y_train
)

print(
    "Best Parameters:",
    search.best_params_
)

best_model = search.best_estimator_
"""

#Feature Importance Plot
import matplotlib.pyplot as plt

importance_df = importance_df.sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Extra Trees Feature Importance")

plt.tight_layout()
plt.show()
plt.savefig(
    "saved_artifacts/et_features.png"
)
plt.close

#Actual vs Predicted Plot
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
    "r--",
    linewidth=2
)

plt.xlabel("Actual Visits")
plt.ylabel("Predicted Visits")
plt.title("Extra Trees Regressor")

plt.tight_layout()
plt.show()
plt.savefig(
    "saved_artifacts/et_actuals_predicted.png"
)
plt.close


#Predict New Data
import pandas as pd
import joblib

model = joblib.load(
    "saved_artifacts/extra_trees_regressor.pkl"
)

new_data = pd.DataFrame({
    "patient_count": [150],
    "admissions": [42],
    "discharges": [36],
    "occupancy": [80],
    "year": [2026],
    "month": [6],
    "day": [8],
    "day_of_week": [0],
    "week_of_year": [24],
    "quarter": [2]
})

prediction = model.predict(
    new_data
)

print(
    f"Predicted Visits: {prediction[0]:.2f}"
)

#Time-Series Forecasting Version (Recommended)
df["lag_1"] = df["visits"].shift(1)
df["lag_7"] = df["visits"].shift(7)
df["lag_14"] = df["visits"].shift(14)
df["lag_30"] = df["visits"].shift(30)

df["rolling_mean_7"] = (
    df["visits"].rolling(7).mean()
)

df["rolling_mean_30"] = (
    df["visits"].rolling(30).mean()
)

df["rolling_std_7"] = (
    df["visits"].rolling(7).std()
)

df.dropna(inplace=True)

print(df.head())