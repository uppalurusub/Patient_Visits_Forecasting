import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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
# Features and Target
# =====================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeature Columns:")
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
# Decision Tree Model
# =====================================================

model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

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

# =====================================================
# Metrics
# =====================================================

print("\nTrain Metrics")
print(
    evaluate(
        y_train,
        y_train_pred
    )
)

print("\nTest Metrics")
print(
    evaluate(
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
    "saved_artifacts/decision_tree_regressor.pkl"
)

print("\nModel Saved Successfully")

#Hyperparameter Tuning
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

print(
    "Best Parameters:",
    grid_search.best_params_
)

best_model = grid_search.best_estimator_

#Feature Importance Visualization
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
plt.ylabel("Features")
plt.title("Decision Tree Feature Importance")

plt.tight_layout()
plt.show()
plt.savefig(
    "saved_artifacts/dt_regression_model.png"
)
plt.close

#Actual vs Predicted Visualization
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
plt.title("Decision Tree Regressor")

plt.tight_layout()
plt.show()
plt.savefig(
    "saved_artifacts/dt_actual_predicted.png"
)
plt.close

#Visualize Tree Structure
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    max_depth=3
)

plt.show()
plt.savefig(
    "saved_artifacts/dt_tree.png"
)
plt.close

#Predict New Data
import pandas as pd
import joblib

model = joblib.load(
    "saved_artifacts/decision_tree_regressor.pkl"
)

new_data = pd.DataFrame({
    "patient_count": [150],
    "admissions": [35],
    "discharges": [28],
    "occupancy": [82],
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
    f"Predicted Visits: {prediction[0]:.2f}"
)

#Time-Series Forecasting Version (Recommended)
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