import warnings
warnings.filterwarnings("ignore")

import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.CRITICAL)

from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

os.makedirs(
    "best_saved_artifacts",
    exist_ok=True
)

# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv("../data/patient_visits.csv")

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
print("\nX_train columns:",X_train.columns.tolist())
print("X_test :", X_test.shape)

# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mape = mean_absolute_percentage_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": round(float(rmse),2),
        "MAPE": mape,
        "R2": r2
    }



# --------------------------------------------------
# MLflow Setup
# --------------------------------------------------

mlflow.set_tracking_uri(
    #"http://127.0.0.1/5000"
    "http://localhost:5000"
)

mlflow.set_experiment(
    "Patient_Visits_Forecasting"
)

# =====================================================
# MODELS
# =====================================================

base_estimator = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

models = {

    "DecisionTree":
    DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "ADABOOST": 
    AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=300,
        learning_rate=0.05,
        loss="square",
        random_state=42
    ),

    "ELASTICNET":
    ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=10000,
        random_state=42
    ),

    "EXTRATREE":
    ExtraTreesRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    ),

    "GB":
    GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    ),

    "LASSO":
    Lasso(
        alpha=0.1,
        max_iter=10000,
        random_state=42
    ),

    "MULTIREG":
    LinearRegression(),

    "RF":
    RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),

    "RIDGE":
    Ridge(
        alpha=1.0,
        random_state=42
    )

    
    
}

# =====================================================
# TRAIN ALL MODELS
# =====================================================

import mlflow.statsmodels
import joblib

results = []

best_rmse = float("inf")
best_run_id = None
best_model_name = None
best_model = None

for model_name, model in models.items():

    try:

        with mlflow.start_run(run_name=model_name):

            fitted = model.fit(X_train, y_train)

            forecast = fitted.predict(X_test)

            print("forecast :", forecast.shape)
                

            metrics = evaluate(
                y_test,
                forecast
            )

            print(metrics)

            mlflow.log_param(
                "model_type",
                model_name
            )

            mlflow.log_metric(
                "MAE",
                metrics["MAE"]
            )

            mlflow.log_metric(
                "RMSE",
                metrics["RMSE"]
            )

            mlflow.log_metric(
                "MAPE",
                metrics["MAPE"]
            )

            mlflow.log_metric(
                "R2",
                metrics["R2"]
            )

            mlflow.sklearn.log_model(
                fitted,
                artifact_path="model"
            )

            results.append({
                "Model": model_name,
                **metrics
            })

            print(results)

            print(
                f"{model_name} RMSE={metrics['RMSE']:.4f}"
            )

            if metrics["RMSE"] < best_rmse:

                best_rmse = metrics["RMSE"]

                best_model = fitted

                best_model_name = model_name

                best_run_id = (
                    mlflow.active_run().info.run_id
                )

    except Exception as e:

        print(
            f"{model_name} Failed:"
            f" {e}"
        )

# --------------------------------------------------
# results
# --------------------------------------------------

results = pd.DataFrame(
    results
)

print(results.head())
print(results.columns.tolist())
print(results.shape)

results = results.sort_values(
    'RMSE'
)

print("\nModel Leaderboard")
print(results)

results.to_csv(
    "best_saved_artifacts/model_comparison.csv",
    index=False
)

# --------------------------------------------------
# Best Model
# --------------------------------------------------

best_model_name = results.iloc[0]["Model"]


print("\nBest Model:", best_model_name)
print("Best RMSE:", best_rmse)
print("Run ID:", best_run_id)

# =====================================================
# REGISTER BEST MODEL
# =====================================================

model_uri = (
    f"runs:/{best_run_id}/model"
)

registered_model_name = (
    "PatientVisitsForecasting"
)

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)

print(
    f"Registered Model: "
    f"{registered_model_name}"
)

print(
    f"Version: "
    f"{registered_model.version}"
)

import joblib

joblib.dump(
    best_model,
    f"best_saved_artifacts/{best_model_name}.pkl"
)

print(
    f"Best model saved as "
    f"best_saved_artifacts/{best_model_name}.pkl"
)


