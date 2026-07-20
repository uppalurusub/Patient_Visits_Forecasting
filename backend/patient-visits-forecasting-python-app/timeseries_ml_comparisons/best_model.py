import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

import joblib
import os
import joblib
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.CRITICAL)

os.makedirs(
    "best_saved_artifacts",
    exist_ok=True
)


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# =====================================================
# MLFLOW
# =====================================================

mlflow.set_tracking_uri(
    #"http://127.0.0.1/5000"
    "http://localhost:5000"
)

EXPERIMENT_NAME = "Patient_Visits_Forecasting"

mlflow.set_experiment(
    EXPERIMENT_NAME
)

# =====================================================
# METRICS
# =====================================================

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
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }




# =====================================================
# DATA
# =====================================================


df = pd.read_csv("../data/patient_visits.csv") 

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

df.set_index("date", inplace=True)

df = df.asfreq("D")

df["visits"] = df["visits"].interpolate()

series = df["visits"]

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

test_size = int(len(series) * 0.20)

train = series[:-test_size]
test = series[-test_size:]



# =====================================================
# MODELS
# =====================================================
from statsmodels.tsa.ar_model import AutoReg

models = {

    "AR":
    AutoReg(
        train,
        lags=12
    )
}

# =====================================================
# TRAIN ALL MODELS
# =====================================================

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

            fitted = model.fit()

            forecast = fitted.forecast(
                steps=len(test)
            )

            metrics = evaluate(
                test,
                forecast
            )

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

            mlflow.statsmodels.log_model(
                fitted,
                artifact_path="model"
            )

            results.append({
                "Model": model_name,
                **metrics
            })

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


        

            
            

print("ML Flow Completed")



results = pd.DataFrame(
    results
)

results = results.sort_values(
    "RMSE"
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