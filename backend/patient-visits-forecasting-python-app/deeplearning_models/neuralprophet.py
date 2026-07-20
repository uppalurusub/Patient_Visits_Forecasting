import warnings
warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import plotly.graph_objects as go
import joblib
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# Load Dataset
# =====================================================

FILE_PATH = "data/patient_visits.csv"

df = pd.read_csv(FILE_PATH)

print("Dataset Shape:", df.shape)

# =====================================================
# Detect Date Column
# =====================================================

possible_date_cols = [
    "date",
    "Date",
    "datetime",
    "Datetime",
    "timestamp"
]

date_col = None

for col in possible_date_cols:
    if col in df.columns:
        date_col = col
        break

if date_col is None:
    raise ValueError("Date column not found")

TARGET = "visits"

if TARGET not in df.columns:
    raise ValueError(
        f"{TARGET} column not found"
    )

# =====================================================
# NeuralProphet Format
# =====================================================

data = df[[date_col, TARGET]].copy()

data.columns = ["ds", "y"]

data["ds"] = pd.to_datetime(data["ds"])

data = data.sort_values("ds")

print(data.head())

# =====================================================
# Train Test Split
# =====================================================

test_size = int(len(data) * 0.20)

train_df = data[:-test_size]
test_df = data[-test_size:]

print("\nTrain Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# =====================================================
# Build NeuralProphet Model
# =====================================================

model = NeuralProphet(
    n_lags=30,
    n_forecasts=1,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    learning_rate=0.01,
    epochs=100
)

# =====================================================
# Train
# =====================================================

metrics = model.fit(
    train_df,
    freq="D"
)

print(metrics.tail())

# =====================================================
# Forecast Test Data
# =====================================================

future = model.make_future_dataframe(
    train_df,
    periods=len(test_df),
    n_historic_predictions=True
)

forecast = model.predict(future)

# =====================================================
# Extract Predictions
# =====================================================

forecast = forecast[["ds", "yhat1"]]

pred_df = forecast.tail(len(test_df))

actual = test_df["y"].values
predicted = pred_df["yhat1"].values

# =====================================================
# Evaluation
# =====================================================

mae = mean_absolute_error(
    actual,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)

r2 = r2_score(
    actual,
    predicted
)

print("\n===== NeuralProphet Results =====")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =====================================================
# Future Forecast (30 Days)
# =====================================================

future_30 = model.make_future_dataframe(
    data,
    periods=30
)

forecast_30 = model.predict(
    future_30
)

print(
    forecast_30[
        ["ds", "yhat1"]
    ].tail(30)
)

# =====================================================
# Save Forecast
# =====================================================

forecast_30.to_csv(
    "saved_artifacts/neuralprophet_forecast.csv",
    index=False
)

print(
    "\nForecast saved as neuralprophet_forecast.csv"
)

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    model,
    "saved_artifacts/neuralprophet_model.pkl"
)

print(
    "Model saved as neuralprophet_model.pkl"
)

# =====================================================
# Plotly Visualization
# =====================================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=train_df["ds"],
        y=train_df["y"],
        mode="lines",
        name="Train"
    )
)

fig.add_trace(
    go.Scatter(
        x=test_df["ds"],
        y=test_df["y"],
        mode="lines",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=pred_df["ds"],
        y=pred_df["yhat1"],
        mode="lines",
        name="Predicted"
    )
)

fig.update_layout(
    title="NeuralProphet Forecast",
    xaxis_title="Date",
    yaxis_title="Patient Count",
    template="plotly_white"
)

fig.show()

print("\nDone!")