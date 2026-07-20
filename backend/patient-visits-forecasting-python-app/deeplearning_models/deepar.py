import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np

from gluonts.dataset.pandas import PandasDataset
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import plotly.graph_objects as go
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

date_col = None

for col in [
    "date",
    "Date",
    "datetime",
    "Datetime",
    "timestamp"
]:
    if col in df.columns:
        date_col = col
        break

if date_col is None:
    raise ValueError("Date column not found")

TARGET = "visits"

# =====================================================
# Prepare Data
# =====================================================

df[date_col] = pd.to_datetime(df[date_col])

df = df.sort_values(date_col)

series = df[[date_col, TARGET]].copy()

series.columns = ["ds", "y"]

# =====================================================
# Train Test Split
# =====================================================

test_size = int(len(series) * 0.20)

train_df = series[:-test_size]
test_df = series[-test_size:]

print("Train:", train_df.shape)
print("Test :", test_df.shape)

# =====================================================
# GluonTS Dataset
# =====================================================

train_dataset = PandasDataset.from_long_dataframe(
    pd.DataFrame({
        "item_id": "patients",
        "timestamp": train_df["ds"],
        "target": train_df["y"]
    }),
    item_id="item_id",
    timestamp="timestamp"
)

# =====================================================
# DeepAR Model
# =====================================================

estimator = DeepAREstimator(
    prediction_length=test_size,
    context_length=90,
    freq="D",

    hidden_size=64,
    num_layers=2,

    dropout_rate=0.1,

    distr_output=StudentTOutput(),

    trainer_kwargs={
        "max_epochs": 50
    }
)

predictor = estimator.train(train_dataset)

# =====================================================
# Forecast Test Horizon
# =====================================================

forecast_it, ts_it = make_evaluation_predictions(
    dataset=train_dataset,
    predictor=predictor,
    num_samples=100
)

forecasts = list(forecast_it)

prediction = forecasts[0].mean

prediction = prediction[:test_size]

actual = test_df["y"].values

# =====================================================
# Metrics
# =====================================================

mae = mean_absolute_error(
    actual,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        prediction
    )
)

r2 = r2_score(
    actual,
    prediction
)

mape = np.mean(
    np.abs(
        (actual - prediction) / actual
    )
) * 100

print("\n===== DeepAR Results =====")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

# =====================================================
# Train Final Model
# =====================================================

full_dataset = PandasDataset.from_long_dataframe(
    pd.DataFrame({
        "item_id": "patients",
        "timestamp": series["ds"],
        "target": series["y"]
    }),
    item_id="item_id",
    timestamp="timestamp"
)

final_estimator = DeepAREstimator(
    prediction_length=30,
    context_length=90,
    freq="D",

    hidden_size=64,
    num_layers=2,

    trainer_kwargs={
        "max_epochs": 50
    }
)

final_predictor = final_estimator.train(
    full_dataset
)

# =====================================================
# Future Forecast
# =====================================================

forecast_it, ts_it = make_evaluation_predictions(
    dataset=full_dataset,
    predictor=final_predictor,
    num_samples=100
)

forecast = list(forecast_it)[0]

future_values = forecast.mean[:30]

future_dates = pd.date_range(
    start=series["ds"].max() + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast": future_values
})

print("\n30 Day Forecast")
print(forecast_df.head())

forecast_df.to_csv(
    "saved_artifacts/deepar_forecast.csv",
    index=False
)

# =====================================================
# Plot
# =====================================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=test_df["ds"],
        y=actual,
        mode="lines",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=test_df["ds"],
        y=prediction,
        mode="lines",
        name="Predicted"
    )
)

fig.update_layout(
    title="DeepAR Forecast",
    xaxis_title="Date",
    yaxis_title="Patient Count",
    template="plotly_white"
)

fig.show()

print("\nDone")