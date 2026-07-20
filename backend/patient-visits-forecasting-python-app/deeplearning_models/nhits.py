
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

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

date_cols = [
    "date",
    "Date",
    "datetime",
    "Datetime",
    "timestamp"
]

date_col = None

for col in date_cols:
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
# NeuralForecast Format
# =====================================================
#
# unique_id : series id
# ds        : datetime
# y         : target
#
# =====================================================





data = pd.DataFrame()

data["unique_id"] = "patients"
data["ds"] = pd.to_datetime(df[date_col])
data["y"] = df[TARGET]

data["unique_id"] = data["unique_id"].fillna("patients")

print(data.head())

print(data.head())

print(data["unique_id"].isna().sum())
print(data["unique_id"].unique())

# =====================================================
# Train Test Split
# =====================================================

test_size = int(len(data) * 0.20)

train_df = data[:-test_size].copy()

test_df = data[-test_size:].copy()

print("\nTrain:", train_df.shape)
print("Test :", test_df.shape)

# =====================================================
# N-HITS Model
# =====================================================

model = NHITS(
    h=test_size,
    input_size=30,
    max_steps=500,
    learning_rate=1e-3,
    batch_size=32,
    scaler_type='standard'
)

print("="*50)
print(df.shape)
print(df.columns.tolist())
print(train_df.shape)
print(train_df.head())
print(train_df.tail())
print(train_df.columns)

print("Nulls:")
print(train_df.isnull().sum())

print("Unique IDs:")
print(train_df["unique_id"].unique())

print("Rows:", len(train_df))
print("="*50)

# =====================================================
# Neural Forecast
# =====================================================

nf = NeuralForecast(
    models=[model],
    freq='D'
)

nf.fit(df=train_df)

# =====================================================
# Forecast Test Set
# =====================================================

forecast = nf.predict()

predictions = forecast["NHITS"].values

actual = test_df["y"].values

# =====================================================
# Metrics
# =====================================================

mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

r2 = r2_score(
    actual,
    predictions
)

mape = (
    np.mean(
        np.abs(
            (actual - predictions) /
            actual
        )
    ) * 100
)

print("\n===== N-HITS RESULTS =====")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

# =====================================================
# Train Full Dataset
# =====================================================

final_model = NHITS(
    h=30,
    input_size=30,
    max_steps=500,
    learning_rate=1e-3,
    batch_size=32,
    scaler_type='standard'
)

final_nf = NeuralForecast(
    models=[final_model],
    freq='D'
)

final_nf.fit(df=data)

# =====================================================
# Future Forecast
# =====================================================

future_forecast = final_nf.predict()

print("\n30-Day Forecast")
print(future_forecast)

future_forecast.to_csv(
    "saved_artifacts/nhits_forecast.csv",
    index=False
)

# =====================================================
# Visualization
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
        y=predictions,
        mode="lines",
        name="Predicted"
    )
)

fig.update_layout(
    title="N-HITS Forecast",
    xaxis_title="Date",
    yaxis_title="Patient Count",
    template="plotly_white"
)

fig.show()

print("\nDone")