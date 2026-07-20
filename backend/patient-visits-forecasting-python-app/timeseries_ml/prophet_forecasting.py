import pandas as pd
import numpy as np
from prophet import Prophet
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
print(df.head())

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
    raise ValueError(
        "No date column found. Expected one of: "
        f"{possible_date_cols}"
    )

# =====================================================
# Target Column
# =====================================================

TARGET = "visits"

if TARGET not in df.columns:
    raise ValueError(
        f"{TARGET} column not found in dataset"
    )

# =====================================================
# Prophet Format
# Prophet requires:
# ds -> datetime column
# y  -> target column
# =====================================================

prophet_df = df[[date_col, TARGET]].copy()

prophet_df.columns = ["ds", "y"]

prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

prophet_df = prophet_df.sort_values("ds")

print("\nPrepared Data:")
print(prophet_df.head())

# =====================================================
# Train/Test Split
# =====================================================

test_size = int(len(prophet_df) * 0.2)

train_df = prophet_df[:-test_size]
test_df = prophet_df[-test_size:]

print("\nTrain Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# =====================================================
# Build Prophet Model
# =====================================================

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

model.fit(train_df)

# =====================================================
# Forecast on Test Period
# =====================================================

future = model.make_future_dataframe(
    periods=len(test_df),
    freq="D"
)

forecast = model.predict(future)

# =====================================================
# Extract Test Predictions
# =====================================================

pred_df = forecast[["ds", "yhat"]].tail(len(test_df))

actual = test_df["y"].values
predicted = pred_df["yhat"].values

# =====================================================
# Evaluation
# =====================================================

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("\n===== Prophet Results =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =====================================================
# Forecast Next 30 Days
# =====================================================

future_30 = model.make_future_dataframe(
    periods=30,
    freq="D"
)

future_forecast = model.predict(future_30)

print("\nNext 30 Days Forecast:")
print(
    future_forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].tail(30)
)

# =====================================================
# Save Forecast
# =====================================================

future_forecast.to_csv(
    "prophet_forecast.csv",
    index=False
)

print("\nForecast saved as prophet_forecast.csv")

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    model,
    "saved_artifacts/prophet_model.pkl"
)

print("Model saved as prophet_model.pkl")

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
        y=pred_df["yhat"],
        mode="lines",
        name="Predicted"
    )
)

fig.update_layout(
    title="Prophet Forecast - Patient Visits",
    xaxis_title="Date",
    yaxis_title="Patient Count",
    template="plotly_white"
)

fig.show()

# =====================================================
# Prophet Components Plot
# =====================================================

from prophet.plot import plot_plotly

fig2 = plot_plotly(model, future_forecast)
fig2.show()

print("\nDone!")