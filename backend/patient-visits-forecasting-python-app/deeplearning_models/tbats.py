import pandas as pd
import numpy as np
from tbats import TBATS
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

# =====================================================
# Target Variable
# =====================================================

TARGET = "visits"

if TARGET not in df.columns:
    raise ValueError(
        f"{TARGET} column not found"
    )

# =====================================================
# Prepare Data
# =====================================================

df[date_col] = pd.to_datetime(df[date_col])

df = df.sort_values(date_col)

series = df[TARGET].values

# =====================================================
# Train/Test Split
# =====================================================

test_size = int(len(series) * 0.20)

train = series[:-test_size]
test = series[-test_size:]

print("Train Size:", len(train))
print("Test Size :", len(test))

# =====================================================
# TBATS Model
# =====================================================

estimator = TBATS(
    use_box_cox=True,
    use_trend=True,
    use_damped_trend=True,
    seasonal_periods=[7, 30]
)

model = estimator.fit(train)

# =====================================================
# Test Forecast
# =====================================================

predictions = model.forecast(
    steps=len(test)
)

# =====================================================
# Evaluation
# =====================================================

mae = mean_absolute_error(
    test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        test,
        predictions
    )
)

r2 = r2_score(
    test,
    predictions
)

mape = (
    np.mean(
        np.abs(
            (test - predictions) / test
        )
    ) * 100
)

print("\n===== TBATS Results =====")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

# =====================================================
# Refit on Full Dataset
# =====================================================

final_model = estimator.fit(series)

# =====================================================
# Forecast Next 30 Days
# =====================================================

future_forecast = final_model.forecast(
    steps=30
)

future_dates = pd.date_range(
    start=df[date_col].max() + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast": future_forecast
})

print("\nNext 30 Day Forecast:")
print(forecast_df.head())

# =====================================================
# Save Forecast
# =====================================================

forecast_df.to_csv(
    "saved_artifacts/tbats_forecast.csv",
    index=False
)

print("\nForecast saved")

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    final_model,
    "saved_artifacts/tbats_model.pkl"
)

print("Model saved")

# =====================================================
# Plotly Visualization
# =====================================================

test_dates = df[date_col].iloc[-test_size:]

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=test,
        mode="lines",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=predictions,
        mode="lines",
        name="Predicted"
    )
)

fig.update_layout(
    title="TBATS Forecasting",
    xaxis_title="Date",
    yaxis_title="Patient Count",
    template="plotly_white"
)

fig.show()

print("\nDone!")