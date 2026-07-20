import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import matplotlib.pyplot as plt
import plotly.graph_objects as go


# =====================================================
# Load Data
# =====================================================

df = pd.read_csv("data/patient_visits.csv")

print(df.head())

# Example:
# Date,Visits
# 2023-01-01,120
# 2023-01-02,135

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

df.set_index("date", inplace=True)

df = df.asfreq("D")

target_column = "visits"

series = df[target_column]

print("\nShape:", series.shape)


# =====================================================
# Train Test Split
# =====================================================

test_size = int(len(series) * 0.2)

train = series[:-test_size]
test = series[-test_size:]

print("\nTrain Size:", len(train))
print("Test Size :", len(test))


# =====================================================
# Build Auto Regression Model
# =====================================================

lag_order = 12

model = AutoReg(
    train,
    lags=lag_order,
    old_names=False
)

model_fit = model.fit()

print("\nModel Summary")
print(model_fit.summary())


# =====================================================
# Forecast
# =====================================================

predictions = model_fit.predict(
    start=len(train),
    end=len(train) + len(test) - 1,
    dynamic=False
)

predictions.index = test.index


# =====================================================
# Evaluation Metrics
# =====================================================

mae = mean_absolute_error(
    test,
    predictions
)

mse = mean_squared_error(
    test,
    predictions
)

rmse = np.sqrt(mse)

mape = mean_absolute_percentage_error(
    test,
    predictions
)

r2 = r2_score(
    test,
    predictions
)

print("\n===== Auto Regression Metrics =====")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}")
print(f"R2   : {r2:.4f}")


# =====================================================
# Forecast DataFrame
# =====================================================

forecast_df = pd.DataFrame({
    "Actual": test,
    "Forecast": predictions
})

print("\nForecast Sample")
print(forecast_df.head())

forecast_df.to_csv(
    "autoregression_forecast.csv"
)

print(
    "\nForecast saved as autoregression_forecast.csv"
)

#Matplotlib Visualization
plt.figure(figsize=(14, 6))

plt.plot(
    train.index,
    train,
    label="Train"
)

plt.plot(
    test.index,
    test,
    label="Actual"
)

plt.plot(
    predictions.index,
    predictions,
    label="Forecast"
)

plt.title("Auto Regression Forecast")
plt.xlabel("Date")
plt.ylabel("Visits")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "saved_artifacts/autoregression_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

#Plotly Interactive Visualization
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=train.index,
        y=train.values,
        mode="lines",
        name="Train"
    )
)

fig.add_trace(
    go.Scatter(
        x=test.index,
        y=test.values,
        mode="lines",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=predictions.index,
        y=predictions.values,
        mode="lines",
        name="Forecast"
    )
)

fig.update_layout(
    title="Auto Regression Forecast",
    xaxis_title="Date",
    yaxis_title="Visits",
    template="plotly_white",
    height=600
)

# VS Code Compatible
fig.show(renderer="browser")

# Save HTML
fig.write_html(
    "saved_artifacts/autoregression_forecast.html"
)

print(
    "Plotly visualization saved as autoregression_forecast.html"
)

#Automatic Lag Selection Using AIC
best_aic = np.inf
best_lag = None
best_model = None

for lag in range(1, 25):

    try:
        model = AutoReg(
            train,
            lags=lag,
            old_names=False
        )

        result = model.fit()

        if result.aic < best_aic:
            best_aic = result.aic
            best_lag = lag
            best_model = result

    except:
        pass

print("Best Lag :", best_lag)
print("Best AIC :", best_aic)

#Walk-Forward Validation (Recommended for Forecasting)
history = list(train)

predictions = []

for t in range(len(test)):

    model = AutoReg(
        history,
        lags=12,
        old_names=False
    )

    model_fit = model.fit()

    forecast = model_fit.predict(
        start=len(history),
        end=len(history)
    )

    predictions.append(
        forecast[0]
    )

    history.append(
        test.iloc[t]
    )

predictions = pd.Series(
    predictions,
    index=test.index
)

print("\npredictions: ")
print(predictions)