import warnings
warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
)

import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import matplotlib.pyplot as plt
import plotly.graph_objects as go

df = pd.read_csv("data/patient_visits.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

df.set_index("date", inplace=True)

# Set daily frequency
df = df.asfreq("D")

# Fill missing values
df["visits"] = df["visits"].interpolate()

series = df["visits"]

print(series.head())

#Train/Test Split
test_size = int(len(series) * 0.2)

train = series[:-test_size]

test = series[-test_size:]

print("Train:", len(train))
print("Test :", len(test))

#Build ARMA Model - ARMA(5,2)
p = 5
q = 2

model = ARIMA(
    train,
    order=(p, 0, q)
)

model_fit = model.fit()

print(model_fit.summary())

#Forecast
forecast = model_fit.forecast(
    steps=len(test)
)

forecast.index = test.index

#Metrics
mae = mean_absolute_error(
    test,
    forecast
)

mse = mean_squared_error(
    test,
    forecast
)

rmse = np.sqrt(mse)

mape = mean_absolute_percentage_error(
    test,
    forecast
)

r2 = r2_score(
    test,
    forecast
)

print("\n===== ARMA Metrics =====")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}")
print(f"R2   : {r2:.4f}")

#Save Forecast
results = pd.DataFrame({
    "Actual": test,
    "Forecast": forecast
})

results.to_csv(
    "arma_forecast.csv"
)

print(results.head())

#Matplotlib Visualization
plt.figure(figsize=(15,6))

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
    forecast.index,
    forecast,
    label="ARMA Forecast"
)

plt.title("ARMA Forecast")

plt.xlabel("Date")

plt.ylabel("Visits")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "arma_forecast.png",
    dpi=300
)

plt.show()
plt.close()

#Plotly Visualization
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=train.index,
        y=train,
        mode="lines",
        name="Train"
    )
)

fig.add_trace(
    go.Scatter(
        x=test.index,
        y=test,
        mode="lines",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast.index,
        y=forecast,
        mode="lines",
        name="Forecast"
    )
)

fig.update_layout(
    title="ARMA Forecast",
    xaxis_title="Date",
    yaxis_title="Patient Visits",
    template="plotly_white",
    height=600
)

fig.show(renderer="browser")

fig.write_html(
    "arma_forecast.html"
)



print("\nforecast: ")
print(forecast)

#Stationarity Check (Required for ARMA)
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)

print("ADF Statistic :", result[0])
print("P-value       :", result[1])

if result[1] < 0.05:
    print("Series is Stationary")
else:
    print("Series is NOT Stationary")


"""
#Build ARMA Model - ARMA(5,2)
best_aic = float("inf")
best_model = None


for p in range(5):
    for q in range(5):

        try:

            model = ARIMA(
                train,
                order=(p,0,q)
            )

            result = model.fit()

            if not result.mle_retvals.get("converged", False):
                continue

            if result.aic < best_aic:

                best_aic = result.aic
                best_model = result

        except:
            continue


#Forecast
forecast = model.forecast(
    steps=len(test)
)

forecast.index = test.index

#Metrics
mae = mean_absolute_error(
    test,
    forecast
)

mse = mean_squared_error(
    test,
    forecast
)

rmse = np.sqrt(mse)

mape = mean_absolute_percentage_error(
    test,
    forecast
)

r2 = r2_score(
    test,
    forecast
)

print("\n===== ARMA Metrics =====")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}")
print(f"R2   : {r2:.4f}")

#Save Forecast
results = pd.DataFrame({
    "Actual": test,
    "Forecast": forecast
})

results.to_csv(
    "arma_forecast.csv"
)

print(results.head())

#Matplotlib Visualization
plt.figure(figsize=(15,6))

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
    forecast.index,
    forecast,
    label="ARMA Forecast"
)

plt.title("ARMA Forecast")

plt.xlabel("Date")

plt.ylabel("Visits")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "arma_forecast.png",
    dpi=300
)

plt.show()

#Plotly Visualization
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=train.index,
        y=train,
        mode="lines",
        name="Train"
    )
)

fig.add_trace(
    go.Scatter(
        x=test.index,
        y=test,
        mode="lines",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast.index,
        y=forecast,
        mode="lines",
        name="Forecast"
    )
)

fig.update_layout(
    title="ARMA Forecast",
    xaxis_title="Date",
    yaxis_title="Patient Visits",
    template="plotly_white",
    height=600
)

fig.show(renderer="browser")

fig.write_html(
    "arma_forecast.html"
)


#Automatic Selection of Best ARMA(p,q)
best_aic = np.inf
best_order = None
best_model = None

for p in range(0, 6):

    for q in range(0, 6):

        try:

            model = ARIMA(
                train,
                order=(p, 0, q)
            )

            results = model.fit()

            if results.aic < best_aic:

                best_aic = results.aic

                best_order = (p, q)

                best_model = results

        except:
            continue

print("Best ARMA Order:", best_order)
print("Best AIC:", best_aic)

forecast = best_model.forecast(
    steps=len(test)
)

print("\nforecast: ")
print(forecast)

#Stationarity Check (Required for ARMA)
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)

print("ADF Statistic :", result[0])
print("P-value       :", result[1])

if result[1] < 0.05:
    print("Series is Stationary")
else:
    print("Series is NOT Stationary")

#Walk-Forward Validation
history = list(train)

predictions = []

for t in range(len(test)):

    model = ARIMA(
        history,
        order=(5,0,2)
    )

    model_fit = model.fit()

    pred = model_fit.forecast()[0]

    predictions.append(pred)

    history.append(test.iloc[t])

predictions = pd.Series(
    predictions,
    index=test.index
)


#ARMA with MLflow Tracking
import mlflow

with mlflow.start_run():

    model = ARIMA(
        train,
        order=(5,0,2)
    )

    model_fit = model.fit()

    forecast = model_fit.forecast(
        steps=len(test)
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            forecast
        )
    )

    mlflow.log_param(
        "model_type",
        "ARMA"
    )

    mlflow.log_param(
        "p",
        5
    )

    mlflow.log_param(
        "q",
        2
    )

    mlflow.log_metric(
        "rmse",
        rmse
    )

    mlflow.log_metric(
        "mae",
        mean_absolute_error(
            test,
            forecast
        )
    )

    print("Run Logged Successfully")
"""