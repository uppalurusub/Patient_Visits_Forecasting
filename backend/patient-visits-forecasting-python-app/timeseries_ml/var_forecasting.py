import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from statsmodels.tsa.vector_ar.var_model import VAR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt
import plotly.graph_objects as go

df = pd.read_csv("data/patient_visits.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

print(df.shape)


print(df.columns)

df.set_index("date", inplace=True)

df = df.asfreq("D")

df = df.interpolate()

print("\nAfter interpolate: ")
print(df.shape)


print(df.columns)



print(df.head())

#Train/Test Split
test_size = int(len(df) * 0.20)

train = df[:-test_size]

test = df[-test_size:]

print("Train Shape:", train.shape)
print("Test Shape :", test.shape)

#Check Stationarity
from statsmodels.tsa.stattools import adfuller

for column in train.columns:

    result = adfuller(
        train[column]
    )

    print(
        f"{column} p-value:",
        result[1]
    )

#Difference Data If Needed
train_diff = train.diff().dropna()

#Train VAR Model
model = VAR(
    train_diff
)

results = model.fit(
    maxlags=15,
    ic="aic"
)

print(results.summary())

#Selected Lag Order
print(
    "Selected Lag:",
    results.k_ar
)

#Forecast
lag_order = results.k_ar

forecast = results.forecast(
    y=train_diff.values[-lag_order:],
    steps=len(test)
)

#Convert Forecast to DataFrame
forecast_df = pd.DataFrame(
    forecast,
    index=test.index,
    columns=test.columns
)

print(forecast_df.head())

#Reverse Differencing
last_values = train.iloc[-1]

forecast_original = forecast_df.cumsum()

forecast_original = (
    forecast_original +
    last_values
)

print(
    forecast_original.head()
)


#Evaluate Each Variable
for column in test.columns:

    mae = mean_absolute_error(
        test[column],
        forecast_original[column]
    )

    rmse = np.sqrt(
        mean_squared_error(
            test[column],
            forecast_original[column]
        )
    )

    r2 = r2_score(
        test[column],
        forecast_original[column]
    )

    print("\n", column)

    print(
        f"MAE  : {mae:.2f}"
    )

    print(
        f"RMSE : {rmse:.2f}"
    )

    print(
        f"R2   : {r2:.4f}"
    )

#Save Forecast
forecast_original.to_csv(
    "saved_artifacts/var_forecast.csv"
)

print(
    "Forecast saved successfully."
)

#Matplotlib Visualization
plt.figure(figsize=(15,6))

plt.plot(
    train.index,
    train["visits"],
    label="Train"
)

plt.plot(
    test.index,
    test["visits"],
    label="Actual"
)

plt.plot(
    forecast_original.index,
    forecast_original["visits"],
    label="Forecast"
)

plt.title(
    "VAR Forecast - visits"
)

plt.legend()

plt.grid(True)

plt.show()

plt.savefig(
    "saved_artifacts/var_forecast.png",
    dpi=300
)

plt.show()
plt.close()


#Plotly Interactive Visualization
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=test.index,
        y=test["visits"],
        mode="lines",
        name="Actual visits"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_original.index,
        y=forecast_original["visits"],
        mode="lines",
        name="Forecast visits"
    )
)

fig.update_layout(
    title="VAR Forecast",
    xaxis_title="Date",
    yaxis_title="visits",
    template="plotly_white",
    height=600
)

fig.show(renderer="browser")

fig.write_html(
    "var_forecast.html"
)

#Automatic Lag Selection
model = VAR(
    train_diff
)

lag_results = model.select_order(
    maxlags=20
)

print(
    lag_results.summary()
)

best_lag = lag_results.aic

print(
    "Best Lag:",
    best_lag
)

#Impulse Response Analysis
irf = results.irf(10)

irf.plot()

plt.savefig(
    "saved_artifacts/impulse_response.png",
    dpi=300
)

plt.show()
plt.close()

#Forecast Future 30 Days
future_steps = 30

future_forecast = results.forecast(
    train_diff.values[-lag_order:],
    steps=future_steps
)

future_dates = pd.date_range(
    start=df.index.max() +
    pd.Timedelta(days=1),
    periods=future_steps,
    freq="D"
)

future_df = pd.DataFrame(
    future_forecast,
    index=future_dates,
    columns=df.columns
)

future_df = (
    future_df.cumsum() +
    df.iloc[-1]
)

print(future_df.head())


"""
#MLflow Tracking
import mlflow

with mlflow.start_run():

    model = VAR(
        train_diff
    )

    results = model.fit(
        maxlags=15,
        ic="aic"
    )

    mlflow.log_param(
        "model_type",
        "VAR"
    )

    mlflow.log_param(
        "lag_order",
        results.k_ar
    )

    mlflow.log_metric(
        "aic",
        results.aic
    )

    print(
        "VAR run logged."
    )
"""