import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import matplotlib.pyplot as plt
import plotly.graph_objects as go

df = pd.read_csv("data/patient_visits.csv")

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.sort_values(
    "date"
)

df.set_index(
    "date",
    inplace=True
)

# Daily frequency
df = df.asfreq("D")

# Fill missing values
df["visits"] = (
    df["visits"]
    .interpolate()
)

series = df["visits"]

print(series.head())
print(series.shape)

#Train/Test Split
test_size = int(
    len(series) * 0.20
)

train = series[:-test_size]

test = series[-test_size:]

print("Train:", len(train))
print("Test :", len(test))

#Build SARIMA Model
model = SARIMAX(
    train,

    order=(1,1,1),

    seasonal_order=(
        1,
        1,
        1,
        7
    ),

    enforce_stationarity=False,

    enforce_invertibility=False
)

results = model.fit(
    maxiter=1000,
    disp=False
)

print(
    results.summary()
)

#Verify Convergence
print(
    "Converged:",
    results.mle_retvals.get(
        "converged",
        False
    )
)

#Forecast
print("\nForecast: ")
forecast = results.forecast(
    steps=len(test)
)

forecast.index = test.index
print(forecast) 
      

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

print("\n===== SARIMA Metrics =====")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}")
print(f"R2   : {r2:.4f}")

#Save Forecast
forecast_df = pd.DataFrame(
    {
        "Actual": test,
        "Forecast": forecast
    }
)

forecast_df.to_csv(
    "saved_artifacts/sarima_forecast.csv"
)

print(
    forecast_df.head()
)

#Matplotlib Visualization
plt.figure(
    figsize=(15,6)
)

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
    label="SARIMA Forecast"
)

plt.title(
    "SARIMA Forecast"
)

plt.xlabel("Date")

plt.ylabel("Visits")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "saved_artifacts/sarima_forecast.png",
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
    title="SARIMA Forecast",
    xaxis_title="Date",
    yaxis_title="Patient Visits",
    template="plotly_white",
    height=600
)

fig.show(
    renderer="browser"
)

fig.write_html(
    "sarima_forecast.html"
)

"""
#Automatic SARIMA Parameter Search
best_aic = np.inf
best_order = None
best_seasonal = None
best_model = None

for p in range(3):
    for d in range(2):
        for q in range(3):

            for P in range(2):
                for D in range(2):
                    for Q in range(2):

                        try:

                            model = SARIMAX(
                                train,

                                order=(p,d,q),

                                seasonal_order=(
                                    P,
                                    D,
                                    Q,
                                    7
                                ),

                                enforce_stationarity=False,

                                enforce_invertibility=False
                            )

                            result = model.fit(
                                disp=False
                            )

                            if not result.mle_retvals.get(
                                "converged",
                                False
                            ):
                                continue

                            if result.aic < best_aic:

                                best_aic = result.aic

                                best_order = (
                                    p,d,q
                                )

                                best_seasonal = (
                                    P,D,Q,7
                                )

                                best_model = result

                        except:
                            pass

print(
    "Best Order:",
    best_order
)

print(
    "Best Seasonal:",
    best_seasonal
)

print(
    "Best AIC:",
    best_aic
)

#Walk-Forward Validation
history = list(train)

predictions = []

for t in range(len(test)):

    model = SARIMAX(
        history,

        order=(1,1,1),

        seasonal_order=(
            1,
            1,
            1,
            7
        ),

        enforce_stationarity=False,

        enforce_invertibility=False
    )

    result = model.fit(
        disp=False
    )

    pred = result.forecast()[0]

    predictions.append(pred)

    history.append(
        test.iloc[t]
    )

predictions = pd.Series(
    predictions,
    index=test.index
)

#MLflow Tracking
import mlflow

with mlflow.start_run():

    model = SARIMAX(
        train,

        order=(1,1,1),

        seasonal_order=(
            1,
            1,
            1,
            7
        )
    )

    result = model.fit(
        disp=False
    )

    forecast = result.forecast(
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
        "SARIMA"
    )

    mlflow.log_param(
        "order",
        "(1,1,1)"
    )

    mlflow.log_param(
        "seasonal_order",
        "(1,1,1,7)"
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

    print(
        "SARIMA Run Logged"
    )
"""