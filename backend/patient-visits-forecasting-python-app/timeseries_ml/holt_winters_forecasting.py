import pandas as pd
import numpy as np

from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


class HoltWintersForecaster:

    def __init__(
        self,
        dataframe,
        target_column="visits",
        train_size=0.8,
        trend="add",
        seasonal="add",
        seasonal_periods=7
    ):

        self.df = dataframe.copy()

        self.target_column = target_column

        self.train_size = train_size

        self.trend = trend

        self.seasonal = seasonal

        self.seasonal_periods = seasonal_periods

    def fit_predict(self):

        split_index = int(
            len(self.df) *
            self.train_size
        )

        train = self.df[
            self.target_column
        ].iloc[:split_index]

        test = self.df[
            self.target_column
        ].iloc[split_index:]

        self.model = (
            ExponentialSmoothing(
                train,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated"
            )
            .fit(
                optimized=True
            )
        )

        self.predictions = (
            self.model.forecast(
                len(test)
            )
        )

        self.actual = test

        return self.predictions

    def evaluate(self):

        mae = mean_absolute_error(
            self.actual,
            self.predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                self.actual,
                self.predictions
            )
        )

        mape = (
            mean_absolute_percentage_error(
                self.actual,
                self.predictions
            ) * 100
        )

        r2 = r2_score(
            self.actual,
            self.predictions
        )

        return {

            "Model":
                "Holt-Winters",

            "MAE":
                round(float(mae), 4),

            "RMSE":
                round(float(rmse), 4),

            "MAPE (%)":
                round(float(mape), 4),

            "R2 Score":
                round(float(r2), 4)
        }
    
df = pd.read_csv(
    "data/patient_visits.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.sort_values(
    "date"
)

model = HoltWintersForecaster(
    dataframe=df,
    target_column="visits",
    seasonal_periods=7
)

model.fit_predict()

results = model.evaluate()

print("\nHolt-Winters Metrics: ")
print(results)

#Forecast Next 30 Days
print("Forecast Next 30 Days")
future_forecast = (
    model.model.forecast(
        steps=30
    )
)

print(future_forecast)

#Plot Actual vs Forecast
import matplotlib.pyplot as plt

plt.figure(
    figsize=(16,6)
)

plt.plot(
    model.actual.index,
    model.actual,
    label="Actual"
)

plt.plot(
    model.actual.index,
    model.predictions,
    label="Holt-Winters Forecast"
)

plt.title(
    "Holt-Winters Forecast"
)

plt.xlabel("Time")

plt.ylabel("Patient Count")

plt.legend()

plt.grid(True)

plt.show()
plt.savefig(
    "saved_artifacts/hw_actual_predict.png"
)
plt.close()


#Plotly Visualization
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=model.actual.index,
        y=model.actual,
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=model.actual.index,
        y=model.predictions,
        name="Holt-Winters Forecast"
    )
)

fig.update_layout(
    title="Actual vs Holt-Winters Forecast",
    xaxis_title="Date",
    yaxis_title="Patient Count",
    template="plotly_white"
)

fig.show()

fig.write_image(
    "saved_artifacts/holt_winters_plotly_forecast.png",
    width=1600,
    height=800
)

#fig.close()