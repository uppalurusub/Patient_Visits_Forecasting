import pandas as pd
import numpy as np

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


class SESForecaster:

    def __init__(
        self,
        dataframe,
        target_column="visits",
        train_size=0.8
    ):

        self.df = dataframe.copy()

        self.target_column = target_column

        self.train_size = train_size

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
            SimpleExpSmoothing(
                train
            ).fit(
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
                "Simple Exponential Smoothing",

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

model = SESForecaster(
    dataframe=df,
    target_column="visits"
)

model.fit_predict()

results = model.evaluate()

print("\nSimple Exponential Smothing Metrics: ")
print(results)

future_forecast = model.model.forecast(
    30
)


print("\nForecast next 30 days:")
print(future_forecast)

#Plot Actual vs Predicted
import matplotlib.pyplot as plt

plt.figure(
    figsize=(15, 6)
)

plt.plot(
    model.actual.index,
    model.actual,
    label="Actual"
)

plt.plot(
    model.actual.index,
    model.predictions,
    label="SES Forecast"
)

plt.title(
    "Simple Exponential Smoothing Forecast"
)

plt.legend()

plt.show()
plt.savefig(
    "saved_artifacts/ses_trend.png"
)
plt.close()