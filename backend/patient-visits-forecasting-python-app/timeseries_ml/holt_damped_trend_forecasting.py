import pandas as pd
import numpy as np

from statsmodels.tsa.holtwinters import Holt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


class HoltTrendForecaster:

    def __init__(
        self,
        dataframe,
        target_column="visits",
        train_size=0.8,
        damped_trend=False,
        initialization_method=None
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

        self.model = Holt(
            train,
            initialization_method="estimated"
        ).fit(
            optimized=True
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
                "Holt Trend",

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



model = HoltTrendForecaster(
    dataframe=df,
    damped_trend=True,
    initialization_method="estimated",
    target_column="visits"
)


model.fit_predict()

results = model.evaluate()

print("\nHolt Trend Forecasting Metrics: ")
print(results)

future_forecast = model.model.forecast(
    steps=30
)

print("\nForecast Next 30 days: ")
print(future_forecast)

#Actual vs Forecast Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))

plt.plot(
    model.actual.index,
    model.actual,
    label="Actual"
)

plt.plot(
    model.actual.index,
    model.predictions,
    label="Holt Forecast"
)

plt.title(
    "Holt Trend Forecast"
)

plt.xlabel("Time")

plt.ylabel("Patient Count")

plt.legend()

plt.grid(True)

plt.show()
plt.savefig(
    "saved_artifacts/ht_trend.png"
)
plt.close()