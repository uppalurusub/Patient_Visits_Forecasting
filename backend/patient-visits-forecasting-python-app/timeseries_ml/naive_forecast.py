import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
import json

class NaiveForecast:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str = "visits"
    ):
        self.df = dataframe.copy()
        self.target_column = target_column

    def fit_predict(self):

        # ==========================
        # Naive Prediction
        # ==========================

        self.df["prediction"] = (
            self.df[self.target_column]
            .shift(1)
        )

        self.df = self.df.dropna()

        return self.df

    def evaluate(self):

        actual = self.df[
            self.target_column
        ]

        prediction = self.df[
            "prediction"
        ]

        mae = mean_absolute_error(
            actual,
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                prediction
            )
        )

        mape = (
            mean_absolute_percentage_error(
                actual,
                prediction
            ) * 100
        )

        r2 = r2_score(
            actual,
            prediction
        )

        return {

            "Model": "Naive Forecast",

            "MAE": round(float(mae), 4),

            "RMSE": round(float(rmse), 4),

            "MAPE (%)": round(
                float(mape),
                4
            ),

            "R2 Score": round(
                float(r2),
                4
            )
        }
    
df = pd.read_csv(
    "data/patient_visits.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

model = NaiveForecast(
    dataframe=df,
    target_column="visits"
)

predictions_df = (
    model.fit_predict()
)

metrics = model.evaluate()

print("\nNaive Forecast Metrics: ")
print(metrics)

#Matplotlib Visualization
# Plot

# Naive Forecast
df["prediction"] = (
    df["visits"]
    .shift(1)
)

df = df.dropna()

from matplotlib import pyplot as plt

plt.figure(
    figsize=(16, 6)
)

plt.plot(
    df["date"],
    df["visits"],
    label="Actual",
    linewidth=2
)

plt.plot(
    df["date"],
    df["prediction"],
    label="Naive Forecast",
    linestyle="--",
    linewidth=2
)

plt.title(
    "Naive Forecast vs Actual Patient Visits",
    fontsize=14
)

plt.xlabel("Date")

plt.ylabel("Patient Count")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
plt.savefig(
    "saved_artifacts/naive_trend.png"
)
plt.close()