import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

class MovingAverageForecast:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str = "visits",
        window: int = 7
    ):

        self.df = dataframe.copy()

        self.target_column = target_column

        self.window = window

    def fit_predict(self):

        # Moving Average Prediction

        self.df["prediction"] = (
            self.df[self.target_column]
            .rolling(
                window=self.window
            )
            .mean()
            .shift(1)
        )

        self.df = self.df.dropna()

        return self.df

    def evaluate(self):

        y_true = self.df[
            self.target_column
        ]

        y_pred = self.df[
            "prediction"
        ]

        metrics = {

            "Model":
                f"Moving Average ({self.window})",

            "MAE":
                round(
                    float(
                        mean_absolute_error(
                            y_true,
                            y_pred
                        )
                    ),
                    4
                ),

            "RMSE":
                round(
                    float(
                        np.sqrt(
                            mean_squared_error(
                                y_true,
                                y_pred
                            )
                        )
                    ),
                    4
                ),

            "MAPE (%)":
                round(
                    float(
                        mean_absolute_percentage_error(
                            y_true,
                            y_pred
                        ) * 100
                    ),
                    4
                ),

            "R2 Score":
                round(
                    float(
                        r2_score(
                            y_true,
                            y_pred
                        )
                    ),
                    4
                )
        }

        return metrics
    
df = pd.read_csv(
    "data/patient_visits.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

model = MovingAverageForecast(
    dataframe=df,
    target_column="visits",
    window=7
)

predictions_df = (
    model.fit_predict()
)

results = (
    model.evaluate()
)

print("\nMoving Averages Metrics :")
print(results)

#Compare Multiple Windows
results = []

for window in [3, 7, 14, 30]:

    model = MovingAverageForecast(
        dataframe=df,
        window=window
    )

    model.fit_predict()

    results.append(
        model.evaluate()
    )

comparison_df = pd.DataFrame(
    results
)

print("\nComparison DF Metrics: ")
print(
    comparison_df.sort_values(
        "RMSE"
    )
)

#Production Version (Train/Test Split)
window = 7

df = df.sort_values("date")

split_idx = int(
    len(df) * 0.8
)

train = df.iloc[:split_idx]

test = df.iloc[split_idx:].copy()

test["prediction"] = (
    df["visits"]
    .rolling(window)
    .mean()
    .shift(1)
    .loc[test.index]
)

test = test.dropna()

y_true = test["visits"]

y_pred = test["prediction"]

results = {

    "Model":
        "Moving Average (7)",

    "MAE":
        round(
            float(
                mean_absolute_error(
                    y_true,
                    y_pred
                )
            ),
            4
        ),

    "RMSE":
        round(
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred
                    )
                )
            ),
            4
        ),

    "MAPE (%)":
        round(
            float(
                mean_absolute_percentage_error(
                    y_true,
                    y_pred
                ) * 100
            ),
            4
        ),

    "R2 Score":
        round(
            float(
                r2_score(
                    y_true,
                    y_pred
                )
            ),
            4
        )
}

print("\nProduction Version (Train/Test Split) Metrics: ")
print(results)


#Actual vs Forecast Plot
df["ma_7"] = (
    df["visits"]
    .rolling(window=7)
    .mean()
    .shift(1)
)

df["ma_14"] = (
    df["visits"]
    .rolling(14)
    .mean()
)

df["ma_30"] = (
    df["visits"]
    .rolling(30)
    .mean()
)

df = df.dropna()

import matplotlib.pyplot as plt

plot_df = df.tail(180)

plt.figure(figsize=(16,6))

plt.plot(
    plot_df["date"],
    plot_df["visits"],
    label="Actual",
    linewidth=2
)

plt.plot(
    plot_df["date"],
    plot_df["ma_7"],
    label="7-Day Moving Average Forecast",
    linestyle="--",
    linewidth=2
)

plt.plot(
    plot_df["date"],
    plot_df["ma_14"],
    label="MA-14",
    linestyle="-",
    linewidth=2
)

plt.plot(
    plot_df["date"],
    plot_df["ma_30"],
    label="MA-30",
    linestyle=":",
    linewidth=2
)


plt.title(
    "Actual vs Moving Average Forecast"
)

plt.xlabel("Date")

plt.ylabel("Patient Count")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
plt.savefig(
    "saved_artifacts/moving_average.png"
)
plt.close()

#Residual Plot
df["residual"] = (
    df["visits"]
    - df["ma_7"]
)

plt.figure(figsize=(14,5))

plt.plot(
    df["date"],
    df["residual"]
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title(
    "Moving Average Residuals"
)

plt.ylabel(
    "Error"
)

plt.grid(True)

plt.show()
plt.savefig(
    "saved_artifacts/ma_residuals.png"
)
plt.close()

#Actual vs Predicted Scatter Plot
plt.figure(figsize=(8,8))

plt.scatter(
    df["visits"],
    df["ma_7"],
    alpha=0.5
)

min_val = min(
    df["visits"].min(),
    df["ma_7"].min()
)

max_val = max(
    df["visits"].max(),
    df["ma_7"].max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--"
)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title(
    "Actual vs Moving Average Forecast"
)

plt.grid(True)

plt.show()
plt.savefig(
    "saved_artifacts/ma_actual_predict.png"
)
plt.close()

#Plotly Visualization
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "png"

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["visits"],
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["ma_7"],
        name="MA-7 Forecast"
    )
)

fig.update_layout(
    title="Patient Volume Forecast",
    xaxis_title="Date",
    yaxis_title="Patient Count",
    template="plotly_white",
    hovermode="x unified"
)

fig.show()

fig.write_image(
    "saved_artifacts/moving_average_plotly_forecast.png",
    width=1600,
    height=800
)

#fig.close()