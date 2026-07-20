from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd

class ForecastingInterface(ABC):

    @abstractmethod
    def forecast(
        self,
        horizon: int
    ):
        pass


class ForecastingImplementation(
    ForecastingInterface
):

    def __init__(
        self,
        model
    ):
        self.model = model

    FEATURE_COLUMNS = [
        "patient_count",
        "admissions",
        "discharges",
        "occupancy",
        "year",
        "month",
        "day",
        "dayofweek",
        "quarter",
        "weekofyear"
    ]

    def regression_predict(self, request):

        input_df = pd.DataFrame([{
            "patient_count": request.patient_count,
            "admissions": request.admissions,
            "discharges": request.discharges,
            "occupancy": request.occupancy,
            "year": request.year,
            "month": request.month,
            "day": request.day,
            "dayofweek": request.dayofweek,
            "quarter": request.quarter,
            "weekofyear": request.weekofyear
        }])

        prediction = self.model.predict(input_df)[0]

        return float(prediction)

    def forecast(
        self,
        horizon: int
    ):

        # Statsmodels
        if hasattr(
            self.model,
            "forecast"
        ):

            predictions = self.model.forecast(
                steps=horizon
            )

        # Sklearn
        elif hasattr(
            self.model,
            "predict"
        ):

            future = np.arange(
                horizon
            ).reshape(-1, 1)

            predictions = self.model.predict(
                future
            )

        else:

            raise ValueError(
                "Unsupported forecasting model"
            )

        return list(predictions)