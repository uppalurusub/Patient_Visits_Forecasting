import pandas as pd
import numpy as np

class PatientVisitsFeatureEngineer:

    def __init__(
        self,
        dataframe: pd.DataFrame
    ):
        self.df = dataframe.copy()

    def create_features(
        self
    ) -> pd.DataFrame:

        # =====================================
        # Date Conversion
        # =====================================

        self.df["date"] = pd.to_datetime(
            self.df["date"]
        )

        self.df = self.df.sort_values(
            "date"
        )

        # =====================================
        # Calendar Features
        # =====================================

        self.df["day_of_week"] = (
            self.df["date"]
            .dt.dayofweek
        )

        self.df["day_name"] = (
            self.df["date"]
            .dt.day_name()
        )

        self.df["month"] = (
            self.df["date"]
            .dt.month
        )

        self.df["quarter"] = (
            self.df["date"]
            .dt.quarter
        )

        self.df["year"] = (
            self.df["date"]
            .dt.year
        )

        self.df["week_of_year"] = (
            self.df["date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        self.df["day_of_month"] = (
            self.df["date"]
            .dt.day
        )

        self.df["day_of_year"] = (
            self.df["date"]
            .dt.dayofyear
        )

        # =====================================
        # Weekend Flag
        # =====================================

        self.df["is_weekend"] = np.where(
            self.df["day_of_week"] >= 5,
            1,
            0
        )

        # =====================================
        # Holiday Flag
        # =====================================

        try:

            import holidays

            india_holidays = holidays.India()

            self.df["is_holiday"] = (
                self.df["date"]
                .dt.date
                .isin(india_holidays)
                .astype(int)
            )

        except:

            self.df["is_holiday"] = 0

        # =====================================
        # Lag Features
        # =====================================

        self.df["lag_1"] = (
            self.df["patient_count"]
            .shift(1)
        )

        self.df["lag_7"] = (
            self.df["patient_count"]
            .shift(7)
        )

        self.df["lag_14"] = (
            self.df["patient_count"]
            .shift(14)
        )

        self.df["lag_30"] = (
            self.df["patient_count"]
            .shift(30)
        )

        self.df["lag_60"] = (
            self.df["patient_count"]
            .shift(60)
        )

        self.df["lag_90"] = (
            self.df["patient_count"]
            .shift(90)
        )

        # =====================================
        # Rolling Statistics
        # =====================================

        self.df["rolling_mean_7"] = (
            self.df["patient_count"]
            .rolling(7)
            .mean()
        )

        self.df["rolling_mean_30"] = (
            self.df["patient_count"]
            .rolling(30)
            .mean()
        )

        self.df["rolling_mean_90"] = (
            self.df["patient_count"]
            .rolling(90)
            .mean()
        )

        self.df["rolling_std_7"] = (
            self.df["patient_count"]
            .rolling(7)
            .std()
        )

        self.df["rolling_std_30"] = (
            self.df["patient_count"]
            .rolling(30)
            .std()
        )

        # =====================================
        # Exponential Moving Average
        # =====================================

        self.df["ema_7"] = (
            self.df["patient_count"]
            .ewm(span=7)
            .mean()
        )

        self.df["ema_30"] = (
            self.df["patient_count"]
            .ewm(span=30)
            .mean()
        )

        # =====================================
        # Cyclical Features
        # =====================================

        self.df["month_sin"] = np.sin(
            2 * np.pi *
            self.df["month"] / 12
        )

        self.df["month_cos"] = np.cos(
            2 * np.pi *
            self.df["month"] / 12
        )

        self.df["dow_sin"] = np.sin(
            2 * np.pi *
            self.df["day_of_week"] / 7
        )

        self.df["dow_cos"] = np.cos(
            2 * np.pi *
            self.df["day_of_week"] / 7
        )

        # =====================================
        # Trend Feature
        # =====================================

        self.df["time_index"] = np.arange(
            len(self.df)
        )

        # =====================================
        # Remove NaNs
        # =====================================

        self.df = self.df.dropna()

        return self.df