import pandas as pd

df = pd.read_csv("data/patient_visits.csv")

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

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

TARGET = "visits"

X = df[FEATURE_COLUMNS]
y = df[TARGET]