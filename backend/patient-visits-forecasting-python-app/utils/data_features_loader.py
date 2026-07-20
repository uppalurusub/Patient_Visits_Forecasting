import pandas as pd
from utils.patients_visits_feature_engineering import PatientVisitsFeatureEngineer

df = pd.read_csv(
    "data/patient_visits.csv"
)

engineer = (
    PatientVisitsFeatureEngineer(
        df
    )
)

feature_df = (
    engineer.create_features()
)

print("\nOriginal Columns :")
print(df.columns.to_list())


print("\nFeatures Columns :")
print(feature_df.columns.to_list())

print("\n")
print(feature_df.head())