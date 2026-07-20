import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Disable oneDNN messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# LOAD DATA
# =====================================================

FILE_PATH = "data/patient_visits.csv"

df = pd.read_csv(FILE_PATH)

print("Dataset Shape:", df.shape)

# =====================================================
# DATE PROCESSING
# =====================================================

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# =====================================================
# TARGET COLUMN
# =====================================================

TARGET = "visits"

FEATURES = [
    col for col in df.columns
    if col not in ["date", TARGET]
]

print("\nFeatures Used:")
print(FEATURES)

# =====================================================
# HANDLE MISSING VALUES
# =====================================================

df = df.ffill().bfill()

# =====================================================
# SCALING
# =====================================================

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(
    df[FEATURES]
)

y_scaled = target_scaler.fit_transform(
    df[[TARGET]]
)

# =====================================================
# CREATE SEQUENCES
# =====================================================

LOOKBACK = 30

def create_sequences(X, y, lookback):

    X_seq = []
    y_seq = []

    for i in range(len(X) - lookback):

        X_seq.append(
            X[i:i+lookback]
        )

        y_seq.append(
            y[i+lookback]
        )

    return (
        np.array(X_seq),
        np.array(y_seq)
    )

X, y = create_sequences(
    X_scaled,
    y_scaled,
    LOOKBACK
)

print("\nSequence Shapes")
print("X:", X.shape)
print("y:", y.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

split_idx = int(len(X) * 0.8)

X_train = X[:split_idx]
X_test = X[split_idx:]

y_train = y[:split_idx]
y_test = y[split_idx:]

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================================================
# BUILD LSTM MODEL
# =====================================================

model = Sequential([

    LSTM(
        units=128,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Dropout(0.2),

    LSTM(
        units=64,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1
    )
])

# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =====================================================
# EARLY STOPPING
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# =====================================================
# TRAIN
# =====================================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =====================================================
# PREDICTION
# =====================================================

y_pred = model.predict(X_test)

# Inverse Transform
y_test_actual = target_scaler.inverse_transform(
    y_test
)

y_pred_actual = target_scaler.inverse_transform(
    y_pred
)

# =====================================================
# METRICS
# =====================================================

mae = mean_absolute_error(
    y_test_actual,
    y_pred_actual
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        y_pred_actual
    )
)

r2 = r2_score(
    y_test_actual,
    y_pred_actual
)

print("\n=========================")
print("LSTM RESULTS")
print("=========================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(14,6))

plt.plot(
    y_test_actual,
    label="Actual"
)

plt.plot(
    y_pred_actual,
    label="Predicted"
)

plt.title(
    "LSTM Forecast - Actual vs Predicted"
)

plt.xlabel("Time")
plt.ylabel("Patient Count")

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# TRAINING HISTORY
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training History")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

model.save(
    "saved_artifacts/lstm_patient_volume.keras"
)

print(
    "\nModel Saved Successfully"
)

#Future Forecasting (Next 30 Days)
future_days = 30

last_sequence = X_scaled[-LOOKBACK:]

future_predictions = []

current_sequence = last_sequence.copy()

for _ in range(future_days):

    prediction = model.predict(
        current_sequence.reshape(
            1,
            LOOKBACK,
            len(FEATURES)
        ),
        verbose=0
    )

    future_predictions.append(
        prediction[0][0]
    )

    next_row = current_sequence[-1].copy()

    current_sequence = np.vstack([
        current_sequence[1:],
        next_row
    ])

future_predictions = target_scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

forecast_df = pd.DataFrame({
    "Forecast_Patient_Count":
        future_predictions.flatten()
})

print(forecast_df)

forecast_df.to_csv(
    "future_patient_forecast.csv",
    index=False
)

