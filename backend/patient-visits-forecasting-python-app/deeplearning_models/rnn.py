import warnings
warnings.filterwarnings("ignore")

import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Disable oneDNN messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"




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
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/patient_visits.csv")

print(df.head())
print(df.shape)

# =====================================================
# DATE PROCESSING
# =====================================================

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# =====================================================
# FEATURE SELECTION
# =====================================================

TARGET = "visits"

FEATURES = [
    col for col in df.columns
    if col not in ["date", TARGET]
]

print("\nFeatures:")
print(FEATURES)

# =====================================================
# HANDLE MISSING VALUES
# =====================================================

df = df.fillna(method="ffill")
df = df.fillna(method="bfill")

# =====================================================
# SCALING
# =====================================================

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(df[FEATURES])

y_scaled = target_scaler.fit_transform(
    df[[TARGET]]
)

# =====================================================
# CREATE SEQUENCES
# =====================================================

LOOKBACK = 30

def create_sequences(X, y, lookback):

    Xs = []
    ys = []

    for i in range(len(X) - lookback):

        Xs.append(
            X[i:i+lookback]
        )

        ys.append(
            y[i+lookback]
        )

    return np.array(Xs), np.array(ys)

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

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================================================
# BUILD RNN MODEL
# =====================================================

model = Sequential([

    SimpleRNN(
        units=128,
        activation="tanh",
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Dropout(0.2),

    SimpleRNN(
        units=64,
        activation="tanh"
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="linear"
    )
])

# =====================================================
# COMPILE MODEL
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
# TRAIN MODEL
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
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# Inverse Scale
y_test_actual = target_scaler.inverse_transform(y_test)
y_pred_actual = target_scaler.inverse_transform(y_pred)

# =====================================================
# EVALUATION
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

print("\n==============================")
print("RNN RESULTS")
print("==============================")
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

plt.title("RNN Forecast")
plt.xlabel("Time")
plt.ylabel("Patient Count")

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# LOSS CURVE
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
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

model.save("saved_artifacts/rnn_patient_volume.keras")

print("\nModel saved successfully.")

#Forecast Next 30 Days
last_sequence = X_scaled[-LOOKBACK:]

future_predictions = []

current_sequence = last_sequence.copy()

for _ in range(30):

    pred = model.predict(
        current_sequence.reshape(
            1,
            LOOKBACK,
            len(FEATURES)
        ),
        verbose=0
    )

    future_predictions.append(pred[0][0])

    next_row = current_sequence[-1].copy()

    current_sequence = np.vstack([
        current_sequence[1:],
        next_row
    ])

future_predictions = target_scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

print(future_predictions)