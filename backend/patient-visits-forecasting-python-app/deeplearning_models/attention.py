import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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

import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# ====================================================
# LOAD DATA
# ====================================================

FILE_PATH = "data/patient_visits.csv"

df = pd.read_csv(FILE_PATH)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# ====================================================
# TARGET
# ====================================================

TARGET = "visits"

FEATURES = [
    col
    for col in df.columns
    if col not in ["date", TARGET]
]

df = df.ffill().bfill()

# ====================================================
# SCALING
# ====================================================

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(
    df[FEATURES]
)

y_scaled = target_scaler.fit_transform(
    df[[TARGET]]
)

# ====================================================
# CREATE SEQUENCES
# ====================================================

LOOKBACK = 30

def create_sequences(X, y, lookback):

    Xs = []
    ys = []

    for i in range(len(X) - lookback):

        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])

    return np.array(Xs), np.array(ys)

X, y = create_sequences(
    X_scaled,
    y_scaled,
    LOOKBACK
)

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ====================================================
# TRAIN TEST SPLIT
# ====================================================

split_index = int(len(X) * 0.80)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# ====================================================
# ATTENTION MODEL
# ====================================================

inputs = Input(
    shape=(
        X_train.shape[1],
        X_train.shape[2]
    )
)

lstm_out = LSTM(
    128,
    return_sequences=True
)(inputs)

attention_output = Attention()(
    [lstm_out, lstm_out]
)

pooled = GlobalAveragePooling1D()(
    attention_output
)

dropout = Dropout(0.3)(
    pooled
)

dense1 = Dense(
    64,
    activation="relu"
)(dropout)

dense2 = Dense(
    32,
    activation="relu"
)(dense1)

outputs = Dense(
    1,
    activation="linear"
)(dense2)

model = Model(
    inputs=inputs,
    outputs=outputs
)

# ====================================================
# COMPILE
# ====================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ====================================================
# CALLBACKS
# ====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5
)

# ====================================================
# TRAIN
# ====================================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[
        early_stop,
        reduce_lr
    ],
    verbose=1
)

# ====================================================
# PREDICT
# ====================================================

y_pred = model.predict(X_test)

# ====================================================
# INVERSE SCALING
# ====================================================

y_test_actual = target_scaler.inverse_transform(
    y_test
)

y_pred_actual = target_scaler.inverse_transform(
    y_pred
)

# ====================================================
# METRICS
# ====================================================

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

mape = np.mean(
    np.abs(
        (y_test_actual - y_pred_actual)
        / y_test_actual
    )
) * 100

print("\n========================")
print("ATTENTION-LSTM RESULTS")
print("========================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

#Visualization
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
    "Attention-LSTM Forecast"
)

plt.xlabel("Time")
plt.ylabel("Patient Count")

plt.legend()
plt.grid(True)

plt.show()

#Save Model
model.save(
    "saved_artifacts/attention_lstm_patient_volume.keras"
)
