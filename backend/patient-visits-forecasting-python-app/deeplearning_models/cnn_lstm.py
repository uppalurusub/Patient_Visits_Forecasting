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

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# =====================================================
# LOAD DATA
# =====================================================

FILE_PATH = "data/patient_visits.csv"

df = pd.read_csv(FILE_PATH)

print("Dataset Shape:", df.shape)

# =====================================================
# DATE COLUMN
# =====================================================

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

# =====================================================
# TARGET COLUMN
# =====================================================

TARGET = "visits"

FEATURES = [
    col
    for col in df.columns
    if col not in ["date", TARGET]
]

print("\nFeatures:")
print(FEATURES)

# =====================================================
# HANDLE MISSING VALUES
# =====================================================

df = df.ffill().bfill()

# =====================================================
# FEATURE SCALING
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

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

split_index = int(len(X) * 0.80)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================================================
# BUILD CNN-BILSTM MODEL
# =====================================================

model = Sequential([

    Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu',
        padding='same',
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        padding='same'
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Bidirectional(
        LSTM(
            units=128,
            return_sequences=True
        )
    ),

    Dropout(0.3),

    Bidirectional(
        LSTM(
            units=64
        )
    ),

    Dropout(0.3),

    Dense(
        64,
        activation='relu'
    ),

    Dense(
        32,
        activation='relu'
    ),

    Dense(
        1,
        activation='linear'
    )
])

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
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
    callbacks=[
        early_stop,
        reduce_lr
    ],
    verbose=1
)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# INVERSE TRANSFORM
# =====================================================

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

mape = np.mean(
    np.abs(
        (y_test_actual - y_pred_actual)
        / y_test_actual
    )
) * 100

print("\n============================")
print("CNN-BiLSTM PERFORMANCE")
print("============================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(15,6))

plt.plot(
    y_test_actual,
    label="Actual"
)

plt.plot(
    y_pred_actual,
    label="Predicted"
)

plt.title(
    "CNN-BiLSTM Forecast"
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
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title(
    'CNN-BiLSTM Training History'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

model.save(
    "saved_artifacts/cnn_bilstm_patient_volume.keras"
)

print(
    "\nModel Saved Successfully"
)

#Forecast Next 30 Days
future_days = 30

last_sequence = X_scaled[-LOOKBACK:]

future_predictions = []

current_sequence = last_sequence.copy()

for _ in range(future_days):

    pred = model.predict(
        current_sequence.reshape(
            1,
            LOOKBACK,
            len(FEATURES)
        ),
        verbose=0
    )

    future_predictions.append(
        pred[0][0]
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
    "Forecast_visits":
    future_predictions.flatten()
})

forecast_df.to_csv(
    "saved_artifacts/cnn_bilstm_forecast.csv",
    index=False
)

print(forecast_df.head())

