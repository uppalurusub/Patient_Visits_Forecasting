# ==========================================================
# ANN Regressor for Patient Visits Forecasting
# ==========================================================
import warnings
warnings.filterwarnings("ignore")

import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Disable oneDNN messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# Load Dataset
# ==========================================================

FILE_PATH = "data/patient_visits.csv"

df = pd.read_csv(FILE_PATH)

print("\nDataset Shape:", df.shape)
print(df.head())

# ==========================================================
# Target Column
# ==========================================================

TARGET = "visits"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ==========================================================
# Identify Feature Types
# ==========================================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

# ==========================================================
# Preprocessing Pipelines
# ==========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Apply Preprocessing
# ==========================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrix to dense
X_train_processed = X_train_processed.toarray() \
    if hasattr(X_train_processed, "toarray") \
    else X_train_processed

X_test_processed = X_test_processed.toarray() \
    if hasattr(X_test_processed, "toarray") \
    else X_test_processed

print("\nProcessed Train Shape:", X_train_processed.shape)
print("Processed Test Shape:", X_test_processed.shape)

# ==========================================================
# Build ANN Model
# ==========================================================

input_dim = X_train_processed.shape[1]

model = Sequential([
    
    Dense(
        128,
        activation='relu',
        input_shape=(input_dim,)
    ),

    Dropout(0.2),

    Dense(
        64,
        activation='relu'
    ),

    Dropout(0.2),

    Dense(
        32,
        activation='relu'
    ),

    Dense(
        1,
        activation='linear'
    )
])

# ==========================================================
# Compile Model
# ==========================================================

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ==========================================================
# Early Stopping
# ==========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# ==========================================================
# Train Model
# ==========================================================

history = model.fit(
    X_train_processed,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================================
# Predictions
# ==========================================================

y_pred = model.predict(X_test_processed).flatten()

# ==========================================================
# Evaluation Metrics
# ==========================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("ANN Regressor Performance")
print("==============================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ==========================================================
# Visualization
# ==========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Patient Count")
plt.ylabel("Predicted Patient Count")
plt.title("ANN Regressor: Actual vs Predicted")

plt.tight_layout()
plt.show()

# ==========================================================
# Training History
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("ANN Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# Save Model
# ==========================================================

model.save("saved_artifacts/ann_patient_visits.keras")

print("\nModel saved as:")
print("ann_patient_visits.keras")

