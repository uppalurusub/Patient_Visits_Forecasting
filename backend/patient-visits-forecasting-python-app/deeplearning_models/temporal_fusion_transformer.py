import pandas as pd
import numpy as np

import torch

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

from pytorch_forecasting.metrics import RMSE

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/patient_visits.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

# =====================================================
# CREATE TIME INDEX
# =====================================================

df["time_idx"] = np.arange(len(df))

# Single hospital series

df["series"] = "hospital"

# =====================================================
# TARGET
# =====================================================

TARGET = "visits"

# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

max_prediction_length = 30
max_encoder_length = 60

training_cutoff = (
    df["time_idx"].max()
    - max_prediction_length
)

# =====================================================
# DATASET
# =====================================================

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],

    time_idx="time_idx",

    target=TARGET,

    group_ids=["series"],

    min_encoder_length=max_encoder_length,

    max_encoder_length=max_encoder_length,

    min_prediction_length=max_prediction_length,

    max_prediction_length=max_prediction_length,

    static_categoricals=["series"],

    time_varying_known_reals=[
        "time_idx"
    ],

    time_varying_unknown_reals=[
        "patient_count",
        "visits",
        "admissions",
        "discharges",
        "occupancy"
    ]
)

validation = TimeSeriesDataSet.from_dataset(
    training,
    df,
    predict=True,
    stop_randomization=True
)

# =====================================================
# DATALOADERS
# =====================================================

batch_size = 64

train_dataloader = training.to_dataloader(
    train=True,
    batch_size=batch_size
)

val_dataloader = validation.to_dataloader(
    train=False,
    batch_size=batch_size
)

# =====================================================
# EARLY STOPPING
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min"
)

# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(
    max_epochs=50,
    accelerator="auto",
    callbacks=[early_stop]
)

# =====================================================
# TFT MODEL
# =====================================================

from pytorch_forecasting.metrics import QuantileLoss

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4
)

print(
    f"Parameters: {tft.size()/1e3:.1f}k"
)

# =====================================================
# TRAIN
# =====================================================

trainer.fit(
    tft,
    train_dataloader,
    val_dataloader
)

# =====================================================
# SAVE MODEL
# =====================================================

trainer.save_checkpoint(
    "saved_artifacts/tft_patient_volume.ckpt"
)

print(
    "Model saved successfully"
)

#Forecast Next 30 Days
best_tft = TemporalFusionTransformer.load_from_checkpoint(
    "saved_artifacts/tft_patient_volume.ckpt"
)

predictions = best_tft.predict(
    val_dataloader
)

print(predictions)

#Evaluate Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

actuals = torch.cat(
    [y[0] for x, y in iter(val_dataloader)]
)

preds = predictions

actuals = actuals.numpy().flatten()
preds = preds.numpy().flatten()

mae = mean_absolute_error(
    actuals,
    preds
)

rmse = np.sqrt(
    mean_squared_error(
        actuals,
        preds
    )
)

r2 = r2_score(
    actuals,
    preds
)

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

#TFT Feature Importance
raw_predictions, x = best_tft.predict(
    val_dataloader,
    mode="raw",
    return_x=True
)

best_tft.plot_interpretation(
    raw_predictions
)