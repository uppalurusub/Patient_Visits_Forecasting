# 🏥 Patient Visits Forecasting System (Python)

A production-ready **Patient Visits Forecasting System** built using **FastAPI**, **Machine Learning**, **Time Series Forecasting**, and **MLflow**. The application predicts future patient visits using multiple forecasting algorithms and also supports regression-based prediction using engineered healthcare features.

---

# 📌 Features

- REST API using FastAPI
- Multiple Time Series Forecasting Models
- Multiple Machine Learning Regression Models
- MLflow Experiment Tracking
- MLflow Inference Tracking
- Automatic Model Loading
- Feature Engineering Utilities
- Model Comparison
- Best Model Selection
- Pydantic Request Validation
- Interactive Swagger Documentation
- Health Check Endpoint
- Production Ready Architecture

---

# Project Architecture

```
Client
   │
   ▼
FastAPI REST API
   │
   ▼
Forecast Router
   │
   ▼
Forecast Service
   │
   ▼
Forecast Implementation
   │
   ▼
Model Loader
   │
   ▼
Saved ML Models
   │
   ▼
Prediction
   │
   ▼
MLflow Tracking
```

---

# Project Structure

```
patient-visits-forecasting-python-app/

│
├── best_saved_artifacts/
│   ├── model_comparison.csv
│   ├── AR.pkl
│   ├── ELASTICNET.pkl
│
├── deeplearning_models/
│   ├── ann.py
│   ├── lstm.py
│   ├── gru.py
│   ├── transformer.py
│   ├── cnn_lstm.py
│   └── ...
│
├── implementations/
│   ├── forecasting_impl.py
│   ├── mlflow_tracker_impl.py
│   ├── mlflow_inference_tracker.py
│   └── ...
│
├── models/
│   ├── request_model.py
│   └── response_model.py
│
├── regression_ml/
│   ├── linear_regression.py
│   ├── random_forest.py
│   ├── xgboost.py
│   ├── elasticnet.py
│   └── ...
│
├── regression_ml_comparisons/
│
├── routers/
│   └── forecasting_router.py
│
├── services/
│   └── forecasting_service.py
│
├── timeseries_ml/
│   ├── arima_forecasting.py
│   ├── arma_forecasting.py
│   ├── holt_trend_forecasting.py
│   ├── moving_average.py
│   └── ...
│
├── timeseries_ml_comparisons/
│
├── utils/
│   ├── model_loader.py
│   ├── patients_visits_feature_engineering.py
│   └── data_features_loader.py
│
├── main.py
└── mlflow.db
```

---

# Technology Stack

- Python 3.10+
- FastAPI
- Uvicorn
- Scikit-Learn
- StatsModels
- Pandas
- NumPy
- MLflow
- Pydantic

---

# Supported Time Series Models

The project supports numerous forecasting algorithms including:

- Auto Regression (AR)
- ARMA
- ARIMA
- Holt Trend
- Holt Damped Trend
- Moving Average
- Exponential Smoothing
- Seasonal Forecasting
- Additional Statistical Forecasting Models

---

# Supported Regression Models

Examples include:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- CatBoost Regressor
- XGBoost (if installed)

---

# Deep Learning Models

The project also contains implementations of:

- Artificial Neural Network (ANN)
- LSTM
- Bidirectional LSTM
- CNN-LSTM
- Attention Networks
- DeepAR
- Transformer
- Temporal Fusion Transformer (TFT)

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/patient-visits-forecasting-python-app.git

cd patient-visits-forecasting-python-app
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

```bash
uvicorn main:app --reload
```

Application

```
http://localhost:8000
```

Swagger UI

```
http://localhost:8000/docs
```

ReDoc

```
http://localhost:8000/redoc
```

---

# API Endpoints

## Health Check

### GET

```
/
```

Response

```json
{
    "application": "Patient Visits Forecasting API",
    "version": "1.0.0",
    "status": "Running"
}
```

---

### GET

```
/health
```

Response

```json
{
    "status":"healthy"
}
```

---

# Forecast Future Visits

### POST

```
/api/v1/forecast/predict
```

Request

```json
{
    "model_name":"AR",
    "horizon":30
}
```

Parameters

| Field | Description |
|---------|------------|
| model_name | Forecasting model to use |
| horizon | Number of future days to forecast |

Response

```json
{
    "success": true,
    "message": "Forecast generated",
    "model_name": "AR",
    "prediction": [
        112,
        118,
        121,
        123
    ]
}
```

---

# Regression Prediction

### POST

```
/api/v1/forecast/regression-predict
```

Request

```json
{
  "patient_count":120,
  "admissions":32,
  "discharges":30,
  "occupancy":71.5,
  "year":2024,
  "month":6,
  "day":14,
  "dayofweek":4,
  "quarter":2,
  "weekofyear":24
}
```

Response

```json
{
    "prediction":126.84
}
```

---

# MLflow Integration

The application tracks:

- Prediction Requests
- Model Used
- Forecast Horizon
- Prediction Results
- Errors
- Inference Metadata

Launch MLflow

```bash
mlflow ui
```

Open

```
http://localhost:5000
```

---

# Feature Engineering

The project performs feature engineering using temporal variables such as:

- Year
- Month
- Day
- Day of Week
- Quarter
- Week of Year
- Patient Count
- Admissions
- Discharges
- Occupancy

---

# Model Comparison

The project includes utilities for comparing multiple forecasting and regression models based on evaluation metrics and stores comparison results under:

```
best_saved_artifacts/model_comparison.csv
```

---

# Future Improvements

- Docker Support
- Kubernetes Deployment
- Azure Deployment
- Batch Forecasting
- Explainable AI (SHAP/LIME)
- Model Registry
- Drift Detection
- Automated Retraining
- CI/CD Pipelines
- Authentication & Authorization
- Monitoring Dashboard

---

# License

This project is intended for educational and research purposes. Add your preferred open-source license before public distribution.

---

# Author

Developed as a Machine Learning and Time Series Forecasting project using FastAPI, MLflow, and Python for healthcare analytics and patient visit prediction.
