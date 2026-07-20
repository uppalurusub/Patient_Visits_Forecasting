# 🏥 Patient Visits Forecasting System

A full-stack AI-powered Patient Visits Forecasting System that predicts future hospital patient visits using multiple Machine Learning, Statistical, and Deep Learning forecasting algorithms.

The project consists of:

- **Backend** – FastAPI-based REST API for training, forecasting, model comparison, MLflow tracking, and inference.
- **Frontend** – React + TypeScript + Material UI application for interacting with the forecasting APIs.
- **Streamlit UI** – Optional dashboard for quick experimentation.
- **MLflow Integration** – Experiment tracking, model registry, metrics logging, and inference monitoring.

---

# Project Architecture

```
Patient_Visits_Forecasting
│
├── backend/
│   ├── patient-visits-forecasting-python-app/
│   │
│   ├── routers/
│   ├── services/
│   ├── implementations/
│   ├── models/
│   ├── regression_ml/
│   ├── regression_ml_comparisons/
│   ├── statistical_models/
│   ├── deeplearning_models/
│   └── main.py
│
├── ui/
│   ├── reactjs/
│   │     patient-visits-forecasting-react-app/
│   │
│   └── streamlit/
│
├── data/
│
└── requirements.txt
```

---

# Features

## Backend

- FastAPI REST API
- Modular architecture
- Request & Response validation
- MLflow experiment tracking
- MLflow model registry
- Model comparison
- Automatic feature engineering
- Forecast generation
- Inference tracking
- Performance evaluation

---

## Machine Learning Models

The project supports numerous regression algorithms including:

- Multiple Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Decision Tree Regression
- Random Forest Regression
- Extra Trees Regression
- Gradient Boosting Regression
- AdaBoost Regression
- XGBoost Regression
- LightGBM Regression
- CatBoost Regression

---

## Statistical Forecasting Models

- ARIMA
- SARIMA
- Prophet
- Holt-Winters
- TBATS

---

## Deep Learning Models

- ANN
- RNN
- LSTM
- BiLSTM
- GRU
- CNN-LSTM
- Seq2Seq
- Attention Networks
- Multi-Head Attention
- Transformer
- Temporal Fusion Transformer (TFT)
- DeepAR
- N-HiTS
- NeuralProphet

---

# Frontend Features

Built using

- React 19
- TypeScript
- Vite
- Material UI
- Axios
- React Router

The UI provides:

- Forecast Request Form
- Model Selection
- Prediction Results
- Forecast Charts
- API Integration
- Error Handling
- Responsive Material Design

---

# Technology Stack

## Backend

- Python
- FastAPI
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- CatBoost
- Prophet
- Statsmodels
- MLflow
- Joblib
- Plotly
- Uvicorn

---

## Frontend

- React
- TypeScript
- Vite
- Material UI
- Axios
- React Router

---

# Dataset

The application uses

```
patient_visits.csv
```

Typical forecasting features include

- Date
- Patient Visits
- Lag Features
- Rolling Statistics
- Seasonal Features
- Calendar Features

---

# Backend Installation

## Clone

```bash
git clone <repository-url>
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Start MLflow

```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--host 0.0.0.0 \
--port 5000
```

---

## Run Backend

```bash
uvicorn main:app --reload
```

Default

```
http://localhost:8000
```

Swagger

```
http://localhost:8000/docs
```

---

# Frontend Installation

Navigate to

```bash
cd ui/reactjs/patient-visits-forecasting-react-app
```

Install packages

```bash
npm install
```

Run

```bash
npm run dev
```

Default

```
http://localhost:5173
```

Build

```bash
npm run build
```

Preview

```bash
npm run preview
```

---

# API Workflow

```
React UI

      │

      ▼

FastAPI

      │

      ▼

Feature Engineering

      │

      ▼

Selected Forecasting Model

      │

      ▼

Prediction

      │

      ▼

MLflow Logging

      │

      ▼

JSON Response
```

---

# MLflow Tracking

The application tracks

- Experiments
- Parameters
- Metrics
- Model Artifacts
- Feature Importance
- Training Time
- Evaluation Metrics
- Registered Models
- Inference History

---

# Forecasting Workflow

```
Patient Visits Data

        │

        ▼

Preprocessing

        │

        ▼

Feature Engineering

        │

        ▼

Model Training

        │

        ▼

Evaluation

        │

        ▼

Best Model Selection

        │

        ▼

Forecast Generation

        │

        ▼

Visualization
```

---

# Evaluation Metrics

Regression models are evaluated using

- MAE
- RMSE
- MSE
- R² Score
- Adjusted R²
- MAPE
- SMAPE

---

# REST APIs

Typical APIs include

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/forecast` | Generate forecast |
| POST | `/train` | Train model |
| POST | `/compare-models` | Compare forecasting models |
| GET | `/models` | List available models |
| GET | `/metrics` | Retrieve evaluation metrics |
| GET | `/health` | Health check |

> *Note: Refer to the FastAPI Swagger documentation (`/docs`) for the exact endpoints and request/response schemas implemented in your application.*

---

# Folder Structure

```
backend
│
├── routers
├── services
├── implementations
├── models
├── regression_ml
├── regression_ml_comparisons
├── statistical_models
├── deeplearning_models
└── main.py

ui
│
├── reactjs
└── streamlit
```

---

# Future Enhancements

- Auto Model Selection
- Hyperparameter Optimization
- Automated Retraining
- Drift Detection
- Docker Support
- Kubernetes Deployment
- CI/CD Pipeline
- Azure Deployment
- AWS Deployment
- Role-Based Authentication
- Real-Time Forecast Dashboard
- Explainable AI (SHAP/LIME)

---

# License

This project is intended for educational, research, and healthcare analytics purposes.

---

# Author

Developed using

- Python
- FastAPI
- React
- TypeScript
- Material UI
- Machine Learning
- Deep Learning
- Statistical Forecasting
- MLflow
