import requests
TIMESERIES_API='http://127.0.0.1:8000/api/v1/forecast/predict'
REGRESSION_API='http://127.0.0.1:8000/api/v1/forecast/regression-predict'
def predict_timeseries(payload): return requests.post(TIMESERIES_API,json=payload,timeout=120).json()
def predict_regression(payload): return requests.post(REGRESSION_API,json=payload,timeout=120).json()
