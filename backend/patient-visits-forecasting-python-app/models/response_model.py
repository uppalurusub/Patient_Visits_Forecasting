from pydantic import BaseModel
from typing import Optional
from typing import List
from typing import Dict
from typing import Any


class BaseResponse(BaseModel):

    success: bool

    message: str


class TrainingResponse(BaseResponse):

    model_name: Optional[str] = None

    metrics: Optional[Dict[str, float]] = None

    model_path: Optional[str] = None

    mlflow_run_id: Optional[str] = None


class PredictionResponse(BaseResponse):

    success: bool

    message: str

    model_name: Optional[str] = None

    prediction: Optional[List[float]] = None


class ForecastResponse(BaseResponse):

    model_name: Optional[str] = None

    horizon: Optional[int] = None

    forecast: Optional[List[float]] = None


class ErrorResponse(BaseResponse):

    error: Optional[str] = None

class RegressionPredictResponse(BaseModel):

    prediction: float