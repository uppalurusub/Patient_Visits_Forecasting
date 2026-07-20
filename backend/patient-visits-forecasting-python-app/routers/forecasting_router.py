from fastapi import APIRouter
from fastapi import HTTPException

from services.forecasting_service import (
    ForecastingService
)

from models.request_model import (
    PredictRequest,RegressionPredictRequest
)

from models.response_model import (
    PredictionResponse, RegressionPredictResponse
)

router = APIRouter(
    prefix="/forecast",
    tags=["Forecasting"]
)

service = ForecastingService()

@router.post(
    "/regression-predict",
    response_model=RegressionPredictResponse
)
def predict(request: RegressionPredictRequest):

    try:

        prediction = service.regression_predict(request)

        return prediction
    except Exception as ex:

        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )

@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictRequest
):

    try:

        prediction = service.predict(
            model_name=request.model_name,
            horizon=request.horizon
        )

        return PredictionResponse(
            success=True,
            message="Forecast generated",
            model_name=request.model_name,
            prediction=prediction
        )

    except Exception as ex:

        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )