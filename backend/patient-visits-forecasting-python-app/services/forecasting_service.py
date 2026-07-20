from implementations.forecasting_impl import (
    ForecastingImplementation
)

from utils.model_loader import (
    ModelLoader
)

from implementations.mlflow_inference_tracker import (
    track_prediction_request,
    track_prediction_error
)

from implementations.mlflow_tracker_impl import MLflowTrackerImplementation
from models.request_model import (
    RegressionPredictRequest
)
tracker = MLflowTrackerImplementation()


class ForecastingService:

    def regression_predict(self,  request: RegressionPredictRequest):

        model = ModelLoader.get_model(
                "ELASTICNET"
            )

        implementation = ForecastingImplementation(
                model=model
            )

        prediction = implementation.regression_predict(request)

        return {
            "prediction": prediction
        }

    def predict(
        self,
        model_name: str,
        horizon: int
    ):

        try:

            model = ModelLoader.get_model(
                model_name
            )

            implementation = ForecastingImplementation(
                model=model
            )

            forecast = implementation.forecast(
                horizon=horizon
            )

            
            tracker.log_prediction(
                model_name=model_name,
                horizon=horizon,
                predictions=forecast
            )

            return forecast

        except Exception as ex:

            tracker.log_error(
                model_name=model_name,
                error_message="Failure"

            )

            raise ex