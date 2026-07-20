import logging

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

from implementations.mlflow_tracker_interface import (
    MLflowTrackerInterface,
)

logger = logging.getLogger(__name__)


class MLflowTrackerImplementation(MLflowTrackerInterface):

    def log_prediction(
        self,
        model_name,
        horizon,
        predictions
    ):

        if not MLFLOW_AVAILABLE:
            return

        try:

            with mlflow.start_run(
                run_name=f"{model_name}_Inference",
                nested=True
            ):

                mlflow.log_param("model_name", model_name)
                mlflow.log_param("horizon", horizon)
                mlflow.log_metric(
                    "prediction_count",
                    len(predictions)
                )

        except Exception as e:
            logger.warning(f"MLflow logging skipped: {e}")

    def log_error(
        self,
        model_name,
        error_message
    ):

        if not MLFLOW_AVAILABLE:
            return

        try:

            with mlflow.start_run(
                run_name=f"{model_name}_Error",
                nested=True
            ):

                mlflow.log_param("model_name", model_name)
                mlflow.log_param("error", error_message)

        except Exception as e:
            logger.warning(f"MLflow logging skipped: {e}")