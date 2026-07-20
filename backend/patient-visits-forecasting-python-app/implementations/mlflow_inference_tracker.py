import logging
from datetime import datetime

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


def track_prediction_request(model_name, horizon, predictions):
    if not MLFLOW_AVAILABLE:
        return

    try:
        active = mlflow.active_run()

        if active is None:
            with mlflow.start_run(run_name=f"{model_name}_Inference"):
                _log(model_name, horizon, predictions)
        else:
            with mlflow.start_run(
                run_name=f"{model_name}_Inference",
                nested=True
            ):
                _log(model_name, horizon, predictions)

    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")


def _log(model_name, horizon, predictions):

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("horizon", horizon)
    mlflow.log_metric("prediction_count", len(predictions))


def track_prediction_error(model_name, error_message):

    if not MLFLOW_AVAILABLE:
        return

    try:

        with mlflow.start_run(
            run_name=f"{model_name}_Error"
        ):

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("timestamp", str(datetime.now()))
            mlflow.log_param("error", error_message)

    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")