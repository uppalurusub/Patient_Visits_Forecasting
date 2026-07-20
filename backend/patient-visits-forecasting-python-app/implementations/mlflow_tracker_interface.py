from abc import ABC
from abc import abstractmethod


class MLflowTrackerInterface(ABC):

    @abstractmethod
    def log_prediction(
        self,
        model_name: str,
        horizon: int,
        predictions: list
    ):
        pass

    @abstractmethod
    def log_error(
        self,
        model_name: str,
        error_message: str
    ):
        pass