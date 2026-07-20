import os
import joblib


class ModelLoader:

    MODEL_DIR = "best_saved_artifacts"

    @classmethod
    def get_model(
        cls,
        model_name: str
    ):

        model_path = os.path.join(
            cls.MODEL_DIR,
            f"{model_name}.pkl"
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        return joblib.load(model_path)