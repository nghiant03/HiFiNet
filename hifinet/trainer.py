from typing import Any

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(
        self, x_train: Any, y_train: Any, x_val: Any, y_val: Any, x_test: Any, y_test: Any
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def train(self, model_name: str, model_params: dict[str, Any] | None = None):
        logger.info(f"Training {model_name}")
        match model_name:
            case "random_forest":
                if model_params:
                    self.model = (
                        RandomForestClassifier(n_jobs=8, **model_params)
                        if model_params
                        else RandomForestClassifier(n_jobs=8) 
                    )
            case _:
                logger.error(f"Model {model_name} not implemented")
                raise NotImplementedError

        self.model.fit(self.x_train, self.y_train)

        val_pred = self.model.predict(self.x_val)

        return accuracy_score(self.y_val, val_pred)
