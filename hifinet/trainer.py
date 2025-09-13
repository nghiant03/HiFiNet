import re
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
    ):
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.sequence_columns = [
            column
            for column in self.train_data.columns
            if re.match(r"feature_*", column)
        ]
        self.sequence_columns.append("target")

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

        columns = self.sequence_columns + ["seq_id"]
        x_train = self.train_data[columns]
        y_train = self.train_data["type"]

        self.model.fit(x_train, y_train)

        x_val = self.val_data[columns]
        y_val = self.val_data["type"]
        val_pred = self.model.predict(x_val)

        return accuracy_score(y_val, val_pred)
