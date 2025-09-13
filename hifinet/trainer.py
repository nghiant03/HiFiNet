import re
from typing import Any

import pandas as pd
from loguru import logger

from hifinet.model import RandomForestModel


class Trainer:
    def __init__(self, data: pd.DataFrame, ratio: float, temp):
        self.data = data
        self.ratio = ratio
        self.sequence_columns = [
            column for column in self.data.columns if re.match(r"feature_*", column)
        ]
        self.sequence_columns.append("target")
        self.train_data, self.val_data, self.test_data = self._split(temp)

    def train(self, model_name: str, model_params: dict[str, Any] | None = None):
        logger.info(f"Training {model_name}")
        match model_name:
            case "random_forest":
                if model_params:
                    self.model = RandomForestModel(
                        sequence_columns=self.sequence_columns, **model_params
                    )
                else:
                    self.model = RandomForestModel(
                        sequence_columns=self.sequence_columns
                    )
            case _:
                logger.error(f"Model {model_name} not implemented")
                raise NotImplementedError

        instance_data = self.train_data.copy()
        instance_data["seq_idx"] = instance_data.groupby("id").cumcount()
        instance_data["seq_id"] = instance_data["id"].astype(str) + "_" + instance_data["seq_idx"].astype(str)
        columns = self.sequence_columns + ["seq_id"]
        x = instance_data[columns]
        y = instance_data["type"]

        self.model.fit(x, y)

    def _split(self, temp):
        match temp:
            case 0:
                return (
                    self.data[self.data["datetime"] < "2023-08-01"],
                    self.data[
                        (self.data["datetime"] >= "2023-08-01")
                        | (self.data["datetime"] < "2023-10-01")
                    ],
                    self.data[
                        (self.data["datetime"] >= "2023-10-01")
                        | (self.data["datetime"] < "2023-11-01")
                    ],
                )
            case 1:
                return (
                    self.data[
                        (self.data["datetime"] >= "2023-02-01")
                        | (self.data["datetime"] < "2023-09-01")
                    ],
                    self.data[
                        (self.data["datetime"] >= "2023-09-01")
                        | (self.data["datetime"] < "2023-11-01")
                    ],
                    self.data[
                        (self.data["datetime"] >= "2023-11-01")
                        | (self.data["datetime"] < "2023-12-01")
                    ],
                )
            case 2:
                return (
                    self.data[
                        (self.data["datetime"] >= "2023-03-01")
                        | (self.data["datetime"] < "2023-10-01")
                    ],
                    self.data[
                        (self.data["datetime"] >= "2023-10-01")
                        | (self.data["datetime"] < "2023-12-01")
                    ],
                    self.data[self.data["datetime"] >= "2023-12-03"],
                )
            case _:
                raise NotImplementedError
