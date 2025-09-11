from typing import Any

import pandas as pd
from loguru import logger

from hifinet.model.random_forest import RandomForestModel


class Trainer:
    def __init__(self, data: pd.DataFrame, ratio):
        self.data = data
        self.ratio = ratio

    def train(self, model_name: str, model_params: dict[str, Any]):
        match model_name:
            case "random_forest":
                model = RandomForestModel(**model_params)
            case _:
                logger.error(f"Model {model_name} not implemented")
