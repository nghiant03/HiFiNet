from typing import Any

import pandas as pd
from loguru import logger


class Trainer:
    def __init__(self, data: pd.DataFrame, ratio):
        self.data = data
        self.ratio = ratio

    def train(self, model: str, model_params: dict[str, Any]):
        match model:
            case "random_forest":
                pass
            case _:
                logger.error(f"Model {model} not implemented")
