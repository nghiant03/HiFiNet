import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from hifinet.adapter.base import BaseAdaptor

DEFAULT_PATH = Path.cwd() / "data/intel/data.txt"
class IntelAdaptor(BaseAdaptor):
    def __init__(self, path: Optional[str] = None):
        super().__init__(Path(path) if path else DEFAULT_PATH)
            
    def read(self) -> pd.DataFrame:
        logger.info(f"Reading data at path {Path}")
        columns = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]
        data = pd.read_csv(self.path, sep= " ", names=columns)

        data["datetime"] = pd.to_datetime(data["date"] + " " + data["time"], format="mixed")
        data.drop(columns=["date", "time"], inplace=True)
        logger.info(f"Dataframe loaded with shape {data.shape}")

        return data
