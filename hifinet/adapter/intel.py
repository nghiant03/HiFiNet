from pathlib import Path

import pandas as pd
from loguru import logger
from pandera.errors import SchemaErrors

from hifinet.adapter.base import AdaptorConfig, BaseAdaptor, DataSchema

DEFAULT_PATH = Path.cwd() / "data/intel/data.txt"
DEFAULT_CONFIG = AdaptorConfig(
    path=DEFAULT_PATH,
    subset_node=[i for i in range(7, 13)],
    period=[pd.Timestamp("2004-03-01"), pd.Timestamp("2004-03-07")],
    resample_interval="5min",
    rename_columns={
        "moteid": "node_id",
        "temperature": "target",
        "humidity": "feature_1",
        "light": "feature_2",
        "voltage": "feature_3",
    },
)


class IntelAdaptor(BaseAdaptor):
    def __init__(self, config: AdaptorConfig = DEFAULT_CONFIG):
        super().__init__(config)

    def read(self) -> pd.DataFrame:
        logger.debug(f"Current adaptor config: {self.config}")
        logger.info(f"Reading data at path {self.config.path}")
        columns = [
            "date",
            "time",
            "epoch",
            "moteid",
            "temperature",
            "humidity",
            "light",
            "voltage",
        ]
        data = pd.read_csv(self.config.path, sep=" ", names=columns)

        data.dropna(subset=["date", "time", "moteid"], inplace=True)
        data["datetime"] = pd.to_datetime(
            data["date"] + " " + data["time"], format="mixed"
        )
        data.drop(columns=["date", "time"], inplace=True)
        data["moteid"] = data["moteid"].astype("uint16")
        data.sort_values(["datetime", "moteid"], inplace=True)
        logger.info(f"Dataframe loaded with shape {data.shape}")

        data.rename(columns=self.config.rename_columns, inplace=True)

        logger.info("Processing data")
        data = self.process(data)

        try:
            data = DataSchema.validate(data, lazy=True)
        except SchemaErrors:
            logger.error("Failed to validated data")
            raise

        logger.info("Dataframe passed validation")

        return data
