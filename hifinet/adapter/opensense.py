import json
from pathlib import Path

import pandas as pd

from hifinet.adapter.base import BaseAdaptor
from hifinet.config import AdaptorConfig

DEFAULT_PATH = Path.cwd() / "data/opensense/NASA_data_0.1.json"
DEFAULT_CONFIG = AdaptorConfig(
    path=DEFAULT_PATH,
)


class OSAdaptor(BaseAdaptor):
    def __init__(self, config: AdaptorConfig | None = None):
        config = config or DEFAULT_CONFIG
        super().__init__(config)
    def read(self) -> pd.DataFrame:
        super().read()

        record = []
        with open(self.config.path) as json_file:
            row = {}
            data = json.load(json_file)
            for obj in data:
                row["datetime"] = pd.Timestamp(2023, 1, 1) + pd.to_timedelta(
                    int(obj["day"]) - 1, "D"
                )
                row["id"] = int(obj["x"] + obj["y"])
                row["type"] = int(obj["fault"])
                row["feature_1"] = [float(i) for i in obj["data"][0]]
                row["feature_2"] = [float(i) for i in obj["data"][1]]
                row["target"] = [float(i) for i in obj["data"][2]]

                record.append(row)

        return pd.DataFrame(record)
