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
    def read(self) -> pd.DataFrame:
        super().read()

        record = []
        with open(self.config.path) as json_file:
            row = {}
            data = json.load(json_file)
            row["id"] = int(data["x"] + data["y"])
            row["type"] = int(data["fault"])
            row["feature_1"] = [float(i) for i in data["data"][0]]
            row["feature_2"] = [float(i) for i in data["data"][1]]
            row["target"] = [float(i) for i in data["data"][2]]

            record.append(row)

        return pd.DataFrame(record)
