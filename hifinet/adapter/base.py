from abc import ABC, abstractmethod

import pandas as pd
import pandera.pandas as pa
from loguru import logger
from pandera.api.pandas.model_config import BaseConfig

from hifinet.config import AdaptorConfig


class BaseAdaptor(ABC):
    def __init__(self, config: AdaptorConfig):
        self.config = config

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._filter_by_id(data)
        data = self._filter_by_period(data)
        data = self._resample(data)
        return data

    def _filter_by_id(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.config.subset_node:
            return data

        filter_data = data.copy()

        filter_data = filter_data[
            filter_data["node_id"].isin(self.config.subset_node)
        ].copy()

        return filter_data

    def _filter_by_period(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.config.period:
            return data

        filter_data = data.copy()

        filter_data = filter_data[
            (filter_data["datetime"] >= self.config.period[0])
            & (filter_data["datetime"] <= self.config.period[1])
        ].copy()

        return filter_data

    def _resample(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.config.resample_interval:
            return data

        resampled_list = []
        unique_nodes = data["node_id"].unique()

        for node in unique_nodes:
            note_data = data[data["node_id"] == node].copy()
            note_data = note_data.set_index("datetime")

            nums_nan = note_data["target"].isna().sum()
            logger.debug(
                f"Node {node}: Found {nums_nan} NaN values in target before resampling"
            )

            resampled_mote = note_data.resample(self.config.resample_interval).mean()
            resampled_mote["node_id"] = node

            nums_nan = resampled_mote["target"].isna().sum()

            logger.debug(
                f"Node {node}: Found {nums_nan} NaN values in target after resampling, before interpolation"
            )

            resampled_mote["target"] = resampled_mote["target"].interpolate("time")

            nums_nan = resampled_mote["target"].isna().sum()

            logger.debug(
                f"Node {node}: Found {nums_nan} NaN values in target after interpolation"
            )

            resampled_mote = resampled_mote.reset_index()
            resampled_list.append(resampled_mote)

        resampled_data = pd.concat(resampled_list, ignore_index=True)

        return resampled_data


class DataSchema(pa.DataFrameModel):
    datetime: pa.Timestamp = pa.Field(nullable=False, coerce=True)
    node_id: int = pa.Field(nullable=False, coerce=True)
    target: float = pa.Field(nullable=False, coerce=True)
    feature: float | None = pa.Field(
        regex=True, nullable=True, alias=r"^feature(?:_\d+)?$", coerce=True
    )

    @pa.dataframe_check
    def node_same_length(cls, df: pd.DataFrame) -> bool:
        sizes = df.groupby("node_id").size()
        return sizes.nunique() <= 1

    class Config(BaseConfig):
        strict = True
