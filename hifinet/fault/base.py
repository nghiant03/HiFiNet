from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

TTarget = TypeVar("TTarget")

class BaseFault(ABC, Generic[TTarget]):
    def __init__(self, chance: float, seed: int):
        self.chance = chance
        self._rng = np.random.default_rng(seed)

    def _num_points(self, data: pd.DataFrame) -> int:
        return max(1, round(len(data) * self.chance))

    @abstractmethod
    def sample_random(self):
        raise NotImplementedError

    @abstractmethod
    def select_targets(self, data: pd.DataFrame) -> NDArray[Any]:
        raise NotImplementedError

    @abstractmethod
    def get_slice(self, target: TTarget) -> slice:
        raise NotImplementedError

    @abstractmethod
    def transform_slice(
        self,
        result: pd.DataFrame,
        target_slice: slice,
        columns: list[str],
    ) -> pd.DataFrame:
        raise NotImplementedError

    def apply(
        self, data: pd.DataFrame, target_cols: str | list[str], type_idx: int
    ) -> pd.DataFrame:
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        assert not data[target_cols].isnull().any().any(), (
            f"Null values found in target_cols: {target_cols}"
        )
        logger.info(f"Applying fault {self.__class__} to data")

        result = data.copy()

        targets = self.select_targets(result)

        for target in targets:
            row_slice = self.get_slice(target)
            modified = self.transform_slice(result, row_slice, target_cols)
            result.loc[row_slice, target_cols] = modified
            result.loc[row_slice, "type"] = type_idx

        return result


class InstantFault(BaseFault[int]):
    def select_targets(self, data: pd.DataFrame) -> NDArray[Any]:
        num_points = self._num_points(data)
        logger.info(f"Total {num_points} points to be injected.")

        fault_points = self._rng.choice(data.index, size=num_points, replace=False)
        return fault_points

    def get_slice(self, target: int) -> slice:
        return slice(target, target + 1)

    def transform_slice(
        self, result: pd.DataFrame, target_slice: slice, columns: list[str]
    ) -> pd.DataFrame:
        logger.debug(f"Target slice: {target_slice}")
        data_slice = result.iloc[target_slice].copy()
        random_value = self.sample_random()

        logger.debug(f"Random value: {random_value}")
        for column in columns:
            data_slice[column] = self.transform_point(data_slice, column, random_value)
        return data_slice

    @abstractmethod
    def transform_point(
        self, fault_point: pd.DataFrame, target_col: str, random_value: float
    ) -> pd.Series:
        raise NotImplementedError


class IntervalFault(BaseFault[NDArray[np.int64]]):
    def __init__(
        self, min_length: int, max_length: int, gap: int, chance: float, seed: int
    ):
        super().__init__(chance, seed)
        self.min_length = min_length
        self.max_length = max_length
        self.gap = gap

    def transform_slice(
        self,
        result: pd.DataFrame,
        target_slice: slice,
        columns: list[str],
    ) -> pd.DataFrame:
        logger.debug(f"Target slice: {target_slice}")
        data_slice = result.iloc[target_slice].copy()
        random_value = self.sample_random()
        logger.debug(f"Random value: {random_value}")
        for column in columns:
            data_slice[column] = self.transform_interval(
                data_slice, column, random_value
            )
        return data_slice

    @abstractmethod
    def transform_interval(
        self, fault_interval: pd.DataFrame, target_col: str, random_value: float
    ) -> pd.Series:
        raise NotImplementedError

    def select_targets(self, data: pd.DataFrame) -> NDArray[np.int64]:
        num_points = self._num_points(data)
        logger.info(f"Total {num_points} points to be injected.")

        candidates = np.arange(len(data))

        intervals = np.empty((0, 2), dtype=np.int64)
        unavailable = np.zeros(len(data), dtype=bool)
        placed = 0

        self._rng.shuffle(candidates)
        for start in candidates:
            if placed > num_points:
                break

            max_fit = self._try_start(start, data, unavailable)
            if max_fit == -1:
                continue

            length_cap = np.minimum(max_fit, self.max_length)
            length = self._rng.integers(self.min_length, length_cap + 1)
            intervals = np.vstack((intervals, [start, length]), dtype=np.int64)

            unavailable = self._mark_unavailable(unavailable, start, length)
            placed += length

        return intervals

    def _try_start(self, start, data, unavailable) -> int:
        if unavailable[start]:
            return -1

        max_fit = np.minimum(self.max_length, len(data) - start)

        if max_fit < self.min_length:
            return -1

        end = start
        while (
            end < len(data) and not unavailable[end] and (end - start) < self.max_length
        ):
            end += 1

        max_fit = end - start

        if max_fit < self.min_length:
            return -1

        return max_fit

    def _mark_unavailable(self, unavailable, start, length):
        left = max(0, start - self.gap)
        right = min(len(unavailable), start + length + self.gap)
        unavailable[left:right] = True

        return unavailable

    def get_slice(self, target: NDArray[np.int64]) -> slice:
        start, length = int(target[0]), int(target[1])
        return slice(start, start + length)
