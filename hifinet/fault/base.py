import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import List, Union, Any, TypeVar, Generic
from loguru import logger

TTarget = TypeVar("TTarget")

class BaseFault(ABC, Generic[TTarget]):
    def __init__(self, chance: float, seed: int):
        self.chance = chance
        self._rng = np.random.default_rng(seed)

    def _num_points(self, data: pd.DataFrame) -> int:
        return max(1, round(len(data) * self.chance))

    @abstractmethod
    def _sample_random(self):
        raise NotImplementedError

    @abstractmethod
    def select_targets(self, data: pd.DataFrame) -> NDArray[Any]:
        raise NotImplementedError

    @abstractmethod
    def transform_slice(
        self,
        result: pd.DataFrame,
        target: TTarget,
        columns: List[str],
        type_index: int,
    ) -> Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, data: pd.DataFrame, type_index: int, target_cols: Union[str, List[str]]
    ) -> pd.DataFrame:
        assert data[target_cols].isnull().any().any(), f"Null values found in target_cols: {target_cols}"
        logger.info(f"Applying fault {type_index} to dataframe.")

        result = data.copy()
        result["type"] = 0
        result["type"] = result["type"].astype("int16")

        targets = self.select_targets(result)

        if isinstance(target_cols, str):
            target_cols = [target_cols]
        for target in targets:
            modified = self.transform_slice(result, target, target_cols, type_index)
            result.iloc[target] = modified

        return result

class InstantFault(BaseFault[int]):
    def select_targets(self, data: pd.DataFrame) -> NDArray[Any]:
        num_points = self._num_points(data)
        logger.info(f"Total {num_points} points to be injected.")

        fault_points = self._rng.choice(data.index, size=num_points, replace=False)
        return fault_points

    def transform_slice(
        self,
        result: pd.DataFrame,
        target: int,
        columns: List[str],
        type_index: int,
    ) -> pd.Series:
        logger.debug(f"Target type: {type(target)}")
        target_slice = result.iloc[target].copy()
        random_value = self._sample_random()

        logger.debug(f"Random value: {random_value}")
        for column in columns:
            target_slice[column] = self._transform_point(target_slice, column, random_value)
        target_slice["type"] = type_index
        return target_slice

    @abstractmethod
    def _transform_point(self, fault_point: pd.Series, target_col: str, random_value: float) -> pd.Series:
        raise NotImplementedError

class IntervalFault(BaseFault[NDArray[np.uint16]]):
    def __init__(self, min_length: int, max_length: int, gap: int, chance: float, seed: int):
        super().__init__(chance, seed)
        self.min_length = min_length
        self.max_length = max_length
        self.gap = gap

    def transform_slice(
        self,
        result: pd.DataFrame,
        target: NDArray[np.uint16],
        columns: List[str],
        type_index: int,
    ) -> pd.DataFrame:
        start, length = target
        target_slice = result.iloc[start : start + length].copy()
        random_value = self._sample_random() 
        for column in columns:
            target_slice[column] = self._transform_interval(target_slice, column, random_value)
        target_slice["type"] = type_index
        return target_slice

    @abstractmethod
    def _transform_interval(self, fault_interval: pd.DataFrame, target_col: str, random_value: float) -> pd.DataFrame:
        raise NotImplementedError

    def select_targets(self, data: pd.DataFrame) -> NDArray[np.uint16]:
        num_points = self._num_points(data)
        logger.info(f"Total {num_points} points to be injected.")

        candidates = np.arange(len(data))

        intervals = np.empty((0, 2), dtype=np.uint16)
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
            intervals = np.vstack((intervals, [start, length]), dtype=np.uint16)

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
        while end < len(data) and not unavailable[end] and (end - start) < self.max_length:
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
