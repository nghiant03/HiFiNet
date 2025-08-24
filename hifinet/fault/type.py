import numpy as np
import pandas as pd
from typing import List, Optional

from hifinet.fault.base import InstantFault, IntervalFault


class HardoverFault(IntervalFault):
    def __init__(
        self,
        min_length: int,
        max_length: int,
        gap: int,
        bias_range: List[float],
        chance: float,
        seed: int,
    ):
        super().__init__(min_length, max_length, gap, chance, seed)
        self.bias_range = bias_range

    def sample_random(self):
        bias = self._rng.uniform(self.bias_range[0], self.bias_range[1])
        bias = self._rng.choice([-1, 1]) * bias
        return bias

    def transform_interval(
        self, fault_interval: pd.DataFrame, target_col: str, random_value: float
    ) -> pd.Series:
        result = fault_interval[target_col].copy()
        result += random_value
        return result


class DriftFault(IntervalFault):
    def __init__(
        self,
        sigma: float,
        min_drift: float,
        min_length: int,
        max_length: int,
        gap: int,
        chance: float,
        seed: int,
    ):
        super().__init__(min_length, max_length, gap, chance, seed)
        self.sigma = sigma
        self.min_drift = min_drift

    def sample_random(self):
        initial_value = self._rng.normal(0, self.sigma)
        initial_value = (
            initial_value + self.min_drift
            if initial_value > 0
            else initial_value - self.min_drift
        )
        return initial_value

    def transform_interval(
        self, fault_interval: pd.DataFrame, target_col: str, random_value: float
    ) -> pd.Series:
        result = fault_interval[target_col].copy()
        n_points = len(result)

        fault_values = [n * random_value for n in range(n_points)]
        result += fault_values
        return result


class SpikeFault(InstantFault):
    def __init__(self, bias_range: List[float], chance: float, seed: int):
        super().__init__(chance, seed)
        self.bias_range = bias_range

    def sample_random(self):
        bias = self._rng.uniform(self.bias_range[0], self.bias_range[1])
        bias = self._rng.choice([-1, 1]) * bias
        return bias

    def transform_point(
        self, fault_point: pd.DataFrame, target_col: str, random_value: float
    ) -> pd.Series:
        result = fault_point[target_col].copy()
        result += random_value
        return result


class ErraticFault(IntervalFault):
    def __init__(
        self,
        min_multiplier: float,
        scale: float,
        min_length: int,
        max_length: int,
        gap: int,
        chance: float,
        seed: int,
    ):
        super().__init__(min_length, max_length, gap, chance, seed)
        self.min_multiplier = min_multiplier
        self.scale = scale

    def sample_random(self):
        variance_multiplier = self.min_multiplier + self._rng.lognormal(
            0, sigma=self.scale
        )
        return variance_multiplier

    def transform_interval(
        self, fault_interval: pd.DataFrame, target_col: str, random_value: float
    ) -> pd.Series:
        result = fault_interval[target_col].copy()
        np_col = result.to_numpy()
        variance = np.var(np_col)
        sigma_noise = np.sqrt((random_value - 1.0) * variance)
        noise = self._rng.normal(0, sigma_noise, len(np_col))
        result += noise
        return result


class StuckFault(IntervalFault):
    def __init__(
        self,
        stuck_value: Optional[float],
        min_length: int,
        max_length: int,
        gap: int,
        chance: float,
        seed: int,
    ):
        super().__init__(min_length, max_length, gap, chance, seed)
        self.stuck_value = stuck_value

    def sample_random(self):
        return 0

    def transform_interval(
        self, fault_interval: pd.DataFrame, target_col: str, random_value: float
    ) -> pd.Series:
        assert random_value == 0
        result = fault_interval[target_col].copy()
        value = self.stuck_value if self.stuck_value else result.iloc[0]
        return pd.Series(value, index=result.index, name=result.name)
