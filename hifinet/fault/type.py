import numpy as np
import pandas as pd
from typing import List, Optional

from hifinet.fault.base import InstantFault, IntervalFault

class HardoverFault(IntervalFault):
    def __init__(self, min_length: int, max_length: int, gap: int, bias_range: List[float], chance: float, seed: int):
        super().__init__(min_length, max_length, gap, chance, seed)
        self.bias_range = bias_range

    def _sample_random(self):
        bias = self._rng.uniform(self.bias_range[0], self.bias_range[1])
        bias = self._rng.choice([-1, 1]) * bias
        return bias

    def _transform_interval(self, fault_interval: pd.DataFrame, target_col: str, random_value: float) -> pd.DataFrame:
        result = fault_interval.copy()
        result[target_col] += random_value
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

    def _sample_random(self):
        initial_value = self._rng.normal(0, self.sigma)
        initial_value = initial_value + self.min_drift if initial_value > 0 else initial_value - self.min_drift
        return initial_value
    
    def _transform_interval(self, fault_interval: pd.DataFrame, target_col: str, random_value: float) -> pd.DataFrame:
        result = fault_interval.copy()
        n_points = len(result)

        fault_values = [n * random_value for n in range(n_points)]
        result[target_col] += fault_values
        return result

class SpikeFault(InstantFault):
    def __init__(self, bias_range: List[float],chance: float, seed: int):
        super().__init__(chance, seed)
        self.bias_range = bias_range
    
    def _sample_random(self):
        bias = self._rng.uniform(self.bias_range[0], self.bias_range[1])
        bias = self._rng.choice([-1, 1]) * bias
        return bias

    def _transform_point(self, fault_point: pd.Series, target_col: str, random_value: float) -> pd.Series:
        result = fault_point.copy()
        result[target_col] += random_value
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
            seed: int
        ):
        super().__init__(min_length, max_length, gap, chance, seed)
        self.min_multiplier = min_multiplier
        self.scale = scale
    
    def _sample_random(self):
        variance_multiplier = self.min_multiplier + self._rng.lognormal(0, sigma=self.scale)
        return variance_multiplier

    def _transform_interval(self, fault_interval: pd.DataFrame, target_col: str, random_value: float) -> pd.DataFrame:
        result = fault_interval.copy()
        np_col = result[target_col].to_numpy()
        variance = np.var(np_col)
        sigma_noise = np.sqrt((random_value - 1.0) * variance)
        noise = self._rng.normal(0, sigma_noise, len(np_col))
        result[target_col] += noise
        return result

class StuckFault(IntervalFault):
    def __init__(
            self, 
            stuck_value: Optional[float],
            min_length: int,
            max_length: int, 
            gap: int,
            chance: float, 
            seed: int
        ):
        super().__init__(min_length, max_length, gap, chance, seed)
        self.stuck_value = stuck_value

    def _sample_random(self):
        return 0

    def _transform_interval(self, fault_interval: pd.DataFrame, target_col: str, random_value: float) -> pd.DataFrame:
        assert random_value == 0
        result = fault_interval.copy()
        result[target_col] = self.stuck_value if self.stuck_value else result[target_col].iloc[0]
        return result
