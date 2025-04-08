import random
import pandas as pd
import numpy as np
from typing import Union, List, Optional

from .base_fault import InstantFault, PeriodFault

class HardoverFault(InstantFault):
    """
    Hard-over fault adds a high constant bias value to all non-faulty signal elements.
    S_hardover = S_normal + b, where b = constant
    """
    
    def __init__(self, chance: float, bias_value: float):
        super().__init__(chance, {'bias_value': bias_value})
    
    def _apply_to_point(self, point: pd.Series, col: List[str]) -> pd.Series:
        """Apply hard-over fault to a single data point"""
        result = point.copy()
        result[col] += self.params['bias_value']
        return result


class DriftFault(PeriodFault):
    """
    Drift fault appears when the output signal keeps increasing linearly over time.
    S_drift = S_normal + b_n, where b_n = n*b_0, b_0 = constant, n is the index
    """
    
    def __init__(
            self, 
            max_duration: Union[pd.Timedelta, int], 
            min_duration: Union[pd.Timedelta, int], 
            chance: float, 
            initial_bias: float
        ):
        super().__init__(max_duration, min_duration, chance, {'initial_bias': initial_bias})
    
    def _apply_to_period(self, period_data: pd.DataFrame, col: List[str]) -> pd.DataFrame:
        """Apply drift fault to a period of data"""
        result = period_data.copy()
        n_points = len(result)
        
        bias_values = [n * self.params['initial_bias'] for n in range(n_points)]
        bias_array = np.array(bias_values).reshape(-1, 1)  
        
        bias_array = np.tile(bias_array, (1, len(col)))  
        result[col] += bias_array
        
        return result


class SpikeFault(InstantFault):
    """
    Spike fault is observed intermittently in the form of high-amplitude spikes.
    S_spike = S_normal + b_n, where n = v × η is the elements index in the signal,
    v = (1, 2, . . . ) as natural numbers, and η ≥ 2 as a positive integer.
    """
    
    def __init__(self, chance: float, spike_value: float, eta: int = 2):
        super().__init__(chance, {'spike_value': spike_value, 'eta': eta})
        self.spike_value = spike_value
        self.eta = max(2, eta)
    
    def _calculate_fault_indices(self, data: pd.DataFrame) -> List[int]:
        """Override to ensure spikes occur at intervals specified by eta"""
        target_points = self._calculate_target_points(data)
        target_points = min(target_points, len(data) // self.eta)
        
        max_i = (len(data) - 1) // self.eta
        all_indices = [i * self.eta for i in range(1, max_i + 1)]
        
        if not all_indices:
            return []
            
        if target_points < len(all_indices):
            faulty_indices = random.sample(all_indices, target_points)
        else:
            faulty_indices = all_indices
            
        return sorted(faulty_indices)
    
    def _apply_to_point(self, point: pd.Series, col: List[str]) -> pd.Series:
        """Apply spike fault to a single data point"""
        result = point.copy()
        result[col] += self.spike_value
        return result


class ErraticFault(PeriodFault):
    """
    Erratic/precision degradation fault causes the sensor's output variance 
    to increase significantly above the usual state over a period of time.
    S_erratic = S_normal + S_n, where S_n ~ N(0, δ²), δ² >> δ²_normal
    """
    
    def __init__(
            self, 
            max_duration: Union[pd.Timedelta, int], 
            min_duration: Union[pd.Timedelta, int], 
            chance: float, 
            variance_multiplier: float
        ):
        """
        Args:
            max_duration: Maximum duration of erratic period
            min_duration: Minimum duration of erratic period
            chance: Probability of a period being affected
            variance_multiplier: How much higher the variance should be compared to normal (δ² >> δ²_normal)
        """
        super().__init__(max_duration, min_duration, chance, {'variance_multiplier': variance_multiplier})
        self.variance_multiplier = variance_multiplier
        self._column_variances = {}
    
    def _apply_to_period(self, period_data: pd.DataFrame, col: List[str]) -> pd.DataFrame:
        """Apply erratic fault to a period of data"""
        result = period_data.copy()

        for c in col:
            self._column_variances[c] = result[c].var()
            fault_variance = self._column_variances[c] * self.variance_multiplier
            noise = np.random.normal(0, np.sqrt(fault_variance), size=len(result))
            result.loc[:, c] += noise

        return result

class StuckFault(PeriodFault):
    """
    Stuck fault causes nil or almost nil variations in the output signal.
    In case of complete failure, the output is stuck persistently at a constant value.
    S_stuck = α, where α = constant
    """
    
    def __init__(
            self, 
            max_duration: Union[pd.Timedelta, int], 
            min_duration: Union[pd.Timedelta, int], 
            chance: float, 
            stuck_value: Optional[float] = None
        ):
        """
        Args:
            max_duration: Maximum duration of stuck period
            min_duration: Minimum duration of stuck period
            chance: Probability of a period being affected
            stuck_value: Value to which signal gets stuck. If None, uses first value of period.
        """
        params = {'stuck_value': stuck_value} if stuck_value else {}
        super().__init__(max_duration, min_duration, chance, params)
    
    def _apply_to_period(self, period_data: pd.DataFrame, col: List[str]) -> pd.DataFrame:
        """Apply stuck fault to a period of data"""
        result = period_data.copy()
        
        value = self.params['stuck_value'] if 'stuck_value' in self.params else result[col].iloc[0].item()
        result.loc[:, col] = value
        
        return result
