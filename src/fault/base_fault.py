import random
import pandas as pd
from typing import Dict, List, Union
from abc import ABC, abstractmethod

class BaseFault(ABC):
    def __init__(self, chance: float, params: Dict[str, float]):
        self.params = params
        self.chance = chance
    
    def _calculate_target_points(self, data: pd.DataFrame) -> int:
        return max(1, int(len(data) * self.chance))
    
    @abstractmethod
    def apply(self, data: pd.DataFrame, type_index: int) -> pd.DataFrame:
        pass

class InstantFault(BaseFault):
    
    def _calculate_fault_indices(self, data: pd.DataFrame) -> List[int]:
        target_points = self._calculate_target_points(data)
        target_points = min(target_points, len(data))
        
        all_indices = list(range(len(data)))
        faulty_indices = random.sample(all_indices, target_points)
        
        return sorted(faulty_indices)
    
    @abstractmethod
    def _apply_to_point(self, point: pd.Series, col: List[str]) -> pd.Series:
        pass
    
    def apply(self, data: pd.DataFrame, type_index: int) -> pd.DataFrame:
        indices = self._calculate_fault_indices(data)
        result = data.copy()
        
        result['type'] = 0
        result['type'] = result['type'].astype('int16')
        for idx in indices:
            result.iloc[idx] = self._apply_to_point(result.iloc[idx], ['temperature'])
            result.loc[idx, 'type'] = type_index
            
        return result

class PeriodFault(BaseFault):
    def __init__(
            self, 
            max_duration: Union[pd.Timedelta, int], 
            min_duration: Union[pd.Timedelta, int], 
            chance: float, 
            params: Dict[str, float]
        ):
        super().__init__(chance, params)

        if isinstance(max_duration, int):
            max_duration = pd.Timedelta(max_duration, unit='s')
        if isinstance(min_duration, int):
            min_duration = pd.Timedelta(min_duration, unit='s')

        self.max_duration = max_duration
        self.min_duration = min_duration

    def _calculate_fault_periods(self, data: pd.DataFrame) -> List[List[int]]:
        target_points = self._calculate_target_points(data)
        current_points = 0
        faulty_indices = set()

        dts = data['datetime']
        start_dt = dts.min()
        end_dt = dts.max()
        
        max_attempts = len(data) * 2
        attempts = 0

        while current_points < target_points and attempts < max_attempts:
            attempts += 1
            
            duration = pd.Timedelta(seconds=random.uniform(
                self.min_duration.total_seconds(),
                self.max_duration.total_seconds()
            ))

            max_start = end_dt - duration
            if max_start < start_dt:
                continue  

            start = start_dt + pd.Timedelta(seconds=random.uniform(
                0, (max_start - start_dt).total_seconds()
            ))
            end = start + duration

            mask = (dts >= start) & (dts <= end)
            period_indices = data[mask].index.tolist()

            if not period_indices:
                continue

            new_indices = set(period_indices) - faulty_indices
            if not new_indices:
                continue

            if current_points + len(new_indices) > target_points:
                needed = target_points - current_points
                new_indices = set(sorted(new_indices)[:needed])

            faulty_indices.update(new_indices)
            current_points += len(new_indices)

        return self._group_indices_into_periods(sorted(faulty_indices))
    
    def _group_indices_into_periods(self, sorted_indices: List[int]) -> List[List[int]]:
        periods = []
        current_period = []

        for idx in sorted_indices:
            if not current_period:
                current_period.append(idx)
            elif idx == current_period[-1] + 1:
                current_period.append(idx)
            else:
                periods.append(current_period)
                current_period = [idx]

        if current_period:
            periods.append(current_period)

        return periods
    
    @abstractmethod
    def _apply_to_period(self, period_data: pd.DataFrame, col: List[str]) -> pd.DataFrame:
        pass
    
    def apply(self, data: pd.DataFrame, type_index: int) -> pd.DataFrame:
        periods = self._calculate_fault_periods(data)
        result = data.copy()
        
        result['type'] = 0
        result['type'] = result['type'].astype('int16')
        for period in periods:
            if not period:
                continue
                
            period_data = result.iloc[period]
            modified_period = self._apply_to_period(period_data, ['temperature'])
            
            for i, idx in enumerate(period):
                result.iloc[idx] = modified_period.iloc[i]
                result.loc[idx, 'type'] = type_index
            
        return result
