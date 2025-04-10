import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union, List

@dataclass
class IntelArgs:
    start_date: Union[pd.Timestamp, str] = '2004-03-01'
    end_date: Union[pd.Timestamp, str] = '2004-03-07' 
    mote_id: List[str] = ['7-12']
    interval: str = '5min'
