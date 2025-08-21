import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from pydantic_settings import BaseSettings
from typing import Dict, Optional

class BaseAdaptor(ABC):
    def __init__(self, path: Path):
        self.path = path
    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass

class AdaptorConfig(BaseSettings):
    path: Path
    rename_columns: Optional[Dict[str, str]] = None


