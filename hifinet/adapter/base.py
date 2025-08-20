import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod

class BaseAdaptor(ABC):
    def __init__(self, path: Path):
        self.path = path
    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass
