from pydantic_settings import BaseSettings
from typing import Optional

class InjectConfig(BaseSettings):
    dataset: str
    chance: float
    seed: Optional[int]
