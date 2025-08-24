from datetime import datetime
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Optional, Union


class AdaptorConfig(BaseModel):
    path: Path
    period: Optional[List[datetime]] = None
    subset_node: Optional[List[int]] = None
    resample_interval: Optional[str] = None
    rename_columns: Optional[Dict[str, str]] = None
