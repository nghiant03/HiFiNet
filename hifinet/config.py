from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class AdaptorConfig(BaseModel):
    path: Path
    period: list[datetime] | None = None
    subset_node: list[int] | None = None
    resample_interval: str | None = None
    rename_columns: dict[str, str] | None = None


class _IntervalParams(BaseModel):
    min_length: int
    max_length: int
    gap: int = Field(ge=0)
    chance: float = Field(ge=0.0, le=1.0)
    seed: int | None = None

    @model_validator(mode="after")
    def _check_interval_params(self):
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")
        return self


class HardoverFaultConfig(_IntervalParams):
    bias_range: list[float]

    @field_validator("bias_range")
    @classmethod
    def _bias_range_two_positive(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("bias_range must be length-2: [low, high]")
        low, high = v
        if low < 0 or high < 0:
            raise ValueError("bias_range values must be >= 0")
        if low >= high:
            raise ValueError("bias_range must satisfy low < high")
        return v


class DriftFaultConfig(_IntervalParams):
    sigma: float = Field(gt=0)
    min_drift: float = Field(ge=0)


class SpikeFaultConfig(BaseModel):
    bias_range: list[float]
    chance: float = Field(ge=0.0, le=1.0)
    seed: int | None = None

    @field_validator("bias_range")
    @classmethod
    def _bias_range_two_positive(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("bias_range must be length-2: [low, high]")
        low, high = v
        if low < 0 or high < 0:
            raise ValueError("bias_range values must be >= 0")
        if low >= high:
            raise ValueError("bias_range must satisfy low < high")
        return v


class ErraticFaultConfig(_IntervalParams):
    min_multiplier: float = Field(ge=1.0)
    scale: float = Field(gt=0)


class StuckFaultConfig(_IntervalParams):
    stuck_value: float | None = Field(
        None,
    )


FaultConfig = (
    HardoverFaultConfig
    | DriftFaultConfig
    | SpikeFaultConfig
    | ErraticFaultConfig
    | StuckFaultConfig
)


class InjectorConfig(BaseModel):
    fault_config: dict[str, FaultConfig]
    type_mapping: dict[str, int] = {
        "hardover": 1,
        "drift": 2,
        "spike": 3,
        "erratic": 4,
        "stuck": 5,
    }
    mode: str = "singular"
    exclude: list[int] | None = None


DEFAULT_HARDOVER = HardoverFaultConfig(
    min_length=5,
    max_length=12,
    gap=2,
    chance=0.2,
    bias_range=[7.0, 10.0],
)

DEFAULT_DRIFT = DriftFaultConfig(
    min_length=5,
    max_length=12,
    gap=2,
    chance=0.2,
    sigma=0.01,
    min_drift=0.1,
)

DEFAULT_SPIKE = SpikeFaultConfig(
    bias_range=[3.0, 5.0],
    chance=0.2,
)

DEFAULT_ERRATIC = ErraticFaultConfig(
    min_length=5,
    max_length=12,
    gap=2,
    chance=0.2,
    min_multiplier=1.5,
    scale=0.1,
)

DEFAULT_STUCK = StuckFaultConfig(
    min_length=5,
    max_length=12,
    gap=2,
    chance=0.2,
    stuck_value=None,
)


NAME_CONFIG_MAPPING = (
    {
        "hardover": DEFAULT_HARDOVER,
        "drift": DEFAULT_DRIFT,
        "spike": DEFAULT_SPIKE,
        "erratic": DEFAULT_ERRATIC,
        "stuck": DEFAULT_STUCK,
    },
)
