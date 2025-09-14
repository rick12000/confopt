from typing import Union
from pydantic import BaseModel, field_validator, ValidationInfo, ConfigDict
import numpy as np


class IntRange(BaseModel):
    """Range of integer values for hyperparameter optimization."""

    min_value: int
    max_value: int
    log_scale: bool = False  # Whether to sample on a logarithmic scale

    @field_validator("max_value")
    def max_gt_min(cls, v, info: ValidationInfo):
        if (
            hasattr(info, "data")
            and "min_value" in info.data
            and v <= info.data["min_value"]
        ):
            raise ValueError("max_value must be greater than min_value")
        return v

    @field_validator("log_scale")
    def log_scale_positive_values(cls, v, info: ValidationInfo):
        if (
            v
            and hasattr(info, "data")
            and "min_value" in info.data
            and info.data["min_value"] <= 0
        ):
            raise ValueError("log_scale=True requires min_value > 0")
        return v


class FloatRange(BaseModel):
    """Range of float values for hyperparameter optimization."""

    min_value: float
    max_value: float
    log_scale: bool = False  # Whether to sample on a logarithmic scale

    @field_validator("max_value")
    def max_gt_min(cls, v, info: ValidationInfo):
        if (
            hasattr(info, "data")
            and "min_value" in info.data
            and v <= info.data["min_value"]
        ):
            raise ValueError("max_value must be greater than min_value")
        return v


class CategoricalRange(BaseModel):
    """Categorical values for hyperparameter optimization."""

    choices: list[Union[str, int, float, bool]]

    @field_validator("choices")
    def non_empty_choices(cls, v):
        if len(v) == 0:
            raise ValueError("choices must not be empty")
        return v


ParameterRange = Union[IntRange, FloatRange, CategoricalRange]


class ConformalBounds(BaseModel):
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)
