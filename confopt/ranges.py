from typing import List, TypeVar, Union, Generic
from pydantic import BaseModel, validator

T = TypeVar("T")


class IntRange(BaseModel):
    """Range of integer values for hyperparameter optimization."""

    min_value: int
    max_value: int

    @validator("max_value")
    def max_gt_min(cls, v, values):
        if "min_value" in values and v <= values["min_value"]:
            raise ValueError("max_value must be greater than min_value")
        return v


class FloatRange(BaseModel):
    """Range of float values for hyperparameter optimization."""

    min_value: float
    max_value: float
    log_scale: bool = False  # Whether to sample on a logarithmic scale

    @validator("max_value")
    def max_gt_min(cls, v, values):
        if "min_value" in values and v <= values["min_value"]:
            raise ValueError("max_value must be greater than min_value")
        return v


class CategoricalRange(BaseModel, Generic[T]):
    """Categorical values for hyperparameter optimization."""

    choices: List[T]

    @validator("choices")
    def non_empty_choices(cls, v):
        if len(v) == 0:
            raise ValueError("choices must not be empty")
        return v


ParameterRange = Union[IntRange, FloatRange, CategoricalRange]
