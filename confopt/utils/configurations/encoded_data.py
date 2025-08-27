"""Custom data structure for encoded configurations with column metadata.

This module provides the EncodedData class which encapsulates both the encoded
configuration data and metadata about each column's type (continuous, one_hot, or level).
This eliminates the need for separate categorical information passing and provides
a clean interface for downstream processes.
"""

import logging
from typing import List, Optional
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ColumnType(Enum):
    """Enumeration of column types for ML training purposes."""

    CONTINUOUS = "continuous"
    ONE_HOT = "one_hot"
    LEVEL = "level"


class ColumnInfo:
    """Information about a single column."""

    def __init__(
        self, name: str, column_type: ColumnType, original_param: Optional[str] = None
    ):
        """Initialize column information.

        Args:
            name: Column name in the encoded data
            column_type: Type of the column (continuous, one_hot, or level)
            original_param: Original parameter name from search space (for categorical expansions)
        """
        self.name = name
        self.type = column_type
        self.original_param = original_param or name

    def __repr__(self):
        return f"ColumnInfo(name='{self.name}', type={self.type.value}, original_param='{self.original_param}')"


class EncodedData:
    """Custom data structure for encoded configurations with column metadata.

    This class encapsulates both the encoded data and metadata about each column,
    providing a clean interface for downstream ML processes to understand the
    nature of each feature without requiring separate metadata passing.

    Attributes:
        data: The encoded data as a pandas DataFrame
        columns: List of ColumnInfo objects describing each column
        column_types: Dictionary mapping column names to their types
    """

    def __init__(self, data: pd.DataFrame, columns: List[ColumnInfo]):
        """Initialize EncodedData with data and column metadata.

        Args:
            data: Encoded configuration data
            columns: List of ColumnInfo objects describing each column
        """
        if len(data.columns) != len(columns):
            raise ValueError(
                f"Number of data columns ({len(data.columns)}) must match number of column info objects ({len(columns)})"
            )

        self.data = data
        self.columns = columns
        self.column_types = {col.name: col.type for col in columns}

        # Create convenient access attributes
        self._continuous_columns = [
            col.name for col in columns if col.type == ColumnType.CONTINUOUS
        ]
        self._one_hot_columns = [
            col.name for col in columns if col.type == ColumnType.ONE_HOT
        ]
        self._level_columns = [
            col.name for col in columns if col.type == ColumnType.LEVEL
        ]

    @property
    def continuous_columns(self) -> List[str]:
        """Names of continuous columns."""
        return self._continuous_columns.copy()

    @property
    def one_hot_columns(self) -> List[str]:
        """Names of one-hot encoded columns."""
        return self._one_hot_columns.copy()

    @property
    def level_columns(self) -> List[str]:
        """Names of level-encoded columns."""
        return self._level_columns.copy()

    @property
    def categorical_columns(self) -> List[str]:
        """Names of all categorical columns (one-hot + level)."""
        return self._one_hot_columns + self._level_columns

    def get_continuous_data(self) -> pd.DataFrame:
        """Get only the continuous columns as a DataFrame."""
        if not self._continuous_columns:
            return pd.DataFrame(index=self.data.index)
        return self.data[self._continuous_columns]

    def get_categorical_data(self) -> pd.DataFrame:
        """Get only the categorical columns as a DataFrame."""
        categorical_cols = self.categorical_columns
        if not categorical_cols:
            return pd.DataFrame(index=self.data.index)
        return self.data[categorical_cols]

    def get_level_data(self) -> pd.DataFrame:
        """Get only the level-encoded columns as a DataFrame."""
        if not self._level_columns:
            return pd.DataFrame(index=self.data.index)
        return self.data[self._level_columns]

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.values

    def to_dataframe(self) -> pd.DataFrame:
        """Get the underlying DataFrame."""
        return self.data.copy()

    def get_column_mask(self, column_type: ColumnType) -> np.ndarray:
        """Get boolean mask for columns of specified type.

        Args:
            column_type: Type of columns to mask

        Returns:
            Boolean array where True indicates columns of the specified type
        """
        return np.array([col.type == column_type for col in self.columns])

    def get_is_categorical_mask(self) -> np.ndarray:
        """Get boolean mask indicating categorical columns (one-hot or level)."""
        return np.array(
            [col.type in (ColumnType.ONE_HOT, ColumnType.LEVEL) for col in self.columns]
        )

    def get_is_level_mask(self) -> np.ndarray:
        """Get boolean mask indicating level-encoded columns."""
        return self.get_column_mask(ColumnType.LEVEL)

    def slice_by_type(self, column_type: ColumnType) -> "EncodedData":
        """Create a new EncodedData with only columns of specified type.

        Args:
            column_type: Type of columns to include

        Returns:
            New EncodedData with filtered columns
        """
        mask = self.get_column_mask(column_type)
        filtered_columns = [col for col, include in zip(self.columns, mask) if include]
        filtered_data = self.data.iloc[:, mask]
        return EncodedData(filtered_data, filtered_columns)

    def __len__(self) -> int:
        """Number of rows in the data."""
        return len(self.data)

    def __getitem__(self, key):
        """Support indexing like a DataFrame."""
        return self.data[key]

    def __repr__(self):
        return f"EncodedData(shape={self.data.shape}, continuous={len(self._continuous_columns)}, one_hot={len(self._one_hot_columns)}, level={len(self._level_columns)})"
