import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from confopt.wrapping import CategoricalRange, ParameterRange
from confopt.utils.configurations.encoded_data import (
    EncodedData,
    ColumnInfo,
    ColumnType,
)

logger = logging.getLogger(__name__)


class ConfigurationEncoder:
    """
    Encodes configuration dictionaries into numerical feature matrices.

    The encoder supports both continuous and categorical parameters. For categorical
    variables, it can use either one-hot encoding or categorical level
    encoding depending on the encoding method. QuantileGP uses categorical level
    encoding to enable proper Hamming distance kernels, while other estimators
    use one-hot encoding for compatibility.

    Args:
        search_space (Dict[str, ParameterRange]):
            Dictionary mapping parameter names to their respective ParameterRange objects.
            Categorical parameters must use CategoricalRange.
        encoding_method (str):
            Encoding strategy for categorical variables. Either "one_hot" for
            one-hot encoding or "level" for categorical level encoding.
    """

    def __init__(
        self, search_space: Dict[str, ParameterRange], encoding_method: str = "one_hot"
    ):
        """
        Initialize the encoder and build the encoding schema from the search space.

        Args:
            search_space (Dict[str, ParameterRange]):
                Parameter search space definition.
            encoding_method (str):
                Encoding strategy for categorical variables.
        """
        self.search_space = search_space
        self.encoding_method = encoding_method
        self.categorical_mappings = {}
        self.column_names = []
        self.use_categorical_levels = encoding_method == "level"
        self._build_encoding_schema()

    def transform(self, configurations: List[Dict]) -> EncodedData:
        """
        Transform a list of configuration dictionaries into an EncodedData object.

        Args:
            configurations (List[Dict]):
                List of configuration dictionaries, each mapping parameter names to values.

        Returns:
            EncodedData: Encoded data with column metadata.
        """
        feature_matrix = self._create_feature_matrix(configurations)
        data = pd.DataFrame(data=feature_matrix, columns=self.column_names)
        columns = self._build_column_info()
        return EncodedData(data, columns)

    def _build_encoding_schema(self) -> None:
        """
        Construct the encoding schema and categorical mappings from the search space.

        Ensures deterministic column ordering and explicit one-hot encoding for
        categorical parameters.
        """
        self.categorical_mappings = {}
        self.column_names = []

        for param_name in sorted(self.search_space.keys()):
            param_range = self.search_space[param_name]

            if isinstance(param_range, CategoricalRange):
                self._add_categorical_columns(param_name, param_range.choices)
            else:
                self.column_names.append(param_name)

    def _add_categorical_columns(self, param_name: str, choices: List) -> None:
        """
        Add columns for a categorical parameter.

        Uses either categorical level encoding (single column with integer levels)
        or one-hot encoding (multiple binary columns) depending on estimator type.

        Args:
            param_name (str): Name of the categorical parameter.
            choices (List): List of possible categorical values.
        """
        sorted_values = sorted(choices, key=str)
        param_mappings = {}

        if self.use_categorical_levels:
            # Categorical level encoding: single column with integer levels
            self.column_names.append(param_name)

            # Map categorical values to integer levels (0, 1, 2, ...)
            for i, value in enumerate(sorted_values):
                param_mappings[value] = i
        else:
            # One-hot encoding: multiple binary columns
            for value in sorted_values:
                column_idx = len(self.column_names)
                column_name = f"{param_name}_{value}"
                param_mappings[value] = column_idx
                self.column_names.append(column_name)

        self.categorical_mappings[param_name] = param_mappings

    def _build_column_info(self) -> List[ColumnInfo]:
        """
        Build column information for each column in the encoded data.

        Returns:
            List of ColumnInfo objects describing each column.
        """
        column_info = []

        for param_name in sorted(self.search_space.keys()):
            param_range = self.search_space[param_name]

            if isinstance(param_range, CategoricalRange):
                if self.use_categorical_levels:
                    # Level encoding: single column
                    column_info.append(
                        ColumnInfo(param_name, ColumnType.LEVEL, param_name)
                    )
                else:
                    # One-hot encoding: multiple columns
                    for value in sorted(param_range.choices, key=str):
                        column_name = f"{param_name}_{value}"
                        column_info.append(
                            ColumnInfo(column_name, ColumnType.ONE_HOT, param_name)
                        )
            else:
                # Continuous parameter
                column_info.append(
                    ColumnInfo(param_name, ColumnType.CONTINUOUS, param_name)
                )

        return column_info

    def _create_feature_matrix(self, configurations: List[Dict]) -> np.ndarray:
        """
        Create a numerical feature matrix from a list of configurations.

        Args:
            configurations (List[Dict]):
                List of configuration dictionaries.

        Returns:
            np.ndarray: 2D array of shape (n_samples, n_features) with encoded features.
        """
        n_samples = len(configurations)
        n_features = len(self.column_names)
        feature_matrix = np.zeros((n_samples, n_features))

        for row_idx, config in enumerate(configurations):
            self._encode_single_config(config, feature_matrix, row_idx)

        return feature_matrix

    def _encode_single_config(
        self, config: Dict, feature_matrix: np.ndarray, row_idx: int
    ) -> None:
        """
        Encode a single configuration into the feature matrix row.

        Args:
            config (Dict): Configuration dictionary for a single sample.
            feature_matrix (np.ndarray): Feature matrix to populate.
            row_idx (int): Row index for the current configuration.
        """
        column_idx = 0

        for param_name in sorted(config.keys()):
            param_value = config[param_name]

            if param_name in self.categorical_mappings:
                if param_value in self.categorical_mappings[param_name]:
                    if self.use_categorical_levels:
                        # Categorical level encoding: set integer level
                        level_value = self.categorical_mappings[param_name][param_value]
                        feature_matrix[row_idx, column_idx] = level_value
                        column_idx += 1
                    else:
                        # One-hot encoding: set binary indicator
                        one_hot_idx = self.categorical_mappings[param_name][param_value]
                        feature_matrix[row_idx, one_hot_idx] = 1
                        column_idx += len(self.categorical_mappings[param_name])
                else:
                    # Handle unknown categorical value
                    if self.use_categorical_levels:
                        column_idx += 1
                    else:
                        column_idx += len(self.categorical_mappings[param_name])
            else:
                feature_matrix[row_idx, column_idx] = param_value
                column_idx += 1
