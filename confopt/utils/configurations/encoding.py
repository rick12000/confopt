import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from confopt.wrapping import CategoricalRange, ParameterRange

logger = logging.getLogger(__name__)


class ConfigurationEncoder:
    """
    Encodes configuration dictionaries into numerical feature matrices.

    The encoder supports both continuous and categorical parameters, using one-hot
    encoding for categorical variables. The encoding schema is constructed from a
    provided search space and is deterministic, ensuring reproducibility across runs.
    Intended for use in hyperparameter optimization workflows where explicit and
    consistent feature representation is required.

    Args:
        search_space (Dict[str, ParameterRange]):
            Dictionary mapping parameter names to their respective ParameterRange objects.
            Categorical parameters must use CategoricalRange.
    """

    def __init__(self, search_space: Dict[str, ParameterRange]):
        """
        Initialize the encoder and build the encoding schema from the search space.

        Args:
            search_space (Dict[str, ParameterRange]):
                Parameter search space definition.
        """
        self.search_space = search_space
        self.categorical_mappings = {}
        self.column_names = []
        self._build_encoding_schema()

    def transform(self, configurations: List[Dict]) -> pd.DataFrame:
        """
        Transform a list of configuration dictionaries into a numerical DataFrame.

        Args:
            configurations (List[Dict]):
                List of configuration dictionaries, each mapping parameter names to values.

        Returns:
            pd.DataFrame: Feature matrix with columns corresponding to the encoding schema.
        """
        feature_matrix = self._create_feature_matrix(configurations)
        return pd.DataFrame(data=feature_matrix, columns=self.column_names)

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
        Add one-hot encoded columns for a categorical parameter.

        Args:
            param_name (str): Name of the categorical parameter.
            choices (List): List of possible categorical values.
        """
        sorted_values = sorted(choices, key=str)
        param_mappings = {}

        for value in sorted_values:
            column_idx = len(self.column_names)
            column_name = f"{param_name}_{value}"
            param_mappings[value] = column_idx
            self.column_names.append(column_name)

        self.categorical_mappings[param_name] = param_mappings

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
                    one_hot_idx = self.categorical_mappings[param_name][param_value]
                    feature_matrix[row_idx, one_hot_idx] = 1
                column_idx += len(self.categorical_mappings[param_name])
            else:
                feature_matrix[row_idx, column_idx] = param_value
                column_idx += 1
