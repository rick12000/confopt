import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from confopt.wrapping import CategoricalRange, ParameterRange

logger = logging.getLogger(__name__)


class ConfigurationEncoder:
    def __init__(self, search_space: Dict[str, ParameterRange]):
        self.search_space = search_space
        self.categorical_mappings = {}
        self.column_names = []
        self._build_encoding_schema()

    def transform(self, configurations: List[Dict]) -> pd.DataFrame:
        feature_matrix = self._create_feature_matrix(configurations)
        return pd.DataFrame(data=feature_matrix, columns=self.column_names)

    def _build_encoding_schema(self) -> None:
        self.categorical_mappings = {}
        self.column_names = []

        for param_name in sorted(self.search_space.keys()):
            param_range = self.search_space[param_name]

            if isinstance(param_range, CategoricalRange):
                self._add_categorical_columns(param_name, param_range.choices)
            else:
                self.column_names.append(param_name)

    def _add_categorical_columns(self, param_name: str, choices: List) -> None:
        """Add one-hot encoded columns for a categorical parameter."""
        sorted_values = sorted(choices, key=str)
        param_mappings = {}

        for value in sorted_values:
            column_idx = len(self.column_names)
            column_name = f"{param_name}_{value}"
            param_mappings[value] = column_idx
            self.column_names.append(column_name)

        self.categorical_mappings[param_name] = param_mappings

    def _create_feature_matrix(self, configurations: List[Dict]) -> np.ndarray:
        """Create numerical feature matrix from configurations."""
        n_samples = len(configurations)
        n_features = len(self.column_names)
        feature_matrix = np.zeros((n_samples, n_features))

        for row_idx, config in enumerate(configurations):
            self._encode_single_config(config, feature_matrix, row_idx)

        return feature_matrix

    def _encode_single_config(
        self, config: Dict, feature_matrix: np.ndarray, row_idx: int
    ) -> None:
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
