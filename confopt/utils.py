import logging
import random
from typing import Dict, List, Optional, Any
import math

import numpy as np
import pandas as pd
from confopt.ranges import IntRange, FloatRange, CategoricalRange, ParameterRange

logger = logging.getLogger(__name__)


def get_tuning_configurations(
    parameter_grid: Dict[str, ParameterRange],
    n_configurations: int,
    random_state: Optional[int] = None,
    warm_start_configs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict]:
    """
    Randomly sample list of unique hyperparameter configurations.

    Each configuration is constructed from parameter ranges defined in the parameter grid.
    If warm start configurations are provided, they are included in the output.

    Parameters
    ----------
    parameter_grid :
        Dictionary of parameter names to their range definitions.
    n_configurations :
        Number of desired configurations to randomly construct.
    random_state :
        Random seed.
    warm_start_configs :
        Optional list of pre-defined configurations to include in the output.

    Returns
    -------
    configurations :
        Unique randomly constructed hyperparameter configurations including warm starts.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # Initialize with warm start configurations if provided
    if warm_start_configs:
        configurations = warm_start_configs.copy()
        # Create a set of hashable configurations for deduplication
        configurations_set = {
            tuple(
                sorted(
                    (k, str(v) if isinstance(v, (list, dict, set)) else v)
                    for k, v in config.items()
                )
            )
            for config in warm_start_configs
        }
    else:
        configurations = []
        configurations_set = set()

    # Calculate how many additional configurations we need
    n_additional = max(0, n_configurations - len(configurations))

    attempts = 0
    max_attempts = n_additional * 10  # Prevent infinite loops

    while len(configurations) < n_configurations and attempts < max_attempts:
        configuration = {}
        for parameter_name, parameter_range in parameter_grid.items():
            if isinstance(parameter_range, IntRange):
                # Sample integer from range
                value = random.randint(
                    parameter_range.min_value, parameter_range.max_value
                )
            elif isinstance(parameter_range, FloatRange):
                # Sample float from range, with optional log scaling
                if parameter_range.log_scale:
                    log_min = math.log(max(parameter_range.min_value, 1e-10))
                    log_max = math.log(parameter_range.max_value)
                    value = math.exp(random.uniform(log_min, log_max))
                else:
                    value = random.uniform(
                        parameter_range.min_value, parameter_range.max_value
                    )
            elif isinstance(parameter_range, CategoricalRange):
                # Sample from categorical choices
                value = random.choice(parameter_range.choices)
            else:
                raise TypeError(
                    f"Unsupported parameter range type: {type(parameter_range)}"
                )

            configuration[parameter_name] = value

        # Convert configuration to hashable representation for deduplication
        config_tuple = tuple(
            sorted(
                (k, str(v) if isinstance(v, (list, dict, set)) else v)
                for k, v in configuration.items()
            )
        )

        if config_tuple not in configurations_set:
            configurations_set.add(config_tuple)
            configurations.append(configuration)

        attempts += 1

    if len(configurations) < n_configurations:
        logger.warning(
            f"Could only generate {len(configurations)} unique configurations "
            f"out of {n_configurations} requested after {max_attempts} attempts."
        )

    return configurations


class ConfigurationEncoder:
    """
    Handles encoding and transformation of hyperparameter configurations.

    Maintains mappings for categorical features to ensure consistent one-hot encoding.
    """

    def __init__(self):
        self.categorical_mappings = {}  # {param_name: {value: column_index}}
        self.column_names = []

    def fit(self, configurations: List[Dict]) -> None:
        """Build mappings from a list of configurations."""
        # First pass: identify categorical parameters and their unique values
        categorical_values = {}

        for config in configurations:
            for param_name, value in config.items():
                if not isinstance(value, (int, float, bool)):
                    if param_name not in categorical_values:
                        categorical_values[param_name] = set()
                    categorical_values[param_name].add(value)

        # Create mappings for categorical features
        col_idx = 0
        for param_name in sorted(configurations[0].keys()):
            if param_name in categorical_values:
                # Categorical parameter
                self.categorical_mappings[param_name] = {}
                sorted_values = sorted(categorical_values[param_name], key=str)
                for value in sorted_values:
                    column_name = f"{param_name}_{value}"
                    self.categorical_mappings[param_name][value] = col_idx
                    self.column_names.append(column_name)
                    col_idx += 1
            else:
                # Numeric parameter
                self.column_names.append(param_name)
                col_idx += 1

    def transform(self, configurations: List[Dict]) -> pd.DataFrame:
        """Transform configurations into a tabular format with proper encoding."""
        if not self.column_names:
            self.fit(configurations)

        n_samples = len(configurations)
        n_features = len(self.column_names)
        X = np.zeros((n_samples, n_features))

        # Fill in the feature matrix
        for i, config in enumerate(configurations):
            col_idx = 0
            for param_name in sorted(config.keys()):
                value = config[param_name]

                if param_name in self.categorical_mappings:
                    # Handle categorical parameter with one-hot encoding
                    if value in self.categorical_mappings[param_name]:
                        one_hot_idx = self.categorical_mappings[param_name][value]
                        X[i, one_hot_idx] = 1
                    else:
                        # Handle unseen categorical value - could raise error or skip
                        logger.warning(
                            f"Unseen categorical value {value} for parameter {param_name}"
                        )

                    # Skip ahead by the number of categories for this parameter
                    col_idx += len(self.categorical_mappings[param_name])
                else:
                    # Handle numeric parameter
                    X[i, col_idx] = value
                    col_idx += 1

        return pd.DataFrame(X, columns=self.column_names)
