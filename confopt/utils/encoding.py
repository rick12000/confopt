import logging
import random
from typing import Dict, List, Optional, Any, Literal, Set, Tuple

import numpy as np
import pandas as pd
from confopt.wrapping import IntRange, FloatRange, CategoricalRange, ParameterRange

try:
    from scipy.stats import qmc

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


def get_tuning_configurations(
    parameter_grid: Dict[str, ParameterRange],
    n_configurations: int,
    random_state: Optional[int] = None,
    warm_start_configs: Optional[List[Dict[str, Any]]] = None,
    sampling_method: Literal["uniform", "sobol"] = "uniform",
) -> List[Dict]:
    """
    Sample list of unique hyperparameter configurations using the specified sampling method.

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
    sampling_method :
        Method to use for sampling parameter configurations. Options:
        - "uniform": Use uniform random sampling (default)
        - "sobol": Use Sobol sequence sampling for better space coverage

    Returns
    -------
    configurations :
        Unique hyperparameter configurations including warm starts.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # Initialize with warm start configurations if provided
    configurations, configurations_set = _process_warm_starts(warm_start_configs)

    # Calculate how many additional configurations we need
    n_additional = max(0, n_configurations - len(configurations))

    if n_additional > 0:
        # For efficiency, use uniform sampling for most cases
        # Only use Sobol for specific cases where it's most beneficial
        if (
            sampling_method == "sobol" and n_additional > 50
        ):  # Only use Sobol for larger samples
            if not HAS_SCIPY:
                logger.warning(
                    "Sobol sampling requested but scipy is not available. Falling back to uniform sampling."
                )
                return _uniform_sampling(
                    parameter_grid,
                    configurations,
                    configurations_set,
                    n_configurations,
                    random_state,
                )
            else:
                return _sobol_sampling(
                    parameter_grid,
                    configurations,
                    configurations_set,
                    n_configurations,
                    random_state,
                )
        else:  # "uniform" or any other value defaults to uniform
            return _uniform_sampling(
                parameter_grid,
                configurations,
                configurations_set,
                n_configurations,
                random_state,
            )

    return configurations


def _process_warm_starts(
    warm_start_configs: Optional[List[Dict[str, Any]]]
) -> Tuple[List[Dict], Set[Tuple]]:
    """Process warm start configurations and return configurations and their hashable set"""
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

    return configurations, configurations_set


def _uniform_sampling(
    parameter_grid: Dict[str, ParameterRange],
    configurations: List[Dict],
    configurations_set: set,
    n_configurations: int,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """Helper function to perform uniform random sampling of parameter configurations."""
    # Calculate how many additional configurations we need
    n_additional = max(0, n_configurations - len(configurations))

    # Optimization: Generate configurations in batches
    batch_size = min(n_additional * 2, 10000)  # Use reasonable batch size
    param_names = sorted(parameter_grid.keys())

    # Group parameters by type for vectorized operations
    int_params = []
    float_params = []
    log_float_params = []
    categorical_params = []

    for name in param_names:
        param_range = parameter_grid[name]
        if isinstance(param_range, IntRange):
            int_params.append((name, param_range))
        elif isinstance(param_range, FloatRange):
            if param_range.log_scale:
                log_float_params.append((name, param_range))
            else:
                float_params.append((name, param_range))
        elif isinstance(param_range, CategoricalRange):
            categorical_params.append((name, param_range))

    # Generate configurations until we have enough or reach max attempts
    max_attempts = min(int(n_additional * 5), 50000)
    attempts = 0

    while len(configurations) < n_configurations and attempts < max_attempts:
        current_batch_size = min(batch_size, max_attempts - attempts)
        batch_configs = []

        # Create skeleton for batch configurations
        batch_configs = [{} for _ in range(current_batch_size)]

        # Fill configurations with vectorized operations
        # Handle integer parameters
        for name, param_range in int_params:
            values = np.random.randint(
                param_range.min_value,
                param_range.max_value + 1,
                size=current_batch_size,
            )
            for i, value in enumerate(values):
                batch_configs[i][name] = int(value)

        # Handle float parameters with linear scale
        for name, param_range in float_params:
            values = np.random.uniform(
                param_range.min_value, param_range.max_value, size=current_batch_size
            )
            for i, value in enumerate(values):
                batch_configs[i][name] = float(value)

        # Handle float parameters with log scale
        for name, param_range in log_float_params:
            log_min = np.log(max(param_range.min_value, 1e-10))
            log_max = np.log(param_range.max_value)
            log_values = np.random.uniform(log_min, log_max, size=current_batch_size)
            values = np.exp(log_values)
            for i, value in enumerate(values):
                batch_configs[i][name] = float(value)

        # Handle categorical parameters
        for name, param_range in categorical_params:
            choices = param_range.choices
            # Pre-generate all choices
            indices = np.random.randint(0, len(choices), size=current_batch_size)
            for i, idx in enumerate(indices):
                batch_configs[i][name] = choices[idx]

        # Add unique configurations from batch
        for config in batch_configs:
            config_tuple = tuple(
                sorted(
                    (k, str(v) if isinstance(v, (list, dict, set)) else v)
                    for k, v in config.items()
                )
            )

            if config_tuple not in configurations_set:
                configurations_set.add(config_tuple)
                configurations.append(config)

                if len(configurations) >= n_configurations:
                    break

        attempts += current_batch_size

    if len(configurations) < n_configurations:
        logger.warning(
            f"Could only generate {len(configurations)} unique configurations "
            f"out of {n_configurations} requested after {attempts} attempts."
        )

    return configurations


def _sobol_sampling(
    parameter_grid: Dict[str, ParameterRange],
    configurations: List[Dict],
    configurations_set: set,
    n_configurations: int,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """Helper function to perform Sobol sequence sampling of parameter configurations."""
    # Calculate how many additional configurations we need
    n_additional = max(0, n_configurations - len(configurations))

    # Set up parameter ordering for consistent handling
    param_names = sorted(parameter_grid.keys())
    param_ranges = [parameter_grid[name] for name in param_names]

    # Count how many dimensions we need for Sobol sampling
    # (categorical parameters need to be handled differently)
    numeric_params = []
    categorical_params = []

    for i, (name, param_range) in enumerate(zip(param_names, param_ranges)):
        if isinstance(param_range, (IntRange, FloatRange)):
            numeric_params.append((i, name, param_range))
        elif isinstance(param_range, CategoricalRange):
            categorical_params.append((i, name, param_range))
        else:
            raise TypeError(f"Unsupported parameter range type: {type(param_range)}")

    # Create Sobol sampler
    n_dimensions = len(numeric_params)
    if n_dimensions == 0:
        # If no numeric dimensions, fall back to uniform sampling
        logger.info(
            "No numeric parameters found for Sobol sampling, falling back to uniform sampling."
        )
        return _uniform_sampling(
            parameter_grid,
            configurations,
            configurations_set,
            n_configurations,
            random_state,
        )

    # Initialize the Sobol sequence generator
    sobol_engine = qmc.Sobol(d=n_dimensions, scramble=True, seed=random_state)

    # Generate batches efficiently
    batch_size = min(n_additional * 2, 10000)
    max_attempts = min(n_additional * 5, 50000)
    attempts = 0

    while len(configurations) < n_configurations and attempts < max_attempts:
        current_batch_size = min(batch_size, max_attempts - attempts)

        # Generate Sobol samples in [0, 1) for this batch
        sobol_samples = sobol_engine.random(current_batch_size)

        # Process samples in batch
        batch_configs = [{} for _ in range(current_batch_size)]

        # Process numeric parameters using Sobol sequence
        for dim, (_, name, param_range) in enumerate(numeric_params):
            if isinstance(param_range, IntRange):
                # Map from [0, 1) to integer range
                # Vectorized calculation
                values = np.floor(
                    sobol_samples[:, dim]
                    * (param_range.max_value - param_range.min_value + 1e-10)
                    + param_range.min_value
                ).astype(int)
                # Ensure values are within range due to floating point issues
                values = np.clip(values, param_range.min_value, param_range.max_value)

                for i, value in enumerate(values):
                    batch_configs[i][name] = int(value)

            elif isinstance(param_range, FloatRange):
                # Map from [0, 1) to float range
                if param_range.log_scale:
                    log_min = np.log(max(param_range.min_value, 1e-10))
                    log_max = np.log(param_range.max_value)
                    values = np.exp(
                        log_min + sobol_samples[:, dim] * (log_max - log_min)
                    )
                else:
                    values = param_range.min_value + sobol_samples[:, dim] * (
                        param_range.max_value - param_range.min_value
                    )

                for i, value in enumerate(values):
                    batch_configs[i][name] = float(value)

        # Handle categorical parameters with uniform sampling
        for _, name, param_range in categorical_params:
            choices = param_range.choices
            indices = np.random.randint(0, len(choices), size=current_batch_size)
            for i, idx in enumerate(indices):
                batch_configs[i][name] = choices[idx]

        # Add unique configurations from batch
        for config in batch_configs:
            config_tuple = tuple(
                sorted(
                    (k, str(v) if isinstance(v, (list, dict, set)) else v)
                    for k, v in config.items()
                )
            )

            if config_tuple not in configurations_set:
                configurations_set.add(config_tuple)
                configurations.append(config)

                if len(configurations) >= n_configurations:
                    break

        attempts += current_batch_size

    if len(configurations) < n_configurations:
        logger.warning(
            f"Could only generate {len(configurations)} unique configurations "
            f"out of {n_configurations} requested after {attempts} Sobol attempts."
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
        self._cached_transforms = {}  # Cache for transformed configurations
        self._max_cache_size = 10000  # Increased cache size for better performance
        self._np_cache = {}  # Store numpy arrays directly for faster lookups

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

        # Precompute column positions for faster lookup during transform
        self.param_positions = {}
        if configurations:
            self.param_positions = {
                param_name: i
                for i, param_name in enumerate(sorted(configurations[0].keys()))
            }

        # Precompute column ranges for each parameter
        self.col_ranges = {}
        col_idx = 0
        for param_name in (
            sorted(self.param_positions.keys()) if self.param_positions else []
        ):
            if param_name in self.categorical_mappings:
                n_categories = len(self.categorical_mappings[param_name])
                self.col_ranges[param_name] = (col_idx, col_idx + n_categories)
                col_idx += n_categories
            else:
                self.col_ranges[param_name] = (col_idx, col_idx + 1)
                col_idx += 1

        # Clear cache when mappings change
        self._cached_transforms = {}
        self._np_cache = {}

    def transform(self, configurations: List[Dict]) -> pd.DataFrame:
        """Transform configurations into a tabular format with proper encoding."""
        if not self.column_names:
            self.fit(configurations)

        # Fast path: if we only have one configuration, check cache first
        if len(configurations) == 1:
            config = configurations[0]
            config_hash = tuple(
                sorted(
                    (k, str(v) if isinstance(v, (list, dict, set)) else v)
                    for k, v in config.items()
                )
            )

            if config_hash in self._np_cache:
                # Return directly from numpy cache for maximum speed
                return pd.DataFrame(
                    [self._np_cache[config_hash]], columns=self.column_names
                )

        # Regular transform path
        n_samples = len(configurations)
        n_features = len(self.column_names)
        X = np.zeros((n_samples, n_features))

        # Fill in the feature matrix
        for i, config in enumerate(configurations):
            config_hash = None
            if (
                len(configurations) > 50
            ):  # Only cache individual configs for large batches
                config_hash = tuple(
                    sorted(
                        (k, str(v) if isinstance(v, (list, dict, set)) else v)
                        for k, v in config.items()
                    )
                )
                if config_hash in self._np_cache:
                    X[i] = self._np_cache[config_hash]
                    continue

            # Process this configuration
            for param_name, value in config.items():
                if param_name in self.categorical_mappings:
                    # Handle categorical parameter with one-hot encoding
                    if value in self.categorical_mappings[param_name]:
                        one_hot_idx = self.categorical_mappings[param_name][value]
                        X[i, one_hot_idx] = 1
                else:
                    # Handle numeric parameter - use precomputed position
                    col_start, _ = self.col_ranges[param_name]
                    X[i, col_start] = value

            # Cache this configuration if not already in cache
            if config_hash and config_hash not in self._np_cache:
                # Store in cache but limit size
                if len(self._np_cache) >= self._max_cache_size:
                    # Simple LRU-like behavior: clear 20% of the cache
                    keys_to_remove = list(self._np_cache.keys())[
                        : int(self._max_cache_size * 0.2)
                    ]
                    for key in keys_to_remove:
                        self._np_cache.pop(key)

                self._np_cache[config_hash] = X[i].copy()

        result = pd.DataFrame(X, columns=self.column_names)
        return result
