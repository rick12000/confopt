from typing import Dict, List, Optional, Literal
import math
import logging
import random
import numpy as np
from scipy.stats import qmc
from confopt.wrapping import (
    IntRange,
    FloatRange,
    CategoricalRange,
    ParameterRange,
)
from confopt.utils.configurations.utils import create_config_hash

logger = logging.getLogger(__name__)


def get_tuning_configurations(
    parameter_grid: Dict[str, ParameterRange],
    n_configurations: int,
    random_state: Optional[int] = None,
    sampling_method: Literal["uniform", "sobol"] = "uniform",
) -> List[Dict]:
    """
    Generate a list of unique parameter configurations for hyperparameter tuning.

    This function delegates to either uniform or Sobol sampling based on the selected method.
    Uniform sampling draws random values for each parameter independently, while Sobol sampling
    generates low-discrepancy samples for numeric parameters and randomly assigns categorical values.
    Ensures uniqueness of configurations by hashing.

    Args:
        parameter_grid: Dictionary mapping parameter names to their range objects.
        n_configurations: Number of unique configurations to generate.
        random_state: Seed for reproducibility.
        sampling_method: Sampling strategy, either 'uniform' or 'sobol'.

    Returns:
        List of unique parameter configurations as dictionaries.
    """
    if sampling_method == "sobol":
        samples = _sobol_sampling(
            parameter_grid=parameter_grid,
            n_configurations=n_configurations,
            random_state=random_state,
        )
    elif sampling_method == "uniform":
        samples = _uniform_sampling(
            parameter_grid=parameter_grid,
            n_configurations=n_configurations,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Invalid sampling method: {sampling_method}. Must be 'uniform' or 'sobol'."
        )

    return samples


def _uniform_sampling(
    parameter_grid: Dict[str, ParameterRange],
    n_configurations: int,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """
    Generate unique parameter configurations using uniform random sampling.

    For each configuration, samples each parameter independently: integers and floats are drawn
    uniformly from their respective ranges (log-scale supported for both), and categorical
    parameters are chosen randomly from their choices. Ensures uniqueness by hashing each
    configuration. Sampling stops when the requested number of unique configurations is reached
    or a maximum attempt threshold is exceeded.

    Args:
        parameter_grid: Dictionary mapping parameter names to their range objects.
        n_configurations: Number of unique configurations to generate.
        random_state: Seed for reproducibility.

    Returns:
        List of unique parameter configurations as dictionaries.
    """
    configurations: List[Dict] = []
    configurations_set = set()
    if random_state is not None:
        random.seed(a=random_state)
        np.random.seed(seed=random_state)

    param_names = sorted(parameter_grid.keys())
    max_attempts = min(n_configurations * 3, 50000)
    attempts = 0
    while len(configurations) < n_configurations and attempts < max_attempts:
        config = {}
        for name in param_names:
            param_range = parameter_grid[name]
            if isinstance(param_range, IntRange):
                if param_range.log_scale:
                    lmin = np.log(max(param_range.min_value, 1))
                    lmax = np.log(param_range.max_value)
                    config[name] = int(np.round(np.exp(random.uniform(lmin, lmax))))
                    # Ensure the value is within bounds
                    config[name] = max(
                        param_range.min_value, min(config[name], param_range.max_value)
                    )
                else:
                    config[name] = random.randint(
                        param_range.min_value, param_range.max_value
                    )
            elif isinstance(param_range, FloatRange):
                if param_range.log_scale:
                    lmin = np.log(max(param_range.min_value, 1e-10))
                    lmax = np.log(param_range.max_value)
                    config[name] = float(np.exp(random.uniform(lmin, lmax)))
                else:
                    config[name] = random.uniform(
                        param_range.min_value, param_range.max_value
                    )
            elif isinstance(param_range, CategoricalRange):
                value = random.choice(param_range.choices)
                # Ensure bools don't get auto type cast to numpy.bool_ or int:
                # Check if ALL choices are actually boolean types, not just equal to True/False
                if all(isinstance(choice, bool) for choice in param_range.choices):
                    value = bool(value)
                config[name] = value
        config_hash = create_config_hash(config)
        if config_hash not in configurations_set:
            configurations_set.add(config_hash)
            configurations.append(config)
        attempts += 1

    if len(configurations) < n_configurations:
        logger.warning(
            f"Could only generate {len(configurations)} unique configurations "
        )
    return configurations


def _sobol_sampling(
    parameter_grid: Dict[str, ParameterRange],
    n_configurations: int,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """
    Generate unique parameter configurations using Sobol sequence sampling.

    Applies a low-discrepancy Sobol sequence to sample numeric parameters (int and float),
    mapping each dimension to a parameter. Categorical parameters are assigned randomly.
    Ensures uniqueness by hashing each configuration. At least one numeric parameter is required.
    Sampling stops when the requested number of unique configurations is reached.

    Args:
        parameter_grid: Dictionary mapping parameter names to their range objects.
        n_configurations: Number of unique configurations to generate.
        random_state: Seed for reproducibility.

    Returns:
        List of unique parameter configurations as dictionaries.
    """
    configurations: List[Dict] = []
    configurations_set = set()
    # Seed random generators for reproducible categorical assignments
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    param_names = sorted(parameter_grid.keys())
    param_ranges = [parameter_grid[name] for name in param_names]
    # Separate numeric and categorical parameters for Sobol and random sampling
    numeric_params = [
        (i, name, pr)
        for i, (name, pr) in enumerate(zip(param_names, param_ranges))
        if isinstance(pr, (IntRange, FloatRange))
    ]
    categorical_params = [
        (i, name, pr)
        for i, (name, pr) in enumerate(zip(param_names, param_ranges))
        if isinstance(pr, CategoricalRange)
    ]

    if not numeric_params:
        raise ValueError("Sobol sampling requires at least one numeric parameter.")

    # Generate Sobol samples for numeric parameters.
    # SciPy's Sobol implementation expects a power-of-two sample size for balance.
    # Use `random_base2(m)` to generate 2**m samples (power of two) and then
    # slice to the requested `n_configurations` to avoid the UserWarning.
    if n_configurations <= 0:
        raise ValueError(
            "n_configurations must be a positive integer for Sobol sampling"
        )
    sobol_engine = qmc.Sobol(d=len(numeric_params), scramble=False, seed=random_state)
    # Compute the smallest m such that 2**m >= n_configurations
    m = math.ceil(math.log2(n_configurations))
    samples_all = sobol_engine.random_base2(m)
    samples = samples_all[:n_configurations]
    for row in samples:
        config = {}
        # Map Sobol sample to each numeric parameter
        for dim, (_, name, pr) in enumerate(numeric_params):
            if isinstance(pr, IntRange):
                if pr.log_scale:
                    lmin = np.log(max(pr.min_value, 1))
                    lmax = np.log(pr.max_value)
                    value = int(np.round(np.exp(lmin + row[dim] * (lmax - lmin))))
                    config[name] = max(pr.min_value, min(value, pr.max_value))
                else:
                    # Use round instead of floor for more balanced integer sampling
                    value = int(
                        np.round(
                            row[dim] * (pr.max_value - pr.min_value) + pr.min_value
                        )
                    )
                    config[name] = max(pr.min_value, min(value, pr.max_value))
            else:
                if pr.log_scale:
                    lmin = np.log(max(pr.min_value, 1e-10))
                    lmax = np.log(pr.max_value)
                    config[name] = float(np.exp(lmin + row[dim] * (lmax - lmin)))
                else:
                    config[name] = float(
                        pr.min_value + row[dim] * (pr.max_value - pr.min_value)
                    )
        # Assign categorical parameters randomly
        for _, name, pr in categorical_params:
            value = random.choice(pr.choices)
            # Ensure bools are Python bool, not numpy.bool_ or int
            # Check if ALL choices are actually boolean types, not just equal to True/False
            if all(isinstance(choice, bool) for choice in pr.choices):
                value = bool(value)
            config[name] = value
        config_hash = create_config_hash(config)
        # Ensure uniqueness of each configuration
        if config_hash not in configurations_set:
            configurations_set.add(config_hash)
            configurations.append(config)
        if len(configurations) >= n_configurations:
            break
    if len(configurations) < n_configurations:
        logger.warning(
            f"Could only generate {len(configurations)} unique configurations "
        )
    return configurations
