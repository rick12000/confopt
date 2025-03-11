import numpy as np

from confopt.utils import (
    get_tuning_configurations,
    ConfigurationEncoder,
)
from confopt.ranges import IntRange, FloatRange, CategoricalRange

DEFAULT_SEED = 1234


def test_get_tuning_configurations(dummy_parameter_grid):
    """Test that _get_tuning_configurations creates valid configurations"""

    n_configurations = 50
    configurations = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=n_configurations,
        random_state=DEFAULT_SEED,
    )

    # Check correct number of configurations generated
    assert len(configurations) == n_configurations

    # Check all configurations have the expected parameters
    for config in configurations:
        assert set(config.keys()) == set(dummy_parameter_grid.keys())

        # Check each parameter value is within its defined range
        for param_name, param_value in config.items():
            param_range = dummy_parameter_grid[param_name]
            assert param_range.min_value <= param_value <= param_range.max_value

            # For log scale params, check distribution is appropriate
            if hasattr(param_range, "log_scale") and param_range.log_scale:
                # Values should be distributed across orders of magnitude
                assert param_value > 0  # Log-scaled values must be positive


def test_get_tuning_configurations__reproducibility(dummy_parameter_grid):
    """Test reproducibility of configuration generation"""
    dummy_n_configurations = 10

    # First call with seed and explicitly setting warm_start_configs=None
    np.random.seed(DEFAULT_SEED)
    tuning_configs_first_call = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=DEFAULT_SEED,
        warm_start_configs=None,
    )

    # Second call with same seed
    np.random.seed(DEFAULT_SEED)
    tuning_configs_second_call = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=DEFAULT_SEED,
        warm_start_configs=None,
    )

    # Check that configurations are identical
    for idx, (config1, config2) in enumerate(
        zip(tuning_configs_first_call, tuning_configs_second_call)
    ):
        for param in config1:
            assert config1[param] == config2[param]


def test_get_tuning_configurations_with_warm_start():
    """Test that get_tuning_configurations properly includes warm start configurations"""
    # Define a simple parameter grid
    parameter_grid = {
        "int_param": IntRange(min_value=1, max_value=10),
        "float_param": FloatRange(min_value=0.1, max_value=1.0),
        "cat_param": CategoricalRange(choices=["option1", "option2", "option3"]),
    }

    # Create warm start configurations
    warm_start_configs = [
        {"int_param": 5, "float_param": 0.5, "cat_param": "option1"},
        {"int_param": 8, "float_param": 0.8, "cat_param": "option3"},
    ]

    n_configurations = 10
    configurations = get_tuning_configurations(
        parameter_grid=parameter_grid,
        n_configurations=n_configurations,
        random_state=DEFAULT_SEED,
        warm_start_configs=warm_start_configs,
    )

    # Check correct number of configurations generated
    assert len(configurations) == n_configurations

    # Verify warm start configs are included in the result
    for warm_start in warm_start_configs:
        assert any(
            all(config[k] == warm_start[k] for k in warm_start)
            for config in configurations
        )

    # All configurations should meet parameter constraints
    for config in configurations:
        # Check all keys exist
        assert set(config.keys()) == set(parameter_grid.keys())

        # Check values are within ranges
        assert 1 <= config["int_param"] <= 10
        assert 0.1 <= config["float_param"] <= 1.0
        assert config["cat_param"] in ["option1", "option2", "option3"]


def test_configuration_encoder():
    """Test that ConfigurationEncoder properly encodes configurations"""
    # Create configurations with mixed parameter types
    configs = [
        {"numeric1": 1.0, "numeric2": 5, "cat1": "a", "cat2": True},
        {"numeric1": 2.0, "numeric2": 10, "cat1": "b", "cat2": False},
        {"numeric1": 3.0, "numeric2": 15, "cat1": "a", "cat2": True},
    ]

    # Test initialization and fitting
    encoder = ConfigurationEncoder()
    encoder.fit(configs)

    # Verify categorical mappings are created correctly
    assert "cat1" in encoder.categorical_mappings

    # Test transformation
    df = encoder.transform(configs)

    # Check shape - should have columns for numeric1, numeric2, cat1_a, cat1_b
    # Boolean values may be treated as numeric (0/1) rather than categorical
    assert df.shape[0] == 3  # 3 rows

    # Verify numeric columns are preserved
    assert "numeric1" in df.columns
    assert "numeric2" in df.columns

    # Check one-hot encoding worked correctly for string categorical values
    cat1_cols = [col for col in df.columns if col.startswith("cat1_")]
    assert len(cat1_cols) == 2  # "a" and "b"

    cat1_a_col = next(col for col in cat1_cols if "a" in col)
    cat1_b_col = next(col for col in cat1_cols if "b" in col)

    # First row has cat1="a", so a=1, b=0
    assert df.loc[0, cat1_a_col] == 1
    assert df.loc[0, cat1_b_col] == 0

    # Second row has cat1="b", so a=0, b=1
    assert df.loc[1, cat1_a_col] == 0
    assert df.loc[1, cat1_b_col] == 1

    # Check how boolean values are handled - could be either numeric or categorical
    if "cat2" in df.columns:
        # Treated as numeric
        assert df.loc[0, "cat2"] == 1  # True
        assert df.loc[1, "cat2"] == 0  # False
    else:
        # Treated as categorical
        cat2_cols = [col for col in df.columns if col.startswith("cat2_")]
        assert len(cat2_cols) > 0
