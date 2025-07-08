import pytest


from confopt.utils.configurations.sampling import get_tuning_configurations

RANDOM_STATE = 1234


@pytest.mark.parametrize("method", ["uniform", "sobol"])
def test_reproducibility(dummy_parameter_grid, method):
    configs1 = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=10,
        random_state=RANDOM_STATE,
        sampling_method=method,
    )
    configs2 = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=10,
        random_state=RANDOM_STATE,
        sampling_method=method,
    )
    assert configs1 == configs2


@pytest.mark.parametrize("method", ["uniform", "sobol"])
def test_config_value_ranges(dummy_parameter_grid, method):
    n = 50
    configs = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=n,
        random_state=RANDOM_STATE,
        sampling_method=method,
    )
    assert len(configs) == n

    for config in configs:
        int_val = config["param_2"]
        assert isinstance(int_val, int)
        assert 1 <= int_val <= 100

        float_val = config["param_1"]
        assert isinstance(float_val, float)
        assert 0.01 <= float_val <= 100

        cat_val = config["param_3"]
        assert cat_val in dummy_parameter_grid["param_3"].choices


@pytest.mark.parametrize("method", ["uniform", "sobol"])
def test_sampling_uniqueness(dummy_parameter_grid, method):
    n = 100
    configs = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=n,
        random_state=123,
        sampling_method=method,
    )
    unique_configs = {frozenset(cfg.items()) for cfg in configs}
    assert len(unique_configs) == len(configs)
