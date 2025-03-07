import pytest

from confopt.utils import (
    get_tuning_configurations,
    tabularize_configurations,
)

DEFAULT_SEED = 1234


@pytest.mark.parametrize("dummy_n_configurations", [100, 1000, 10000])
def test_get_tuning_configurations(dummy_parameter_grid, dummy_n_configurations):
    tuning_configurations = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=DEFAULT_SEED,
    )
    assert len(tuning_configurations) == dummy_n_configurations
    configuration_lens = []
    for configuration in tuning_configurations:
        for k, v in configuration.items():
            # Check configuration only has parameter names from parameter grid prompt:
            assert k in dummy_parameter_grid.keys()
            # Check values in configuration come from range in parameter grid prompt:
            assert v in dummy_parameter_grid[k]

            configuration_lens.append(len(configuration))

    assert max(configuration_lens) == min(configuration_lens)


def test_get_tuning_configurations__reproducibility(dummy_parameter_grid):
    dummy_n_configurations = 10

    tuning_configurations_first_call = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=DEFAULT_SEED,
    )
    tuning_configurations_second_call = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=DEFAULT_SEED,
    )
    for configuration_first_call, configuration_second_call in zip(
        tuning_configurations_first_call, tuning_configurations_second_call
    ):
        assert configuration_first_call == configuration_second_call


def test_tabularize_configurations(dummy_parameter_grid):
    dummy_n_configurations = 10
    searchable_configurations = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=DEFAULT_SEED,
    )
    dummy_n_configurations = 10
    searched_configurations = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=DEFAULT_SEED + 1,
    )
    searched_configurations = [
        configuration
        for configuration in searched_configurations
        if configuration not in searchable_configurations
    ]

    (
        tabularized_searchable_configurations,
        tabularized_searched_configurations,
    ) = tabularize_configurations(
        searchable_configurations=searchable_configurations,
        searched_configurations=searched_configurations,
    )

    assert len(tabularized_searchable_configurations) + len(
        tabularized_searched_configurations
    ) == len(searchable_configurations) + len(searched_configurations)
