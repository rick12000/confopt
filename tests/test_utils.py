from confopt.utils import (
    get_tuning_configurations,
    get_perceptron_layers,
    tabularize_configurations,
)

DEFAULT_SEED = 1234


def test_get_perceptron_layers():
    dummy_n_layers_grid = [2, 3, 4]
    dummy_layer_size_grid = [16, 32, 64, 128]

    layer_list = get_perceptron_layers(
        n_layers_grid=dummy_n_layers_grid,
        layer_size_grid=dummy_layer_size_grid,
        random_seed=DEFAULT_SEED,
    )

    for layer in layer_list:
        assert isinstance(layer, tuple)
        assert min(dummy_n_layers_grid) <= len(layer) <= max(dummy_n_layers_grid)
        for layer_size in layer:
            assert (
                min(dummy_layer_size_grid) <= layer_size <= max(dummy_layer_size_grid)
            )


def test_get_perceptron_layers__reproducibility():
    dummy_n_layers_grid = [2, 3, 4]
    dummy_layer_size_grid = [16, 32, 64, 128]

    layer_list_first_call = get_perceptron_layers(
        n_layers_grid=dummy_n_layers_grid,
        layer_size_grid=dummy_layer_size_grid,
        random_seed=DEFAULT_SEED,
    )
    layer_list_second_call = get_perceptron_layers(
        n_layers_grid=dummy_n_layers_grid,
        layer_size_grid=dummy_layer_size_grid,
        random_seed=DEFAULT_SEED,
    )
    for layer_first_call, layer_second_call in zip(
        layer_list_first_call, layer_list_second_call
    ):
        assert layer_first_call == layer_second_call


def test_get_tuning_configurations(dummy_parameter_grid):
    dummy_n_configurations = 10000

    tuning_configurations = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=DEFAULT_SEED,
    )
    assert len(tuning_configurations) < dummy_n_configurations
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
