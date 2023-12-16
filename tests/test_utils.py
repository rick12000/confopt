from acho.utils import (
    get_tuning_configurations,
    get_perceptron_layers,
)

DEFAULT_SEED = 1234


def test_get_perceptron_layers():
    dummy_seed = DEFAULT_SEED
    dummy_n_layers_grid = [2, 3, 4]
    dummy_layer_size_grid = [16, 32, 64, 128]

    layer_list = get_perceptron_layers(
        n_layers_grid=dummy_n_layers_grid,
        layer_size_grid=dummy_layer_size_grid,
        random_seed=dummy_seed,
    )

    for layer in layer_list:
        assert isinstance(layer, tuple)
        assert min(dummy_n_layers_grid) <= len(layer) <= max(dummy_n_layers_grid)
        for layer_size in layer:
            assert (
                min(dummy_layer_size_grid) <= layer_size <= max(dummy_layer_size_grid)
            )


def test_get_perceptron_layers__reproducibility():
    dummy_seed = DEFAULT_SEED
    dummy_n_layers_grid = [2, 3, 4]
    dummy_layer_size_grid = [16, 32, 64, 128]

    layer_list_first_call = get_perceptron_layers(
        n_layers_grid=dummy_n_layers_grid,
        layer_size_grid=dummy_layer_size_grid,
        random_seed=dummy_seed,
    )
    layer_list_second_call = get_perceptron_layers(
        n_layers_grid=dummy_n_layers_grid,
        layer_size_grid=dummy_layer_size_grid,
        random_seed=dummy_seed,
    )
    for layer_first_call, layer_second_call in zip(
        layer_list_first_call, layer_list_second_call
    ):
        assert layer_first_call == layer_second_call


def test_get_tuning_configurations(dummy_parameter_grid):
    dummy_seed = DEFAULT_SEED
    dummy_n_configurations = 10

    tuning_configurations = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=dummy_seed,
    )
    assert len(tuning_configurations) < dummy_n_configurations
    for configuration in tuning_configurations:
        for k, v in configuration.items():
            # Check configuration only has parameter names from parameter grid prompt:
            assert k in dummy_parameter_grid.keys()
            # Check values in configuration come from range in parameter grid prompt:
            assert v in dummy_parameter_grid[k]


def test_get_tuning_configurations__reproducibility(dummy_parameter_grid):
    dummy_seed = DEFAULT_SEED
    dummy_n_configurations = 10

    tuning_configurations_first_call = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=dummy_seed,
    )
    tuning_configurations_second_call = get_tuning_configurations(
        parameter_grid=dummy_parameter_grid,
        n_configurations=dummy_n_configurations,
        random_state=dummy_seed,
    )
    for configuration_first_call, configuration_second_call in zip(
        tuning_configurations_first_call, tuning_configurations_second_call
    ):
        assert configuration_first_call == configuration_second_call
