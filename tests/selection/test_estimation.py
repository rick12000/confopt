import pytest

from confopt.selection.estimation import (
    initialize_estimator,
    average_scores_across_folds,
)

from confopt.selection.estimator_configuration import ESTIMATOR_REGISTRY


@pytest.mark.parametrize("estimator_architecture", list(ESTIMATOR_REGISTRY.keys()))
def test_initialize_estimator_returns_expected_type(estimator_architecture):
    """Test that initialize_estimator returns the correct estimator type."""
    estimator = initialize_estimator(estimator_architecture, random_state=42)
    expected_class = ESTIMATOR_REGISTRY[estimator_architecture].estimator_class
    assert isinstance(estimator, expected_class)


@pytest.mark.parametrize("random_state", [None, 42, 123])
def test_initialize_estimator_with_random_state(random_state):
    """Test that random_state is properly set when supported by estimator."""
    estimator = initialize_estimator(
        estimator_architecture="gbm",
        initialization_params={"random_state": 42} if random_state else {},
        random_state=random_state,
    )
    assert estimator.random_state == random_state


@pytest.mark.parametrize("split_type", ["k_fold", "ordinal_split"])
@pytest.mark.parametrize("n_searches", [1, 3, 10])
def test_point_tuner_returns_valid_configuration(
    point_tuner, estimation_test_data, split_type, n_searches
):
    """Test that PointTuner returns a valid configuration for different search counts."""
    X_train, X_val, y_train, y_val = estimation_test_data

    # Use an estimator we know exists
    estimator_architecture = "gbm"
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

    best_config = point_tuner.tune(
        X_train,
        y_train,
        estimator_architecture,
        n_searches=n_searches,
        train_split=0.8,
        split_type=split_type,
    )

    # Configuration should be a dictionary
    assert isinstance(best_config, dict)

    # All parameter keys should be valid for this estimator
    valid_params = set(estimator_config.estimator_parameter_space.keys())
    assert set(best_config.keys()).issubset(valid_params)


@pytest.mark.parametrize("split_type", ["k_fold", "ordinal_split"])
def test_quantile_tuner_returns_valid_configuration(
    quantile_tuner_with_quantiles, estimation_test_data, split_type
):
    """Test that QuantileTuner returns valid configuration for quantile estimators."""
    tuner, quantiles = quantile_tuner_with_quantiles
    X_train, X_val, y_train, y_val = estimation_test_data

    # Find a quantile estimator
    quantile_architectures = [
        arch
        for arch, config in ESTIMATOR_REGISTRY.items()
        if config.is_quantile_estimator()
    ]
    estimator_architecture = quantile_architectures[0]
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]

    best_config = tuner.tune(
        X_train,
        y_train,
        estimator_architecture,
        n_searches=3,
        train_split=0.8,
        split_type=split_type,
    )

    # Configuration should be a dictionary
    assert isinstance(best_config, dict)

    # All parameter keys should be valid for this estimator
    valid_params = set(estimator_config.estimator_parameter_space.keys())
    assert set(best_config.keys()).issubset(valid_params)


def test_tuning_with_forced_configurations_prioritizes_them(
    point_tuner, estimation_test_data
):
    """Test that forced configurations are prioritized in tuning process."""
    X_train, X_val, y_train, y_val = estimation_test_data

    estimator_architecture = "gbm"
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]
    forced_config = estimator_config.default_params

    best_config = point_tuner.tune(
        X_train,
        y_train,
        estimator_architecture,
        n_searches=1,  # Only one search, should return forced config
        train_split=0.8,
        split_type="ordinal_split",
        forced_param_configurations=[forced_config],
    )

    assert best_config == forced_config


def test_correct_averaging_and_ordering():
    """Test that order of unique configurations is preserved during averaging."""
    configs = [
        {"param": "first"},
        {"param": "second"},
        {"param": "first"},  # duplicate
        {"param": "third"},
    ]
    scores = [1.0, 2.0, 3.0, 4.0]

    unique_configs, unique_scores = average_scores_across_folds(configs, scores)

    # First unique should be "first", second should be "second", third should be "third"
    assert unique_configs[0]["param"] == "first"
    assert unique_configs[1]["param"] == "second"
    assert unique_configs[2]["param"] == "third"

    # Check scores are averaged correctly
    assert unique_scores[0] == 2.0  # (1.0 + 3.0) / 2
    assert unique_scores[1] == 2.0
    assert unique_scores[2] == 4.0
