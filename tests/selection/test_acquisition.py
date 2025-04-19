import pytest
import numpy as np
import random
from unittest.mock import patch
from confopt.wrapping import ConformalBounds
from confopt.selection.acquisition import (
    calculate_ucb_predictions,
    calculate_thompson_predictions,
    calculate_expected_improvement,
    calculate_information_gain,
    flatten_conformal_bounds,
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,
)
from confopt.selection.sampling import (
    PessimisticLowerBoundSampler,
    LowerBoundSampler,
    ThompsonSampler,
    ExpectedImprovementSampler,
    InformationGainSampler,
)
from confopt.selection.conformalization import QuantileConformalEstimator
from conftest import (
    POINT_ESTIMATOR_ARCHITECTURES,
    QUANTILE_ESTIMATOR_ARCHITECTURES,
    SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
)


def test_calculate_ucb_predictions():
    lower_bound = np.array([0.5, 0.7, 0.3, 0.9])
    interval_width = np.array([0.2, 0.1, 0.3, 0.05])
    beta = 0.5

    result = calculate_ucb_predictions(
        lower_bound=lower_bound, interval_width=interval_width, beta=beta
    )
    expected = np.array([0.4, 0.65, 0.15, 0.875])

    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "enable_optimistic, point_predictions",
    [(False, None), (True, np.array([0.05, 0.35, 0.75, 0.25, 0.95]))],
)
def test_calculate_thompson_predictions(
    conformal_bounds, enable_optimistic, point_predictions
):
    fixed_indices = np.array([0, 3, 5, 1, 4])

    with patch.object(np.random, "randint", return_value=fixed_indices):
        result = calculate_thompson_predictions(
            predictions_per_interval=conformal_bounds,
            enable_optimistic_sampling=enable_optimistic,
            point_predictions=point_predictions,
        )

    flattened_bounds = flatten_conformal_bounds(conformal_bounds)
    expected_sampled_bounds = np.array(
        [flattened_bounds[i, idx] for i, idx in enumerate(fixed_indices)]
    )

    if enable_optimistic:
        expected = np.minimum(expected_sampled_bounds, point_predictions)
    else:
        expected = expected_sampled_bounds

    np.testing.assert_array_almost_equal(result, expected)


def test_thompson_predictions_randomized(conformal_bounds):
    np.random.seed(42)

    predictions = calculate_thompson_predictions(conformal_bounds)
    assert len(predictions) == 5

    point_predictions = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    predictions = calculate_thompson_predictions(
        conformal_bounds,
        enable_optimistic_sampling=True,
        point_predictions=point_predictions,
    )
    assert len(predictions) == 5
    assert np.all(predictions <= point_predictions) or np.all(predictions < np.inf)


@pytest.fixture
def simple_conformal_bounds():
    """Create simple conformal bounds for testing."""
    lower_bounds1 = np.array([0.1, 0.3, 0.5])
    upper_bounds1 = np.array([0.4, 0.6, 0.8])

    lower_bounds2 = np.array([0.2, 0.4, 0.6])
    upper_bounds2 = np.array([0.5, 0.7, 0.9])

    return [
        ConformalBounds(lower_bounds=lower_bounds1, upper_bounds=upper_bounds1),
        ConformalBounds(lower_bounds=lower_bounds2, upper_bounds=upper_bounds2),
    ]


def test_calculate_expected_improvement_detailed(simple_conformal_bounds):
    with patch.object(
        np.random,
        "randint",
        side_effect=[np.array([[0], [1], [2]]), np.array([[0], [1], [2]])],
    ):
        result = calculate_expected_improvement(
            predictions_per_interval=simple_conformal_bounds,
            best_historical_y=0.4,
            num_samples=1,
        )

    expected = np.array([0.0, -0.2, -0.2])
    np.testing.assert_array_almost_equal(result, expected)

    with patch.object(
        np.random,
        "randint",
        side_effect=[np.array([[0], [1], [2]]), np.array([[0], [1], [2]])],
    ):
        result = calculate_expected_improvement(
            predictions_per_interval=simple_conformal_bounds,
            best_historical_y=0.6,
            num_samples=1,
        )

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_expected_improvement_randomized(conformal_bounds):
    np.random.seed(42)

    ei = calculate_expected_improvement(
        predictions_per_interval=conformal_bounds,
        best_historical_y=0.5,
        num_samples=10,
    )

    assert len(ei) == 5
    assert np.all(ei <= 0)


@pytest.mark.parametrize("sampling_strategy", ["uniform", "thompson"])
def test_information_gain_with_toy_dataset(big_toy_dataset, sampling_strategy):
    X, y = big_toy_dataset
    n_X_candidates = 50

    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]

    np.random.seed(42)
    random.seed(42)

    quantile_estimator = QuantileConformalEstimator(
        quantile_estimator_architecture="ql",
        alphas=[0.1, 0.5, 0.9],
        n_pre_conformal_trials=5,
    )

    X_val, y_val = X[train_size:], y[train_size:]

    quantile_estimator.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )

    # Only predict on the same number of points as X_train (to avoid shape mismatch)
    real_predictions = quantile_estimator.predict_intervals(X_train)

    ig = calculate_information_gain(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,  # Pass validation data
        y_val=y_val,  # Pass validation data
        X_space=X_train,  # Use X_train for both to match shapes
        conformal_estimator=quantile_estimator,
        predictions_per_interval=real_predictions,
        n_paths=10,
        n_y_candidates_per_x=5,
        n_X_candidates=n_X_candidates,
        sampling_strategy=sampling_strategy,
    )

    assert isinstance(ig, np.ndarray)
    assert len(ig) == len(X_train)
    assert np.all(ig <= 0)
    assert np.sum(np.where(ig < 0)) <= n_X_candidates
    assert np.sum(ig < 0) < 0


def test_flatten_conformal_bounds_detailed(simple_conformal_bounds):
    flattened = flatten_conformal_bounds(simple_conformal_bounds)

    assert flattened.shape == (3, 4)

    expected = np.array(
        [
            [0.1, 0.4, 0.2, 0.5],
            [0.3, 0.6, 0.4, 0.7],
            [0.5, 0.8, 0.6, 0.9],
        ]
    )

    np.testing.assert_array_equal(flattened, expected)


def test_flatten_conformal_bounds(conformal_bounds):
    flattened = flatten_conformal_bounds(conformal_bounds)

    assert flattened.shape == (5, len(conformal_bounds) * 2)

    for i, interval in enumerate(conformal_bounds):
        assert np.array_equal(flattened[:, i * 2], interval.lower_bounds.flatten())
        assert np.array_equal(flattened[:, i * 2 + 1], interval.upper_bounds.flatten())


@pytest.fixture
def larger_toy_dataset():
    """Create a larger toy dataset for searcher tests"""
    X = np.random.rand(10, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    return X, y


@pytest.mark.parametrize(
    "sampler_class,sampler_kwargs",
    [
        (PessimisticLowerBoundSampler, {"interval_width": 0.8}),
        (LowerBoundSampler, {"interval_width": 0.8}),
        (ThompsonSampler, {"n_quantiles": 4}),
        (ExpectedImprovementSampler, {"n_quantiles": 4}),
        (InformationGainSampler, {"n_quantiles": 4}),
    ],
)
@pytest.mark.parametrize("point_arch", POINT_ESTIMATOR_ARCHITECTURES[:1])
@pytest.mark.parametrize("variance_arch", POINT_ESTIMATOR_ARCHITECTURES[:1])
def test_locally_weighted_conformal_searcher(
    sampler_class, sampler_kwargs, point_arch, variance_arch, larger_toy_dataset
):
    X, y = larger_toy_dataset
    X_train, y_train = X[:7], y[:7]
    X_val, y_val = X[7:], y[7:]

    sampler = sampler_class(**sampler_kwargs)
    searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture=point_arch,
        variance_estimator_architecture=variance_arch,
        sampler=sampler,
    )

    searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )

    predictions = searcher.predict(X_val)
    assert len(predictions) == len(X_val)

    X_update = X_val[0].reshape(1, -1)
    y_update = y_val[0]
    initial_X_train_len = len(searcher.X_train)
    initial_y_train_len = len(searcher.y_train)

    searcher.update(X_update, y_update)

    assert len(searcher.X_train) == initial_X_train_len + 1
    assert len(searcher.y_train) == initial_y_train_len + 1
    assert np.array_equal(searcher.X_train[-1], X_update.flatten())
    assert searcher.y_train[-1] == y_update


@pytest.mark.parametrize(
    "sampler_class,sampler_kwargs",
    [
        (PessimisticLowerBoundSampler, {"interval_width": 0.8}),
        (LowerBoundSampler, {"interval_width": 0.8}),
        (ThompsonSampler, {"n_quantiles": 4}),
        (ExpectedImprovementSampler, {"n_quantiles": 4}),
        (InformationGainSampler, {"n_quantiles": 4}),
    ],
)
@pytest.mark.parametrize(
    "quantile_arch",
    [
        QUANTILE_ESTIMATOR_ARCHITECTURES[0],
        SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES[0],
    ],
)
def test_quantile_conformal_searcher(
    sampler_class, sampler_kwargs, quantile_arch, larger_toy_dataset
):
    X, y = larger_toy_dataset
    X_train, y_train = X[:7], y[:7]
    X_val, y_val = X[7:], y[7:]

    sampler = sampler_class(**sampler_kwargs)
    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture=quantile_arch,
        sampler=sampler,
        n_pre_conformal_trials=5,
    )

    searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )

    predictions = searcher.predict(X_val)
    assert len(predictions) == len(X_val)

    X_update = X_val[0].reshape(1, -1)
    y_update = y_val[0]
    initial_X_train_len = len(searcher.X_train)
    initial_y_train_len = len(searcher.y_train)

    searcher.update(X_update, y_update)

    assert len(searcher.X_train) == initial_X_train_len + 1
    assert len(searcher.y_train) == initial_y_train_len + 1
    assert np.array_equal(searcher.X_train[-1], X_update.flatten())
    assert searcher.y_train[-1] == y_update
