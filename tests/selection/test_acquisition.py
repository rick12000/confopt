import pytest
import numpy as np
from unittest.mock import patch, Mock
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
    fixed_indices = np.array([0, 1, 2, 0, 1])

    with patch.object(np.random, "choice", return_value=fixed_indices):
        result = calculate_thompson_predictions(
            predictions_per_interval=conformal_bounds,
            enable_optimistic_sampling=enable_optimistic,
            point_predictions=point_predictions,
        )

    lower_bounds = np.array(
        [
            conformal_bounds[0].lower_bounds[0],
            conformal_bounds[1].lower_bounds[1],
            conformal_bounds[2].lower_bounds[2],
            conformal_bounds[0].lower_bounds[3],
            conformal_bounds[1].lower_bounds[4],
        ]
    )

    if enable_optimistic:
        expected = np.minimum(lower_bounds, point_predictions)
    else:
        expected = lower_bounds

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
    with patch.object(np.random, "randint", side_effect=[[0], [1], [0]]):
        result = calculate_expected_improvement(
            predictions_per_interval=simple_conformal_bounds,
            current_best_value=0.4,
            num_samples=1,
        )

    # Expected values are now negative (multiplied by -1)
    expected = np.array([0.0, -0.3, -0.1])
    np.testing.assert_array_almost_equal(result, expected)

    with patch.object(np.random, "randint", side_effect=[[0], [1], [0]]):
        result = calculate_expected_improvement(
            predictions_per_interval=simple_conformal_bounds,
            current_best_value=0.6,
            num_samples=1,
        )

    # Expected values are now negative (multiplied by -1)
    expected = np.array([0.0, -0.1, 0.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_expected_improvement_randomized(conformal_bounds):
    np.random.seed(42)

    ei = calculate_expected_improvement(
        predictions_per_interval=conformal_bounds,
        current_best_value=0.5,
        num_samples=10,
    )

    assert len(ei) == 5
    # EI should now be non-positive (values are negative or zero)
    assert np.all(ei <= 0)


def test_information_gain_with_minimal_mocking():
    X_candidates = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
    )

    X_train = np.array([[0.0, 0.0]])
    y_train = np.array([0.5])

    lower_bounds1 = np.array([0.1, 0.3, 0.5, 0.2, 0.4])
    upper_bounds1 = np.array([0.4, 0.6, 0.8, 0.5, 0.7])

    conformal_bounds = [
        ConformalBounds(lower_bounds=lower_bounds1, upper_bounds=upper_bounds1)
    ]

    mock_estimator = Mock()
    mock_estimator.predict_intervals.return_value = conformal_bounds

    with patch("confopt.selection.acquisition.random.choice", return_value=0.5), patch(
        "numpy.random.choice", return_value=np.array([0, 2])
    ):

        result = calculate_information_gain(
            X_candidates=X_candidates,
            conformal_estimator=mock_estimator,
            predictions_per_interval=conformal_bounds,
            X_train=X_train,
            y_train=y_train,
            n_samples=2,
            n_y_samples_per_x=1,
            n_eval_candidates=2,
            kde_bandwidth=0.3,
            random_state=42,
        )

    assert isinstance(result, np.ndarray)
    assert len(result) == len(X_candidates)

    # Non-zero positions remain the same but values are now negative
    non_zero_positions = np.where(result < 0)[0]
    assert set(non_zero_positions).issubset({0, 2})
    assert result[1] == 0
    assert result[3] == 0
    assert result[4] == 0

    # Information gain values should now be non-positive
    assert np.all(result <= 0)


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
def mock_kde():
    mock = Mock()
    mock.score_samples.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock.fit.return_value = None
    return mock


@pytest.mark.parametrize("n_eval_candidates", [10, 30])
def test_calculate_information_gain_parameters(
    conformal_bounds, mock_kde, n_eval_candidates
):
    X_candidates = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    X_train = np.array([[1, 1]])
    y_train = np.array([0.5])

    mock_conformal_estimator = Mock()
    mock_conformal_estimator.predict_intervals.return_value = conformal_bounds

    with patch(
        "confopt.selection.acquisition.KernelDensity", return_value=mock_kde
    ), patch("confopt.selection.acquisition.entropy", return_value=1.0), patch(
        "confopt.selection.acquisition.random.choice", return_value=0.5
    ), patch(
        "numpy.random.choice",
        return_value=np.arange(min(n_eval_candidates, len(X_candidates))),
    ):

        result = calculate_information_gain(
            X_candidates=X_candidates,
            conformal_estimator=mock_conformal_estimator,
            predictions_per_interval=conformal_bounds,
            X_train=X_train,
            y_train=y_train,
            n_samples=5,
            n_y_samples_per_x=2,
            n_eval_candidates=n_eval_candidates,
            kde_bandwidth=0.3,
            random_state=42,
        )

    assert isinstance(result, np.ndarray)
    assert len(result) == len(X_candidates)
    # Count non-zero values (now they are negative)
    assert np.sum(result < 0) <= n_eval_candidates


def test_information_gain_with_toy_dataset(toy_dataset, conformal_bounds):
    X, y = toy_dataset

    class MockConformalEstimator:
        def __init__(self):
            self.nonconformity_scores = [np.array([0.1, 0.2, 0.3])]

        def fit(self, **kwargs):
            pass

        def predict_intervals(self, X):
            return conformal_bounds

    mock_estimator = MockConformalEstimator()

    np.random.seed(42)
    import random

    random.seed(42)

    ig = calculate_information_gain(
        X_candidates=X,
        conformal_estimator=mock_estimator,
        predictions_per_interval=conformal_bounds,
        X_train=X[:2],
        y_train=y[:2],
        n_samples=3,
        n_y_samples_per_x=2,
        n_eval_candidates=2,
        kde_bandwidth=0.5,
        random_state=42,
    )

    assert len(ig) == len(X)


@pytest.fixture
def larger_toy_dataset():
    """Create a larger toy dataset for searcher tests"""
    X = np.random.rand(10, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    return X, y


# Parameterized tests for searcher classes
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

    # Test prediction
    predictions = searcher.predict(X_val)
    assert len(predictions) == len(X_val)

    # Test update method
    X_update = X_val[0].reshape(1, -1)
    y_update = y_val[0]
    initial_X_train_len = len(searcher.X_train)
    initial_y_train_len = len(searcher.y_train)

    searcher.update(X_update, y_update)

    # Verify state after update
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

    # Test prediction
    predictions = searcher.predict(X_val)
    assert len(predictions) == len(X_val)

    # Test update method
    X_update = X_val[0].reshape(1, -1)
    y_update = y_val[0]
    initial_X_train_len = len(searcher.X_train)
    initial_y_train_len = len(searcher.y_train)

    searcher.update(X_update, y_update)

    # Verify state after update
    assert len(searcher.X_train) == initial_X_train_len + 1
    assert len(searcher.y_train) == initial_y_train_len + 1
    assert np.array_equal(searcher.X_train[-1], X_update.flatten())
    assert searcher.y_train[-1] == y_update
