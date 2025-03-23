import numpy as np
import pytest

from confopt.acquisition import (
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,
)
from confopt.sampling import (
    LowerBoundSampler,
    ThompsonSampler,
    PessimisticLowerBoundSampler,
)
from confopt.adaptation import ACI
from confopt.config import GBM_NAME, QGBM_NAME


@pytest.fixture
def sample_data():
    """Generate synthetic data for testing conformal methods"""
    np.random.seed(42)
    n_samples = 200
    n_features = 3

    # Generate features
    X = np.random.rand(n_samples, n_features) * 10

    # Generate target with heteroscedastic noise (variance increases with x)
    y_base = 3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2]
    noise_scale = 0.5 + 0.3 * X[:, 0]
    y = y_base + np.random.normal(0, noise_scale)

    # Split into train/val/test
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def fitted_locally_weighted_searcher(sample_data):
    """Create a fitted locally weighted conformal searcher"""
    sampler = LowerBoundSampler(c=2.0, interval_width=0.2)  # Removed beta parameter
    searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture=GBM_NAME,
        variance_estimator_architecture=GBM_NAME,
        sampler=sampler,
    )
    searcher.fit(
        X_train=sample_data["X_train"],
        y_train=sample_data["y_train"],
        X_val=sample_data["X_val"],
        y_val=sample_data["y_val"],
        random_state=42,
    )
    return searcher


@pytest.fixture
def fitted_quantile_searcher(sample_data):
    """Create a fitted multi-fit quantile conformal searcher"""
    sampler = LowerBoundSampler(c=2.0, interval_width=0.2)  # Removed beta parameter
    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture=QGBM_NAME, sampler=sampler
    )
    searcher.fit(
        X_train=sample_data["X_train"],
        y_train=sample_data["y_train"],
        X_val=sample_data["X_val"],
        y_val=sample_data["y_val"],
        random_state=42,
    )
    return searcher


class TestLocallyWeightedConformalSearcher:
    def test_fit(self, sample_data):
        """Test fit method correctly trains the conformal estimator"""
        sampler = LowerBoundSampler()
        searcher = LocallyWeightedConformalSearcher(
            point_estimator_architecture=GBM_NAME,
            variance_estimator_architecture=GBM_NAME,
            sampler=sampler,
        )

        # Fit the searcher
        searcher.fit(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            random_state=42,
        )

        # Check that estimators are fitted
        assert searcher.conformal_estimator.pe_estimator is not None
        assert searcher.conformal_estimator.ve_estimator is not None
        assert searcher.conformal_estimator.nonconformity_scores is not None
        assert searcher.primary_estimator_error is not None

    def test_predict_with_ucb(self, fitted_locally_weighted_searcher, sample_data):
        """Test prediction with UCB sampling strategy"""
        searcher = fitted_locally_weighted_searcher
        X_test = sample_data["X_test"]

        # Initial beta value
        initial_beta = searcher.sampler.beta
        initial_t = searcher.sampler.t

        # Make predictions
        predictions = searcher.predict(X_test)

        # Check prediction shape and type
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == X_test.shape[0]

        # Check that predictions_per_interval is updated
        assert searcher.predictions_per_interval is not None
        assert len(searcher.predictions_per_interval) == 1  # Default UCB has 1 interval
        assert searcher.predictions_per_interval[0].shape == (X_test.shape[0], 2)

        # Check that beta is updated
        assert searcher.sampler.t == initial_t + 1
        assert searcher.sampler.beta != initial_beta

    def test_predict_with_dtaci(self, sample_data):
        """Test prediction with DtACI adapter"""
        sampler = LowerBoundSampler(adapter_framework="DtACI")
        searcher = LocallyWeightedConformalSearcher(
            point_estimator_architecture=GBM_NAME,
            variance_estimator_architecture=GBM_NAME,
            sampler=sampler,
        )

        # Fit the searcher
        searcher.fit(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            random_state=42,
        )

        # Make predictions
        X_test = sample_data["X_test"]
        predictions = searcher.predict(X_test)

        # Check prediction shape
        assert predictions.shape[0] == X_test.shape[0]

        # Check that predictions_per_interval has multiple entries (one per expert alpha)
        assert len(searcher.predictions_per_interval) == len(sampler.expert_alphas)

    def test_update_interval_width(self, fitted_locally_weighted_searcher, sample_data):
        """Test updating interval width based on performance"""
        searcher = fitted_locally_weighted_searcher
        X_test = sample_data["X_test"]

        # Make predictions to populate predictions_per_interval
        searcher.predict(X_test)

        # Initial alpha
        initial_alpha = searcher.sampler.alpha

        # Update with a breach
        sampled_idx = 0
        sampled_performance = (
            searcher.predictions_per_interval[0][sampled_idx, 1] + 1
        )  # Above upper bound
        searcher.update_interval_width(sampled_idx, sampled_performance)

        # Alpha should decrease after breach with ACI
        if isinstance(searcher.sampler.adapter, ACI):
            assert searcher.sampler.alpha < initial_alpha

        # Update with no breach
        adjusted_alpha = searcher.sampler.alpha
        sampled_performance = (
            searcher.predictions_per_interval[0][sampled_idx, 0]
            + searcher.predictions_per_interval[0][sampled_idx, 1]
        ) / 2  # Within bounds
        searcher.update_interval_width(sampled_idx, sampled_performance)

        # Alpha should increase after no breach with ACI
        if isinstance(searcher.sampler.adapter, ACI):
            assert searcher.sampler.alpha > adjusted_alpha

    def test_predict_with_pessimistic_lower_bound(self, sample_data):
        """Test prediction with pessimistic lower bound strategy"""
        sampler = PessimisticLowerBoundSampler()
        searcher = LocallyWeightedConformalSearcher(
            point_estimator_architecture=GBM_NAME,
            variance_estimator_architecture=GBM_NAME,
            sampler=sampler,
        )

        # Fit the searcher
        searcher.fit(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            random_state=42,
        )

        # Make predictions
        X_test = sample_data["X_test"]
        predictions = searcher.predict(X_test)

        # Check prediction shape
        assert predictions.shape[0] == X_test.shape[0]

        # Check that predictions_per_interval is updated
        assert searcher.predictions_per_interval is not None
        assert len(searcher.predictions_per_interval) == 1


class TestQuantileConformalSearcher:
    def test_fit_with_ucb_sampler(self, sample_data):
        """Test fit method with UCB sampler"""
        sampler = LowerBoundSampler()
        searcher = QuantileConformalSearcher(
            quantile_estimator_architecture="qrf", sampler=sampler
        )

        # Fit the searcher
        searcher.fit(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            random_state=42,
        )

        # Check that estimator is fitted
        assert searcher.conformal_estimator.quantile_estimator is not None
        assert searcher.primary_estimator_error is not None
        assert searcher.point_estimator is None  # Not used with UCB

    def test_fit_with_thompson_optimistic(self, sample_data):
        """Test fit method with Thompson sampler and optimistic sampling"""
        sampler = ThompsonSampler(enable_optimistic_sampling=True)
        searcher = QuantileConformalSearcher(
            quantile_estimator_architecture="qrf", sampler=sampler
        )

        # Fit the searcher
        searcher.fit(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            random_state=42,
        )

        # Check that both estimators are fitted
        assert searcher.conformal_estimator.quantile_estimator is not None
        assert searcher.point_estimator is not None  # Used with optimistic Thompson

    def test_predict_with_ucb(self, fitted_single_fit_searcher, sample_data):
        """Test prediction with UCB sampling strategy"""
        searcher = fitted_single_fit_searcher
        X_test = sample_data["X_test"]

        # Initial beta value
        initial_beta = searcher.sampler.beta

        # Make predictions
        predictions = searcher.predict(X_test)

        # Check prediction shape and values
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == X_test.shape[0]

        # Check that predictions_per_interval is updated
        assert searcher.predictions_per_interval is not None
        assert len(searcher.predictions_per_interval) == 1  # Default UCB has 1 interval

        # Check that beta is updated
        assert searcher.sampler.beta != initial_beta

    def test_predict_with_thompson(self, sample_data):
        """Test prediction with Thompson sampling strategy"""
        sampler = ThompsonSampler(n_quantiles=4)
        searcher = QuantileConformalSearcher(
            quantile_estimator_architecture="qrf", sampler=sampler
        )

        # Fit the searcher
        searcher.fit(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            random_state=42,
        )

        # Make predictions
        X_test = sample_data["X_test"]
        np.random.seed(42)  # For reproducible Thompson sampling
        predictions = searcher.predict(X_test)

        # Check prediction shape
        assert predictions.shape[0] == X_test.shape[0]

        # Check that predictions_per_interval has one entry per interval
        assert len(searcher.predictions_per_interval) == len(sampler.quantiles)

        # Same seed should give identical predictions
        np.random.seed(42)
        predictions2 = searcher.predict(X_test)
        assert np.array_equal(predictions, predictions2)

        # Different seed should give different predictions due to sampling
        np.random.seed(99)
        predictions3 = searcher.predict(X_test)
        assert not np.array_equal(predictions, predictions3)

    def test_update_interval_width(self, fitted_single_fit_searcher, sample_data):
        """Test updating interval width based on performance"""
        searcher = fitted_single_fit_searcher
        X_test = sample_data["X_test"]

        # Predict to populate predictions_per_interval
        searcher.predict(X_test)

        # Initial alpha
        initial_alpha = searcher.sampler.alpha

        # Update with a breach
        sampled_idx = 0
        sampled_performance = (
            searcher.predictions_per_interval[0][sampled_idx, 1] + 1
        )  # Above upper bound
        searcher.update_interval_width(sampled_idx, sampled_performance)

        # Alpha should decrease after breach with ACI
        if isinstance(searcher.sampler.adapter, ACI):
            assert searcher.sampler.alpha < initial_alpha

    def test_predict_with_pessimistic_lower_bound(self, sample_data):
        """Test prediction with pessimistic lower bound strategy"""
        sampler = PessimisticLowerBoundSampler()
        searcher = QuantileConformalSearcher(
            quantile_estimator_architecture="qrf", sampler=sampler
        )

        # Fit the searcher
        searcher.fit(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            random_state=42,
        )

        # Make predictions
        X_test = sample_data["X_test"]
        predictions = searcher.predict(X_test)

        # Check prediction shape
        assert predictions.shape[0] == X_test.shape[0]

        # Check that predictions_per_interval is updated
        assert searcher.predictions_per_interval is not None
        assert len(searcher.predictions_per_interval) == 1
