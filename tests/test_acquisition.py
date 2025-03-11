import numpy as np
import pytest

from confopt.acquisition import (
    UCBSampler,
    ThompsonSampler,
    PessimisticLowerBoundSampler,
    LocallyWeightedConformalSearcher,
    SingleFitQuantileConformalSearcher,
    MultiFitQuantileConformalSearcher,
)
from confopt.adaptation import ACI, DtACI
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
    sampler = UCBSampler(c=2.0, interval_width=0.2)  # Removed beta parameter
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
def fitted_single_fit_searcher(sample_data):
    """Create a fitted single-fit quantile conformal searcher"""
    sampler = UCBSampler(c=2.0, interval_width=0.2)  # Removed beta parameter
    searcher = SingleFitQuantileConformalSearcher(
        quantile_estimator_architecture="qrf", sampler=sampler
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
def fitted_multi_fit_searcher(sample_data):
    """Create a fitted multi-fit quantile conformal searcher"""
    sampler = UCBSampler(c=2.0, interval_width=0.2)  # Removed beta parameter
    searcher = MultiFitQuantileConformalSearcher(
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


class TestUCBSampler:
    def test_adapter_initialization(self):
        """Test adapter initialization with different frameworks"""
        # ACI adapter
        sampler1 = UCBSampler(adapter_framework="ACI")
        assert isinstance(sampler1.adapter, ACI)
        assert sampler1.adapter.alpha == sampler1.alpha

        # DtACI adapter
        sampler2 = UCBSampler(adapter_framework="DtACI")
        assert isinstance(sampler2.adapter, DtACI)
        assert hasattr(sampler2, "expert_alphas")

        # Invalid adapter
        with pytest.raises(ValueError, match="Unknown adapter framework:"):
            UCBSampler(adapter_framework="InvalidAdapter")

    def test_update_exploration_step(self):
        """Test beta updating with different decay strategies"""
        # Test logarithmic decay
        sampler1 = UCBSampler(
            beta_decay="logarithmic_decay", c=2.0
        )  # Removed beta parameter
        assert sampler1.t == 1
        assert sampler1.beta == 1.0  # Default beta value

        sampler1.update_exploration_step()
        assert sampler1.t == 2
        assert sampler1.beta == 2.0 * np.log(1) / 1  # c * log(t) / t

        sampler1.update_exploration_step()
        assert sampler1.t == 3
        assert sampler1.beta == 2.0 * np.log(2) / 2  # c * log(t) / t

        # Test logarithmic growth
        sampler2 = UCBSampler(beta_decay="logarithmic_growth")  # Removed beta parameter
        assert sampler2.t == 1
        assert sampler2.beta == 1.0  # Default beta value

        sampler2.update_exploration_step()
        assert sampler2.t == 2
        assert sampler2.beta == 2 * np.log(2)  # 2 * log(t + 1)

        sampler2.update_exploration_step()
        assert sampler2.t == 3
        assert sampler2.beta == 2 * np.log(3)  # 2 * log(t + 1)

    def test_update_interval_width(self):
        """Test interval width updating with adapters"""
        # Test ACI adapter
        sampler1 = UCBSampler(adapter_framework="ACI")
        initial_alpha = sampler1.alpha

        # Mock a breach
        sampler1.update_interval_width([1])  # breach
        assert sampler1.alpha < initial_alpha  # Alpha should decrease after breach

        # Mock no breach
        adjusted_alpha = sampler1.alpha
        sampler1.update_interval_width([0])  # no breach
        assert sampler1.alpha > adjusted_alpha  # Alpha should increase after no breach

        # Test ACI with incorrect breach list length
        with pytest.raises(ValueError):
            sampler1.update_interval_width([0, 1])  # Should be single element

        # Test DtACI adapter
        sampler2 = UCBSampler(adapter_framework="DtACI")
        initial_alpha = sampler2.alpha

        # Get the correct number of experts from the adapter
        num_experts = len(sampler2.expert_alphas)

        # Mock breaches - use the correct number of breach indicators
        breaches = [1] * (num_experts - 1) + [0]  # One success, others breach
        sampler2.update_interval_width(breaches)  # Provide correct number of indicators
        assert sampler2.alpha != initial_alpha  # Alpha should adjust

        # Verify quantiles are recalculated
        new_quantiles = sampler2.fetch_interval()
        assert new_quantiles.lower_quantile == sampler2.alpha / 2
        assert new_quantiles.upper_quantile == 1 - (sampler2.alpha / 2)


class TestThompsonSampler:
    def test_quantile_initialization(self):
        """Test quantiles and alphas are correctly initialized"""
        sampler = ThompsonSampler(n_quantiles=4)

        # Check quantiles
        assert len(sampler.quantiles) == 2

        # First interval should be (0.2, 0.8)
        assert sampler.quantiles[0].lower_quantile == 0.2
        assert sampler.quantiles[0].upper_quantile == 0.8

        # Second interval should be (0.4, 0.6)
        assert sampler.quantiles[1].lower_quantile == 0.4
        assert sampler.quantiles[1].upper_quantile == 0.6

        # Check alphas (1 - (upper - lower))
        assert sampler.alphas[0] == 1 - (0.8 - 0.2)  # = 0.4
        assert sampler.alphas[1] == 1 - (0.6 - 0.4)  # = 0.8

    def test_adapter_initialization(self):
        """Test adapter initialization with ThompsonSampler"""
        # With ACI framework
        sampler = ThompsonSampler(n_quantiles=4, adapter_framework="ACI")
        assert len(sampler.adapters) == 2  # One per interval
        assert all(isinstance(adapter, ACI) for adapter in sampler.adapters)

        # With invalid framework
        with pytest.raises(ValueError):
            ThompsonSampler(adapter_framework="InvalidAdapter")

    def test_update_interval_width(self):
        """Test interval width updating with ThompsonSampler"""
        sampler = ThompsonSampler(n_quantiles=4, adapter_framework="ACI")
        original_alphas = sampler.alphas.copy()

        # Update with breaches
        sampler.update_interval_width([1, 0])  # First interval breached, second not

        # First alpha should decrease (breach), second should increase (no breach)
        assert sampler.alphas[0] < original_alphas[0]
        assert sampler.alphas[1] > original_alphas[1]

        # Verify quantiles are updated correctly
        assert sampler.quantiles[0].lower_quantile == sampler.alphas[0] / 2
        assert sampler.quantiles[0].upper_quantile == 1 - (sampler.alphas[0] / 2)


class TestPessimisticLowerBoundSampler:
    def test_initialization(self):
        """Test initialization with different adapter frameworks"""
        # Default initialization
        sampler = PessimisticLowerBoundSampler()
        assert sampler.interval_width == 0.8
        assert pytest.approx(sampler.alpha) == 0.2
        assert sampler.adapter is None

        # ACI adapter
        sampler_aci = PessimisticLowerBoundSampler(adapter_framework="ACI")
        assert isinstance(sampler_aci.adapter, ACI)
        assert sampler_aci.adapter.alpha == sampler_aci.alpha

        # DtACI adapter
        sampler_dtaci = PessimisticLowerBoundSampler(adapter_framework="DtACI")
        assert isinstance(sampler_dtaci.adapter, DtACI)
        assert hasattr(sampler_dtaci, "expert_alphas")

        # Invalid adapter
        with pytest.raises(ValueError):
            PessimisticLowerBoundSampler(adapter_framework="InvalidAdapter")

    def test_fetch_interval(self):
        """Test fetch_interval returns correct quantile interval"""
        sampler = PessimisticLowerBoundSampler(interval_width=0.9)
        interval = sampler.fetch_interval()
        assert pytest.approx(interval.lower_quantile) == 0.05
        assert pytest.approx(interval.upper_quantile) == 0.95

    def test_update_interval_width(self):
        """Test interval width updating with adapters"""
        # Test ACI adapter
        sampler = PessimisticLowerBoundSampler(adapter_framework="ACI")
        initial_alpha = sampler.alpha

        # Mock a breach
        sampler.update_interval_width([1])  # breach
        assert sampler.alpha < initial_alpha  # Alpha should decrease after breach

        # Mock no breach
        adjusted_alpha = sampler.alpha
        sampler.update_interval_width([0])  # no breach
        assert sampler.alpha > adjusted_alpha  # Alpha should increase after no breach

        # Test DtACI adapter
        sampler2 = PessimisticLowerBoundSampler(adapter_framework="DtACI")
        initial_alpha = sampler2.alpha

        # Get the correct number of experts from the adapter
        num_experts = len(sampler2.expert_alphas)

        # Mock breaches with correct number of indicators
        breaches = [0] * num_experts  # all no breach
        sampler2.update_interval_width(breaches)  # mixed breaches
        assert sampler2.alpha != initial_alpha  # Alpha should adjust


class TestLocallyWeightedConformalSearcher:
    def test_fit(self, sample_data):
        """Test fit method correctly trains the conformal estimator"""
        sampler = UCBSampler()
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
        assert searcher.training_time is not None
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
        sampler = UCBSampler(adapter_framework="DtACI")
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


class TestSingleFitQuantileConformalSearcher:
    def test_fit_with_ucb_sampler(self, sample_data):
        """Test fit method with UCB sampler"""
        sampler = UCBSampler()
        searcher = SingleFitQuantileConformalSearcher(
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
        assert searcher.training_time is not None
        assert searcher.primary_estimator_error is not None
        assert searcher.point_estimator is None  # Not used with UCB

    def test_fit_with_thompson_optimistic(self, sample_data):
        """Test fit method with Thompson sampler and optimistic sampling"""
        sampler = ThompsonSampler(enable_optimistic_sampling=True)
        searcher = SingleFitQuantileConformalSearcher(
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
        searcher = SingleFitQuantileConformalSearcher(
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
        searcher = SingleFitQuantileConformalSearcher(
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


class TestMultiFitQuantileConformalSearcher:
    def test_fit_with_ucb_sampler(self, sample_data):
        """Test fit method with UCB sampler"""
        sampler = UCBSampler()
        searcher = MultiFitQuantileConformalSearcher(
            quantile_estimator_architecture=QGBM_NAME, sampler=sampler
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
        assert len(searcher.conformal_estimators) == 1  # One estimator for UCB
        assert searcher.conformal_estimators[0].quantile_estimator is not None
        assert searcher.training_time is not None
        assert searcher.primary_estimator_error is not None

    def test_fit_with_thompson_sampler(self, sample_data):
        """Test fit method with Thompson sampler"""
        sampler = ThompsonSampler(n_quantiles=4)
        searcher = MultiFitQuantileConformalSearcher(
            quantile_estimator_architecture=QGBM_NAME, sampler=sampler
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
        assert (
            len(searcher.conformal_estimators) == 2
        )  # Two intervals for n_quantiles=4
        for estimator in searcher.conformal_estimators:
            assert estimator.quantile_estimator is not None

    def test_predict_with_ucb(self, fitted_multi_fit_searcher, sample_data):
        """Test prediction with UCB sampling strategy"""
        searcher = fitted_multi_fit_searcher
        X_test = sample_data["X_test"]

        # Initial beta value
        initial_beta = searcher.sampler.beta

        # Make predictions
        predictions = searcher.predict(X_test)

        # Check prediction shape
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == X_test.shape[0]

        # Check that predictions_per_interval is updated
        assert searcher.predictions_per_interval is not None
        assert len(searcher.predictions_per_interval) == 1  # Default UCB has 1 interval

        # Check that beta is updated
        assert searcher.sampler.beta != initial_beta

    def test_predict_with_thompson(self, sample_data):
        """Test prediction with Thompson sampling strategy"""
        sampler = ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=True)
        searcher = MultiFitQuantileConformalSearcher(
            quantile_estimator_architecture=QGBM_NAME, sampler=sampler
        )

        # Fit the searcher
        searcher.fit(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            random_state=42,
        )

        # Check that median estimator is fitted (for optimistic sampling)
        assert searcher.point_estimator is not None

        # Make predictions
        X_test = sample_data["X_test"]
        np.random.seed(42)  # For reproducible Thompson sampling
        predictions = searcher.predict(X_test)

        # Check prediction shape
        assert predictions.shape[0] == X_test.shape[0]

        # Check that predictions_per_interval has one entry per interval
        assert len(searcher.predictions_per_interval) == len(
            searcher.conformal_estimators
        )

    def test_predict_with_pessimistic_lower_bound(self, sample_data):
        """Test prediction with pessimistic lower bound strategy"""
        sampler = PessimisticLowerBoundSampler()
        searcher = MultiFitQuantileConformalSearcher(
            quantile_estimator_architecture=QGBM_NAME, sampler=sampler
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

        # Check that the predictions are actually the lower bounds from the interval
        lower_bound = searcher.predictions_per_interval[0][:, 0]
        assert np.array_equal(predictions, lower_bound)
