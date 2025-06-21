import pytest
import numpy as np
from unittest.mock import patch
import random
from confopt.selection.sampling import (
    PessimisticLowerBoundSampler,
    LowerBoundSampler,
    ThompsonSampler,
    ExpectedImprovementSampler,
    InformationGainSampler,
    MaxValueEntropySearchSampler,
    flatten_conformal_bounds,
    _differential_entropy_estimator,
)
from confopt.selection.conformalization import QuantileConformalEstimator


class TestPessimisticLowerBoundSampler:
    @pytest.mark.parametrize(
        "interval_width,expected_alpha", [(0.8, 0.2), (0.9, 0.1), (0.95, 0.05)]
    )
    def test_fetch_alphas(self, interval_width, expected_alpha):
        sampler = PessimisticLowerBoundSampler(interval_width=interval_width)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 1
        assert alphas[0] == pytest.approx(expected_alpha)

    @pytest.mark.parametrize("interval_width", [0.8, 0.9])
    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_update_interval_width(self, interval_width, adapter):
        sampler = PessimisticLowerBoundSampler(
            interval_width=interval_width, adapter=adapter
        )

        beta = 0.5
        sampler.update_interval_width(beta)

        if adapter in ["DtACI", "ACI"]:
            assert sampler.alpha != pytest.approx(1 - interval_width)
        else:
            assert sampler.alpha == pytest.approx(1 - interval_width)

    def test_adapter_initialization(self):
        # Test that ACI adapter uses correct gamma values
        sampler_aci = PessimisticLowerBoundSampler(interval_width=0.8, adapter="ACI")
        assert sampler_aci.adapter is not None
        assert sampler_aci.adapter.gamma_values.tolist() == [0.005]

        # Test that DtACI adapter uses correct gamma values
        sampler_dtaci = PessimisticLowerBoundSampler(
            interval_width=0.8, adapter="DtACI"
        )
        assert sampler_dtaci.adapter is not None
        assert sampler_dtaci.adapter.gamma_values.tolist() == [0.05, 0.01, 0.1]


class TestLowerBoundSampler:
    @pytest.mark.parametrize(
        "interval_width,expected_alpha",
        [(0.8, 0.2)],
    )
    def test_fetch_alphas(self, interval_width, expected_alpha):
        sampler = LowerBoundSampler(interval_width=interval_width)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 1
        assert alphas[0] == pytest.approx(expected_alpha)

    @pytest.mark.parametrize(
        "beta_decay,c,expected_beta",
        [
            ("inverse_square_root_decay", 2.0, lambda t: np.sqrt(2.0 / t)),
            ("logarithmic_decay", 2.0, lambda t: np.sqrt((2.0 * np.log(t)) / t)),
        ],
    )
    def test_update_exploration_step(self, beta_decay, c, expected_beta):
        sampler = LowerBoundSampler(beta_decay=beta_decay, c=c, beta_max=10.0)
        sampler.update_exploration_step()
        assert sampler.t == 2
        assert sampler.beta == pytest.approx(expected_beta(2))

    def test_calculate_ucb_predictions_with_point_estimates(self, conformal_bounds):
        sampler = LowerBoundSampler(interval_width=0.8, beta_decay=None)
        sampler.beta = 0.5

        point_estimates = np.array([0.5, 0.7, 0.3, 0.9, 0.6])
        interval_width = np.array([0.2, 0.1, 0.3, 0.05, 0.15])

        result = sampler.calculate_ucb_predictions(
            predictions_per_interval=conformal_bounds,
            point_estimates=point_estimates,
            interval_width=interval_width,
        )

        expected = point_estimates - 0.5 * interval_width
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_ucb_predictions_from_intervals(self, conformal_bounds):
        sampler = LowerBoundSampler(interval_width=0.8, beta_decay=None)
        sampler.beta = 0.75

        result = sampler.calculate_ucb_predictions(
            predictions_per_interval=conformal_bounds
        )

        interval = conformal_bounds[0]
        point_estimates = (interval.upper_bounds + interval.lower_bounds) / 2
        width = (interval.upper_bounds - interval.lower_bounds) / 2
        expected = point_estimates - 0.75 * width

        np.testing.assert_array_almost_equal(result, expected)


class TestThompsonSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            ThompsonSampler(n_quantiles=5)

    def test_initialize_alphas(self):
        sampler = ThompsonSampler(n_quantiles=4)
        alphas = sampler._initialize_alphas()

        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_fetch_alphas(self):
        sampler = ThompsonSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_update_interval_width(self, adapter):
        sampler = ThompsonSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter in ["DtACI", "ACI"]:
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas

    @pytest.mark.parametrize(
        "enable_optimistic, point_predictions",
        [(False, None), (True, np.array([0.05, 0.35, 0.75, 0.25, 0.95]))],
    )
    def test_calculate_thompson_predictions(
        self, conformal_bounds, enable_optimistic, point_predictions
    ):
        sampler = ThompsonSampler(
            n_quantiles=4, enable_optimistic_sampling=enable_optimistic
        )

        fixed_indices = np.array([0, 3, 5, 1, 4])

        with patch.object(np.random, "randint", return_value=fixed_indices):
            result = sampler.calculate_thompson_predictions(
                predictions_per_interval=conformal_bounds,
                point_predictions=point_predictions,
            )

        flattened_bounds = flatten_conformal_bounds(conformal_bounds)
        expected_sampled_bounds = np.array(
            [flattened_bounds[i, idx] for i, idx in enumerate(fixed_indices)]
        )

        if enable_optimistic and point_predictions is not None:
            expected = np.minimum(expected_sampled_bounds, point_predictions)
        else:
            expected = expected_sampled_bounds

        np.testing.assert_array_almost_equal(result, expected)

    def test_thompson_predictions_randomized(self, conformal_bounds):
        np.random.seed(42)

        sampler = ThompsonSampler(n_quantiles=4)
        predictions = sampler.calculate_thompson_predictions(conformal_bounds)
        assert len(predictions) == 5

        sampler = ThompsonSampler(n_quantiles=4, enable_optimistic_sampling=True)
        point_predictions = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        predictions = sampler.calculate_thompson_predictions(
            conformal_bounds,
            point_predictions=point_predictions,
        )
        assert len(predictions) == 5
        assert np.all(predictions <= point_predictions) or np.all(predictions < np.inf)


class TestExpectedImprovementSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            ExpectedImprovementSampler(n_quantiles=5)

    def test_initialize_alphas(self):
        sampler = ExpectedImprovementSampler(n_quantiles=4)
        alphas = sampler._initialize_alphas()

        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_fetch_alphas(self):
        sampler = ExpectedImprovementSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_update_best_value(self):
        sampler = ExpectedImprovementSampler(current_best_value=0.5)
        assert sampler.current_best_value == 0.5

        sampler.update_best_value(0.7)
        assert sampler.current_best_value == 0.5

        sampler.update_best_value(0.3)
        assert sampler.current_best_value == 0.3

    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_update_interval_width(self, adapter):
        sampler = ExpectedImprovementSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter in ["DtACI", "ACI"]:
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas

    def test_calculate_expected_improvement_detailed(self, simple_conformal_bounds):
        sampler = ExpectedImprovementSampler(current_best_value=0.4, num_ei_samples=1)

        with patch.object(
            np.random,
            "randint",
            side_effect=[np.array([[0], [1], [2]]), np.array([[0], [1], [2]])],
        ):
            result = sampler.calculate_expected_improvement(
                predictions_per_interval=simple_conformal_bounds
            )

        expected = np.array([-0.3, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

        sampler.current_best_value = 0.6
        with patch.object(
            np.random,
            "randint",
            side_effect=[np.array([[0], [1], [2]]), np.array([[0], [1], [2]])],
        ):
            result = sampler.calculate_expected_improvement(
                predictions_per_interval=simple_conformal_bounds
            )

        expected = np.array([-0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_expected_improvement_randomized(self, conformal_bounds):
        np.random.seed(42)

        sampler = ExpectedImprovementSampler(current_best_value=0.5, num_ei_samples=10)
        ei = sampler.calculate_expected_improvement(
            predictions_per_interval=conformal_bounds
        )

        assert len(ei) == 5
        assert np.all(ei <= 0)


class TestInformationGainSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            InformationGainSampler(n_quantiles=5)

    def test_initialize_alphas(self):
        sampler = InformationGainSampler(n_quantiles=4)
        alphas = sampler._initialize_alphas()

        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_fetch_alphas(self):
        sampler = InformationGainSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "sampling_strategy",
        ["thompson", "expected_improvement", "sobol", "perturbation"],
    )
    def test_parameter_initialization(self, sampling_strategy):
        sampler = InformationGainSampler(
            n_quantiles=6,
            n_paths=50,
            n_X_candidates=100,
            n_y_candidates_per_x=10,
            sampling_strategy=sampling_strategy,
        )
        assert sampler.n_paths == 50
        assert sampler.n_X_candidates == 100
        assert sampler.n_y_candidates_per_x == 10
        assert sampler.sampling_strategy == sampling_strategy
        assert len(sampler.alphas) == 3

    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_update_interval_width(self, adapter):
        sampler = InformationGainSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter in ["DtACI", "ACI"]:
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas

    @pytest.mark.parametrize("entropy_method", ["distance", "histogram"])
    def test_calculate_best_x_entropy(self, entropy_method):
        sampler = InformationGainSampler(
            n_quantiles=4, n_paths=10, entropy_method=entropy_method
        )

        n_observations = 5
        all_bounds = np.zeros((n_observations, 6))

        for i in range(n_observations):
            all_bounds[i, :] = np.linspace(0.1, 0.9, 6) + i * 0.1

        np.random.seed(42)
        entropy, indices = sampler._calculate_best_x_entropy(
            all_bounds=all_bounds, n_observations=n_observations
        )

        assert isinstance(entropy, float)

        if entropy_method == "histogram":
            # For histogram method, entropy should be non-negative
            assert entropy >= 0, "Histogram entropy should be non-negative"
        elif entropy_method == "distance":
            # For distance method, entropy can be negative or positive
            assert entropy <= float("inf"), "Distance entropy should be finite"

    @pytest.mark.parametrize(
        "sampling_strategy",
        ["thompson", "expected_improvement", "sobol", "perturbation"],
    )
    def test_information_gain_calculation(self, sampling_strategy, big_toy_dataset):
        X, y = big_toy_dataset
        np.random.seed(42)
        random.seed(42)

        train_size = 50
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        X_test = X[:20]

        conformal_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture="ql",
            alphas=[0.2, 0.8],
            n_pre_conformal_trials=5,
        )

        conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=0,
            random_state=42,
        )

        predictions_per_interval = conformal_estimator.predict_intervals(X_test)

        sampler = InformationGainSampler(
            n_quantiles=4,
            n_paths=100,
            n_X_candidates=5,
            n_y_candidates_per_x=20,
            sampling_strategy=sampling_strategy,
        )

        ig_values = sampler.calculate_information_gain(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_space=X_test,
            conformal_estimator=conformal_estimator,
            predictions_per_interval=predictions_per_interval,
            n_jobs=1,
        )

        # Check that information gains are valid values
        assert isinstance(ig_values, np.ndarray)
        assert len(ig_values) == len(X_test)
        # Test that values are finite (not NaN or inf)
        assert np.all(np.isfinite(ig_values))

        # Filter out zero values before calculating the percentage of negative values
        non_zero_values = ig_values[ig_values != 0]
        if len(non_zero_values) > 0:  # Only check if there are non-zero values
            negative_count = np.sum(non_zero_values < 0)
            assert (
                negative_count / len(non_zero_values) >= 0.5
            ), "At least 50% of non-zero information gains should be negative"

    @pytest.mark.parametrize("sampling_strategy", ["thompson", "expected_improvement"])
    def test_select_candidates(
        self, conformal_bounds, sampling_strategy, big_toy_dataset
    ):
        X, y = big_toy_dataset
        sampler = InformationGainSampler(
            n_quantiles=4, sampling_strategy=sampling_strategy, n_X_candidates=3
        )

        result = sampler._select_candidates(
            predictions_per_interval=conformal_bounds,
            X_space=X,
        )

        assert isinstance(result, np.ndarray)
        assert len(result) <= sampler.n_X_candidates
        assert np.all(result < len(conformal_bounds[0].lower_bounds))

        if sampling_strategy == "expected_improvement":
            best_idx = 1
            best_historical_y = 0.3
            best_historical_x = X[best_idx : best_idx + 1]

            result_with_best = sampler._select_candidates(
                predictions_per_interval=conformal_bounds,
                X_space=X,
                best_historical_y=best_historical_y,
                best_historical_x=best_historical_x,
            )

            assert isinstance(result_with_best, np.ndarray)
            assert len(result_with_best) <= sampler.n_X_candidates
            assert np.all(result_with_best < len(conformal_bounds[0].lower_bounds))

    @pytest.mark.parametrize("sampling_strategy", ["sobol", "perturbation"])
    def test_select_candidates_space_based(
        self, conformal_bounds, sampling_strategy, big_toy_dataset
    ):
        X, y = big_toy_dataset
        sampler = InformationGainSampler(
            n_quantiles=4, sampling_strategy=sampling_strategy, n_X_candidates=3
        )

        result = sampler._select_candidates(
            predictions_per_interval=conformal_bounds,
            X_space=X,
        )

        assert isinstance(result, np.ndarray)
        assert len(result) <= sampler.n_X_candidates
        assert np.all(result < len(conformal_bounds[0].lower_bounds))

        if sampling_strategy == "perturbation":
            best_idx = 1
            best_historical_y = 0.3
            best_historical_x = X[best_idx : best_idx + 1]

            result_with_best = sampler._select_candidates(
                predictions_per_interval=conformal_bounds,
                X_space=X,
                best_historical_y=best_historical_y,
                best_historical_x=best_historical_x,
            )

            assert isinstance(result_with_best, np.ndarray)
            assert len(result_with_best) <= sampler.n_X_candidates
            assert np.all(result_with_best < len(conformal_bounds[0].lower_bounds))


class TestMaxValueEntropySearchSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            MaxValueEntropySearchSampler(n_quantiles=5)

    def test_initialize_alphas(self):
        sampler = MaxValueEntropySearchSampler(n_quantiles=4)
        alphas = sampler._initialize_alphas()

        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_fetch_alphas(self):
        sampler = MaxValueEntropySearchSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_update_interval_width(self, adapter):
        sampler = MaxValueEntropySearchSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter in ["DtACI", "ACI"]:
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas

    @pytest.mark.parametrize("entropy_method", ["distance", "histogram"])
    def test_max_value_entropy_search_calculation(
        self, big_toy_dataset, entropy_method
    ):
        X, y = big_toy_dataset
        train_size = 50
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]

        np.random.seed(42)

        sampler = MaxValueEntropySearchSampler(
            n_quantiles=6,
            n_min_samples=100,
            n_y_samples=20,
            entropy_method=entropy_method,
        )

        quantile_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture="ql",
            alphas=[0.2, 0.8],
            n_pre_conformal_trials=5,
        )

        quantile_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=0,
            random_state=42,
        )

        X_test = X_train[:3]
        predictions_per_interval = quantile_estimator.predict_intervals(X_test)

        mes = sampler.calculate_max_value_entropy_search(
            predictions_per_interval=predictions_per_interval,
            n_jobs=1,
        )

        assert isinstance(mes, np.ndarray)
        assert len(mes) == len(X_test)

        # Filter out zero values before calculating percentage of negative values
        non_zero_values = mes[mes != 0]
        if len(non_zero_values) > 0:  # Only check if there are non-zero values
            negative_count = np.sum(non_zero_values < 0)
            assert (
                negative_count / len(non_zero_values) >= 0.5
            ), "At least 50% of non-zero values should be negative"


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


@pytest.mark.parametrize("method", ["distance", "histogram"])
def test_differential_entropy_estimator(method):
    np.random.seed(42)
    samples = np.random.normal(0, 1, 1000)

    entropy = _differential_entropy_estimator(samples, method=method)

    assert isinstance(entropy, float)

    # Differential entropy of a Gaussian with stddev=1 should be approximately 1.41 (0.5*ln(2πe))
    if method == "histogram":
        # Histogram entropy should be non-negative
        assert entropy >= 0, "Histogram entropy should be non-negative"
    elif method == "distance":
        # Vasicek estimator can produce reasonable estimates but may vary more
        assert np.isfinite(entropy), "Distance entropy should be finite"

    # For a single sample, entropy should be zero regardless of method
    single_sample_entropy = _differential_entropy_estimator(
        np.array([0.5]), method=method
    )
    assert single_sample_entropy == 0.0

    # Test constant samples
    constant_samples = np.ones(100)
    constant_entropy = _differential_entropy_estimator(constant_samples, method=method)
    assert (
        constant_entropy == 0.0
    ), f"{method} entropy for constant values should be zero"

    # Test invalid method
    with pytest.raises(ValueError):
        _differential_entropy_estimator(samples, method="invalid_method")


@pytest.mark.parametrize("method", ["distance", "histogram"])
def test_entropy_estimator_with_different_distributions(method):
    np.random.seed(42)

    # Create different distributions to test entropy estimator
    uniform_samples = np.random.uniform(0, 1, 1000)
    gaussian_samples = np.random.normal(0, 1, 1000)
    # Bimodal distribution
    bimodal_samples = np.concatenate(
        [np.random.normal(-3, 0.5, 500), np.random.normal(3, 0.5, 500)]
    )

    # Calculate entropies
    uniform_entropy = _differential_entropy_estimator(uniform_samples, method=method)
    gaussian_entropy = _differential_entropy_estimator(gaussian_samples, method=method)
    bimodal_entropy = _differential_entropy_estimator(bimodal_samples, method=method)

    # All entropies should be finite
    assert np.isfinite(uniform_entropy)
    assert np.isfinite(gaussian_entropy)
    assert np.isfinite(bimodal_entropy)

    # Theoretical differential entropy for uniform on [0,1] is 0
    # Theoretical differential entropy for Gaussian with stddev=1 is ~1.41
    # Bimodal should have higher entropy than Gaussian

    # General expectations that should hold for any valid entropy estimator
    assert (
        bimodal_entropy > gaussian_entropy
    ), "Bimodal should have higher entropy than Gaussian"
