import pytest
import numpy as np
import random
from confopt.selection.sampling.entropy_samplers import (
    EntropySearchSampler,
    MaxValueEntropySearchSampler,
    calculate_entropy,
)
from confopt.selection.sampling.utils import initialize_quantile_alphas
from confopt.selection.conformalization import QuantileConformalEstimator


class TestInformationGainSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            EntropySearchSampler(n_quantiles=5)

    def test_initialize_alphas_via_utils(self):
        # Test the utility function directly since the method is now abstracted
        alphas = initialize_quantile_alphas(4)
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    def test_fetch_alphas(self):
        sampler = EntropySearchSampler(n_quantiles=4)
        alphas = sampler.fetch_alphas()
        assert len(alphas) == 2
        assert alphas[0] == pytest.approx(0.4)
        assert alphas[1] == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "sampling_strategy",
        ["thompson", "expected_improvement", "sobol", "perturbation"],
    )
    def test_parameter_initialization(self, sampling_strategy):
        sampler = EntropySearchSampler(
            n_quantiles=6,
            n_paths=50,
            n_x_candidates=100,
            n_y_candidates_per_x=10,
            sampling_strategy=sampling_strategy,
        )
        assert sampler.n_paths == 50
        assert sampler.n_x_candidates == 100
        assert sampler.n_y_candidates_per_x == 10
        assert sampler.sampling_strategy == sampling_strategy
        assert len(sampler.alphas) == 3

    @pytest.mark.parametrize("adapter", [None, "DtACI", "ACI"])
    def test_update_interval_width(self, adapter):
        sampler = EntropySearchSampler(n_quantiles=4, adapter=adapter)
        betas = [0.3, 0.5]
        previous_alphas = sampler.alphas.copy()

        sampler.update_interval_width(betas)

        if adapter in ["DtACI", "ACI"]:
            assert sampler.alphas != previous_alphas
        else:
            assert sampler.alphas == previous_alphas

    @pytest.mark.parametrize("entropy_method", ["distance", "histogram"])
    def test_calculate_best_x_entropy(self, entropy_method):
        sampler = EntropySearchSampler(
            n_quantiles=4, n_paths=10, entropy_measure=entropy_method
        )

        n_observations = 5
        all_bounds = np.zeros((n_observations, 6))

        for i in range(n_observations):
            all_bounds[i, :] = np.linspace(0.1, 0.9, 6) + i * 0.1

        np.random.seed(42)
        entropy, indices = sampler.get_entropy_of_optimum_location(
            all_bounds=all_bounds, n_observations=n_observations
        )

        assert isinstance(entropy, float)

        if entropy_method == "histogram":
            assert entropy >= 0
        elif entropy_method == "distance":
            assert entropy <= float("inf")

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

        sampler = EntropySearchSampler(
            n_quantiles=4,
            n_paths=100,
            n_x_candidates=5,
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

        assert isinstance(ig_values, np.ndarray)
        assert len(ig_values) == len(X_test)
        assert np.all(np.isfinite(ig_values))

        non_zero_values = ig_values[ig_values != 0]
        if len(non_zero_values) > 0:
            negative_count = np.sum(non_zero_values < 0)
            assert negative_count / len(non_zero_values) >= 0.5

    @pytest.mark.parametrize("sampling_strategy", ["thompson", "expected_improvement"])
    def test_select_candidates(
        self, conformal_bounds, sampling_strategy, big_toy_dataset
    ):
        X, y = big_toy_dataset
        sampler = EntropySearchSampler(
            n_quantiles=4, sampling_strategy=sampling_strategy, n_x_candidates=3
        )

        result = sampler.select_candidates(
            predictions_per_interval=conformal_bounds,
            candidate_space=X,
        )

        assert isinstance(result, np.ndarray)
        assert len(result) <= sampler.n_x_candidates
        assert np.all(result < len(conformal_bounds[0].lower_bounds))

        if sampling_strategy == "expected_improvement":
            best_idx = 1
            best_historical_y = 0.3
            best_historical_x = X[best_idx : best_idx + 1]

            result_with_best = sampler.select_candidates(
                predictions_per_interval=conformal_bounds,
                candidate_space=X,
                best_historical_y=best_historical_y,
                best_historical_x=best_historical_x,
            )

            assert isinstance(result_with_best, np.ndarray)
            assert len(result_with_best) <= sampler.n_x_candidates
            assert np.all(result_with_best < len(conformal_bounds[0].lower_bounds))

    @pytest.mark.parametrize("sampling_strategy", ["sobol", "perturbation"])
    def test_select_candidates_space_based(
        self, conformal_bounds, sampling_strategy, big_toy_dataset
    ):
        X, y = big_toy_dataset
        sampler = EntropySearchSampler(
            n_quantiles=4, sampling_strategy=sampling_strategy, n_x_candidates=3
        )

        result = sampler.select_candidates(
            predictions_per_interval=conformal_bounds,
            candidate_space=X,
        )

        assert isinstance(result, np.ndarray)
        assert len(result) <= sampler.n_x_candidates
        assert np.all(result < len(conformal_bounds[0].lower_bounds))

        if sampling_strategy == "perturbation":
            best_idx = 1
            best_historical_y = 0.3
            best_historical_x = X[best_idx : best_idx + 1]

            result_with_best = sampler.select_candidates(
                predictions_per_interval=conformal_bounds,
                candidate_space=X,
                best_historical_y=best_historical_y,
                best_historical_x=best_historical_x,
            )

            assert isinstance(result_with_best, np.ndarray)
            assert len(result_with_best) <= sampler.n_x_candidates
            assert np.all(result_with_best < len(conformal_bounds[0].lower_bounds))


class TestMaxValueEntropySearchSampler:
    def test_init_odd_quantiles(self):
        with pytest.raises(ValueError):
            MaxValueEntropySearchSampler(n_quantiles=5)

    def test_initialize_alphas_via_utils(self):
        # Test the utility function directly since the method is now abstracted
        alphas = initialize_quantile_alphas(4)
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
            n_paths=100,
            n_y_candidates_per_x=20,
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

        mes = sampler.calculate_information_gain(
            predictions_per_interval=predictions_per_interval,
            n_jobs=1,
        )

        assert isinstance(mes, np.ndarray)
        assert len(mes) == len(X_test)

        non_zero_values = mes[mes != 0]
        if len(non_zero_values) > 0:
            negative_count = np.sum(non_zero_values < 0)
            assert negative_count / len(non_zero_values) >= 0.5


@pytest.mark.parametrize("method", ["distance", "histogram"])
def test_differential_entropy_estimator(method):
    np.random.seed(42)
    samples = np.random.normal(0, 1, 1000)

    entropy = calculate_entropy(samples, method=method)

    assert isinstance(entropy, float)

    if method == "histogram":
        assert entropy >= 0
    elif method == "distance":
        assert np.isfinite(entropy)

    single_sample_entropy = calculate_entropy(np.array([0.5]), method=method)
    assert single_sample_entropy == 0.0

    constant_samples = np.ones(100)
    constant_entropy = calculate_entropy(constant_samples, method=method)
    assert constant_entropy == 0.0

    with pytest.raises(ValueError):
        calculate_entropy(samples, method="invalid_method")


@pytest.mark.parametrize("method", ["distance", "histogram"])
def test_entropy_estimator_with_different_distributions(method):
    np.random.seed(42)

    uniform_samples = np.random.uniform(0, 1, 1000)
    gaussian_samples = np.random.normal(0, 1, 1000)
    bimodal_samples = np.concatenate(
        [np.random.normal(-3, 0.5, 500), np.random.normal(3, 0.5, 500)]
    )

    uniform_entropy = calculate_entropy(uniform_samples, method=method)
    gaussian_entropy = calculate_entropy(gaussian_samples, method=method)
    bimodal_entropy = calculate_entropy(bimodal_samples, method=method)

    assert np.isfinite(uniform_entropy)
    assert np.isfinite(gaussian_entropy)
    assert np.isfinite(bimodal_entropy)

    assert bimodal_entropy > gaussian_entropy
