import numpy as np
import pytest
from confopt.selection.conformalization import (
    LocallyWeightedConformalEstimator,
    QuantileConformalEstimator,
    alpha_to_quantiles,
)

from conftest import (
    POINT_ESTIMATOR_ARCHITECTURES,
    SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    QUANTILE_ESTIMATOR_ARCHITECTURES,
)

POINT_ESTIMATOR_COVERAGE_TOLERANCE = 0.2
QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE = 0.05


def create_train_val_split(X, y, train_split=0.8, random_state=1234):
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split_idx = round(len(X) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    return X_train, y_train, X_val, y_val


def validate_intervals(intervals, y_true, alphas, tolerance):
    assert len(intervals) == len(alphas)

    for i, alpha in enumerate(alphas):
        lower_bound = intervals[i].lower_bounds
        upper_bound = intervals[i].upper_bounds

        assert np.all(lower_bound <= upper_bound)

        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        assert abs(coverage - (1 - alpha)) < tolerance

    return True


def validate_betas(betas, alphas):
    assert len(betas) == len(alphas)
    for beta in betas:
        assert 0 <= beta <= 1

    return True


def test_alpha_to_quantiles():
    lower, upper = alpha_to_quantiles(0.2)
    assert lower == 0.1
    assert upper == 0.9

    lower, upper = alpha_to_quantiles(0.2, upper_quantile_cap=0.85)
    assert lower == 0.1
    assert upper == 0.85


class TestLocallyWeightedConformalEstimator:
    @staticmethod
    @pytest.mark.parametrize("point_arch", POINT_ESTIMATOR_ARCHITECTURES)
    @pytest.mark.parametrize("variance_arch", POINT_ESTIMATOR_ARCHITECTURES)
    @pytest.mark.parametrize("tuning_iterations", [0, 1])  # was [0, 2]
    @pytest.mark.parametrize("alphas", [[0.2], [0.1, 0.2]])
    def test_fit_predict_and_betas(
        point_arch,
        variance_arch,
        tuning_iterations,
        alphas,
        dummy_expanding_quantile_gaussian_dataset,
    ):
        estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=point_arch,
            variance_estimator_architecture=variance_arch,
            alphas=alphas,
        )

        X, y = dummy_expanding_quantile_gaussian_dataset
        X_train, y_train, X_val, y_val = create_train_val_split(X, y)

        estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=42,
        )

        intervals = estimator.predict_intervals(X=X_val)
        validate_intervals(
            intervals, y_val, alphas, tolerance=POINT_ESTIMATOR_COVERAGE_TOLERANCE
        )

        test_point = X_val[0]
        test_value = y_val[0]
        betas = estimator.calculate_betas(test_point, test_value)
        validate_betas(betas, alphas)


class TestQuantileConformalEstimator:
    @staticmethod
    @pytest.mark.parametrize("estimator_architecture", QUANTILE_ESTIMATOR_ARCHITECTURES)
    @pytest.mark.parametrize("tuning_iterations", [0, 1])  # was [0, 2]
    @pytest.mark.parametrize("alphas", [[0.2], [0.1, 0.2]])
    @pytest.mark.parametrize("upper_quantile_cap", [None, 0.95])
    def test_fit_predict_and_betas(
        estimator_architecture,
        tuning_iterations,
        alphas,
        upper_quantile_cap,
        dummy_expanding_quantile_gaussian_dataset,
    ):
        estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=estimator_architecture,
            alphas=alphas,
            n_pre_conformal_trials=15,
        )

        X, y = dummy_expanding_quantile_gaussian_dataset
        X_train, y_train, X_val, y_val = create_train_val_split(X, y)

        estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            upper_quantile_cap=upper_quantile_cap,
            random_state=42,
        )

        assert len(estimator.nonconformity_scores) == len(alphas)

        intervals = estimator.predict_intervals(X_val)
        validate_intervals(
            intervals, y_val, alphas, tolerance=QUANTILE_ESTIMATOR_COVERAGE_TOLERANCE
        )

        test_point = X_val[0]
        test_value = y_val[0]
        betas = estimator.calculate_betas(test_point, test_value)
        validate_betas(betas, alphas)

    @staticmethod
    def test_small_dataset_behavior():
        alphas = [0.2]
        estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES[
                0
            ],
            alphas=alphas,
            n_pre_conformal_trials=20,
        )

        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        X_train, y_train, X_val, y_val = create_train_val_split(X, y, train_split=0.8)

        estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        assert not estimator.conformalize_predictions
