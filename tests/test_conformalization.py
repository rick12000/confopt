import numpy as np
import pytest
from confopt.conformalization import (
    LocallyWeightedConformalEstimator,
    QuantileInterval,
    SingleFitQuantileConformalEstimator,
    MultiFitQuantileConformalEstimator,
)
from confopt.config import (
    MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    POINT_ESTIMATOR_ARCHITECTURES,
    SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
)

# Global variable for coverage tolerance
COVERAGE_TOLERANCE = 0.01


class TestLocallyWeightedConformalEstimator:
    @pytest.mark.parametrize("estimator_architecture", POINT_ESTIMATOR_ARCHITECTURES)
    @pytest.mark.parametrize("tuning_iterations", [0, 2])
    def test_fit_component_estimator(
        self,
        estimator_architecture,
        tuning_iterations,
        dummy_expanding_quantile_gaussian_dataset,
    ):
        """Test _fit_component_estimator private method"""
        estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=estimator_architecture,
            variance_estimator_architecture=estimator_architecture,
        )

        # Prepare data
        X, y = dummy_expanding_quantile_gaussian_dataset
        train_split = 0.8
        X_train, y_train = (
            X[: round(len(X) * train_split)],
            y[: round(len(y) * train_split)],
        )

        # Test with parameterized tuning iterations
        fitted_est = estimator._tune_fit_component_estimator(
            X=X_train,
            y=y_train,
            estimator_architecture=estimator_architecture,
            tuning_iterations=tuning_iterations,
            random_state=42,
        )

        # Verify estimator is initialized and has predict method
        assert fitted_est is not None
        assert hasattr(fitted_est, "predict")

        # Test predictions
        predictions = fitted_est.predict(X_train)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == X_train.shape[0]

    @pytest.mark.parametrize("point_arch", POINT_ESTIMATOR_ARCHITECTURES)
    @pytest.mark.parametrize("variance_arch", POINT_ESTIMATOR_ARCHITECTURES)
    @pytest.mark.parametrize("tuning_iterations", [0, 2])
    def test_fit_and_predict_interval(
        self,
        point_arch,
        variance_arch,
        tuning_iterations,
        dummy_expanding_quantile_gaussian_dataset,
    ):
        """Test complete fit and predict_interval workflow with variable tuning iterations"""
        estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=point_arch,
            variance_estimator_architecture=variance_arch,
        )

        # Prepare data - use smaller subset for testing
        X, y = dummy_expanding_quantile_gaussian_dataset

        train_split = 0.8
        X_train, y_train = (
            X[: round(len(X) * train_split)],
            y[: round(len(y) * train_split)],
        )
        X_val, y_val = (
            X[round(len(X) * train_split) :],
            y[round(len(y) * train_split) :],
        )

        # Fit the estimator with parameterized tuning iterations
        estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=42,
        )

        # Test predict_interval with just one confidence level
        alphas = [0.8]  # Reduced from three levels to just one
        for alpha in alphas:
            lower_bound, upper_bound = estimator.predict_interval(X=X_val, alpha=alpha)

            assert np.all(lower_bound <= upper_bound)

            coverage = np.mean(
                (y_val >= lower_bound.flatten()) & (y_val <= upper_bound.flatten())
            )
            assert abs((1 - coverage) - alpha) < COVERAGE_TOLERANCE


class TestSingleFitQuantileConformalEstimator:
    @pytest.mark.parametrize(
        "estimator_architecture",
        SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    )
    @pytest.mark.parametrize("tuning_iterations", [0, 2])
    def test_fit_and_predict_interval(
        self,
        estimator_architecture,
        tuning_iterations,
        dummy_expanding_quantile_gaussian_dataset,
    ):
        """Test complete fit and predict_interval workflow with variable tuning iterations"""
        # Create intervals for testing
        intervals = [
            QuantileInterval(lower_quantile=0.1, upper_quantile=0.9),
        ]

        estimator = SingleFitQuantileConformalEstimator(
            quantile_estimator_architecture=estimator_architecture,
            intervals=intervals,
            n_pre_conformal_trials=5,  # Reduced from 20 to 5
        )

        # Prepare data - use smaller subset
        X, y = dummy_expanding_quantile_gaussian_dataset

        train_split = 0.8
        X_train, y_train = (
            X[: round(len(X) * train_split)],
            y[: round(len(y) * train_split)],
        )
        X_val, y_val = (
            X[round(len(X) * train_split) :],
            y[round(len(y) * train_split) :],
        )

        # Fit the estimator with parameterized tuning iterations
        estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=42,
        )

        assert len(estimator.nonconformity_scores) == len(intervals)

        # Test predict_interval for the interval
        for interval in intervals:
            lower_bound, upper_bound = estimator.predict_interval(
                X=X_val, interval=interval
            )

            # Check that lower bounds are <= upper bounds
            assert np.all(lower_bound <= upper_bound)

            # Check interval coverage (approximate)
            target_coverage = interval.upper_quantile - interval.lower_quantile
            actual_coverage = np.mean((y_val >= lower_bound) & (y_val <= upper_bound))
            assert abs(actual_coverage - target_coverage) < COVERAGE_TOLERANCE


class TestMultiFitQuantileConformalEstimator:
    @pytest.mark.parametrize(
        "estimator_architecture",
        MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    )
    @pytest.mark.parametrize("tuning_iterations", [0, 2])
    def test_fit_and_predict_interval(
        self,
        estimator_architecture,
        tuning_iterations,
        dummy_expanding_quantile_gaussian_dataset,
    ):
        """Test complete fit and predict_interval workflow with variable tuning iterations"""
        interval = QuantileInterval(lower_quantile=0.1, upper_quantile=0.9)
        estimator = MultiFitQuantileConformalEstimator(
            quantile_estimator_architecture=estimator_architecture,
            interval=interval,
            n_pre_conformal_trials=5,  # Reduced from 20 to 5
        )

        # Prepare data
        X, y = dummy_expanding_quantile_gaussian_dataset

        train_split = 0.8
        X_train, y_train = (
            X[: round(len(X) * train_split)],
            y[: round(len(y) * train_split)],
        )
        X_val, y_val = (
            X[round(len(X) * train_split) :],
            y[round(len(y) * train_split) :],
        )

        # Fit the estimator with parameterized tuning iterations
        estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=42,
        )

        # Test predict_interval
        lower_bound, upper_bound = estimator.predict_interval(X=X_val)

        # Check that lower bounds are <= upper bounds
        assert np.all(lower_bound <= upper_bound)

        # Check interval coverage (approximate)
        interval = estimator.interval
        target_coverage = interval.upper_quantile - interval.lower_quantile
        actual_coverage = np.mean((y_val >= lower_bound) & (y_val <= upper_bound))
        assert abs(actual_coverage - target_coverage) < COVERAGE_TOLERANCE
