import numpy as np
import pytest
import gc
from confopt.conformalization import (
    # MedianEstimator,
    # PointEstimator,
    LocallyWeightedConformalEstimator,
    QuantileInterval,
    SingleFitQuantileConformalEstimator,
    MultiFitQuantileConformalEstimator,
)
from confopt.config import (
    MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
    POINT_ESTIMATOR_ARCHITECTURES,
)


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up resources after each test to prevent memory accumulation."""
    yield
    # Force garbage collection to clean up any lingering resources
    gc.collect()


# class TestMedianEstimator:
#     @pytest.mark.parametrize("estimator_architecture", QUANTILE_ESTIMATOR_ARCHITECTURES)
#     def test_initialization(self, estimator_architecture):
#         """Test that MedianEstimator initializes correctly"""
#         estimator = MedianEstimator(
#             quantile_estimator_architecture=estimator_architecture
#         )
#         assert estimator.quantile_estimator_architecture == estimator_architecture
#         assert estimator.median_estimator is None

#     @pytest.mark.parametrize(
#         "estimator_architecture", QUANTILE_ESTIMATOR_ARCHITECTURES[:2]
#     )  # Limit to 2 for speed
#     def test_fit_and_predict(
#         self, estimator_architecture, dummy_fixed_quantile_dataset
#     ):
#         """Test that MedianEstimator fits and predicts correctly"""
#         estimator = MedianEstimator(
#             quantile_estimator_architecture=estimator_architecture
#         )

#         # Prepare data
#         X, y = (
#             dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
#             dummy_fixed_quantile_dataset[:, 1],
#         )
#         train_split = 0.8
#         X_train, y_train = (
#             X[: round(len(X) * train_split), :],
#             y[: round(len(y) * train_split)],
#         )
#         X_test = X[round(len(X) * train_split) :, :]

#         # Fit the estimator
#         estimator.fit(X=X_train, y=y_train, random_state=42)

#         # Verify estimator is fitted
#         assert estimator.median_estimator is not None

#         # Test predictions
#         predictions = estimator.predict(X_test)
#         assert isinstance(predictions, np.ndarray)
#         assert predictions.shape[0] == X_test.shape[0]

#     def test_predict_error(self):
#         """Test error case - predict before fit"""
#         estimator = MedianEstimator(
#             quantile_estimator_architecture=QUANTILE_ESTIMATOR_ARCHITECTURES[0]
#         )
#         with pytest.raises(ValueError):
#             estimator.predict(np.random.rand(10, 1))


# class TestPointEstimator:
#     @pytest.mark.parametrize("estimator_architecture", POINT_ESTIMATOR_ARCHITECTURES[:2])
#     def test_initialization(self, estimator_architecture):
#         """Test that PointEstimator initializes correctly"""
#         estimator = PointEstimator(
#             point_estimator_architecture=estimator_architecture
#         )
#         assert estimator.point_estimator_architecture == estimator_architecture
#         assert estimator.point_estimator is None

#     @pytest.mark.parametrize(
#         "estimator_architecture", POINT_ESTIMATOR_ARCHITECTURES[:2]
#     )  # Limit to 2 for speed
#     def test_fit_and_predict(
#         self, estimator_architecture, dummy_fixed_quantile_dataset
#     ):
#         """Test that PointEstimator fits and predicts correctly"""
#         estimator = PointEstimator(
#             point_estimator_architecture=estimator_architecture
#         )

#         # Prepare data
#         X, y = (
#             dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
#             dummy_fixed_quantile_dataset[:, 1],
#         )
#         train_split = 0.8
#         X_train, y_train = (
#             X[: round(len(X) * train_split), :],
#             y[: round(len(y) * train_split)],
#         )
#         X_test = X[round(len(X) * train_split) :, :]

#         # Fit the estimator
#         estimator.fit(X=X_train, y=y_train, random_state=42)

#         # Verify estimator is fitted
#         assert estimator.point_estimator is not None

#         # Test predictions
#         predictions = estimator.predict(X_test)
#         assert isinstance(predictions, np.ndarray)
#         assert predictions.shape[0] == X_test.shape[0]

#     def test_predict_error(self):
#         """Test error case - predict before fit"""
#         estimator = PointEstimator(
#             point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0]
#         )
#         with pytest.raises(ValueError):
#             estimator.predict(np.random.rand(10, 1))


class TestLocallyWeightedConformalEstimator:
    # Reduce parameter combinations significantly for initialization test
    @pytest.mark.parametrize("point_arch", POINT_ESTIMATOR_ARCHITECTURES)
    @pytest.mark.parametrize("variance_arch", POINT_ESTIMATOR_ARCHITECTURES)
    def test_initialization(self, point_arch, variance_arch):
        """Test that LocallyWeightedConformalEstimator initializes correctly"""
        estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=point_arch,
            variance_estimator_architecture=variance_arch,
        )
        assert estimator.point_estimator_architecture == point_arch
        assert estimator.variance_estimator_architecture == variance_arch
        assert estimator.pe_estimator is None
        assert estimator.ve_estimator is None
        assert estimator.nonconformity_scores is None

    @pytest.mark.parametrize("estimator_architecture", POINT_ESTIMATOR_ARCHITECTURES)
    def test_fit_component_estimator(
        self, estimator_architecture, dummy_fixed_quantile_dataset
    ):
        """Test _fit_component_estimator private method"""
        estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=estimator_architecture,
            variance_estimator_architecture=estimator_architecture,
        )

        # Prepare data
        X, y = (
            dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
            dummy_fixed_quantile_dataset[:, 1],
        )
        train_split = 0.8
        X_train, y_train = (
            X[: round(len(X) * train_split), :],
            y[: round(len(y) * train_split)],
        )

        # Test with default configurations (no tuning)
        fitted_est = estimator._fit_component_estimator(
            X=X_train,
            y=y_train,
            estimator_architecture=estimator_architecture,
            tuning_iterations=0,
            random_state=42,
        )

        # Verify estimator is initialized and has predict method
        assert fitted_est is not None
        assert hasattr(fitted_est, "predict")

        # Test predictions
        predictions = fitted_est.predict(X_train)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == X_train.shape[0]

    @pytest.mark.parametrize(
        "point_arch", POINT_ESTIMATOR_ARCHITECTURES
    )  # Drastically reduced combinations
    @pytest.mark.parametrize(
        "variance_arch", POINT_ESTIMATOR_ARCHITECTURES
    )  # Drastically reduced combinations
    def test_fit_and_predict_interval(
        self, point_arch, variance_arch, dummy_fixed_quantile_dataset
    ):
        """Test complete fit and predict_interval workflow"""
        estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=point_arch,
            variance_estimator_architecture=variance_arch,
        )

        # Prepare data - use smaller subset for testing
        X, y = (
            dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
            dummy_fixed_quantile_dataset[:, 1],
        )

        # Use a smaller subset to reduce memory usage
        max_samples = min(len(X), 100)  # Limit to maximum 100 samples
        X = X[:max_samples]
        y = y[:max_samples]

        train_split = 0.8
        X_train, y_train = (
            X[: round(len(X) * train_split), :],
            y[: round(len(y) * train_split)],
        )
        X_val, y_val = (
            X[round(len(X) * train_split) :, :],
            y[round(len(y) * train_split) :],
        )

        # Fit the estimator
        estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=0,
            random_state=42,
        )

        # Verify estimator components are fitted
        assert estimator.pe_estimator is not None
        assert estimator.ve_estimator is not None
        assert estimator.nonconformity_scores is not None
        assert estimator.training_time is not None
        assert estimator.primary_estimator_error is not None

        # Test predict_interval with just one confidence level
        confidence_levels = [0.8]  # Reduced from three levels to just one
        for alpha in confidence_levels:
            lower_bound, upper_bound = estimator.predict_interval(X=X_val, alpha=alpha)

            # Check shapes and types
            assert isinstance(lower_bound, np.ndarray)
            assert isinstance(upper_bound, np.ndarray)
            assert lower_bound.shape[0] == X_val.shape[0]
            assert upper_bound.shape[0] == X_val.shape[0]

            # Check that lower bounds are <= upper bounds
            assert np.all(lower_bound <= upper_bound)

            # Check interval coverage (approximate)
            coverage = np.mean(
                (y_val >= lower_bound.flatten()) & (y_val <= upper_bound.flatten())
            )
            assert abs(coverage - alpha) < 0.2  # Allow for some error in coverage

        # Explicitly delete estimator to free resources
        del estimator
        gc.collect()

    def test_predict_interval_error(self):
        """Test error handling in predict_interval"""
        estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
            variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        )
        X = np.random.rand(10, 1)
        with pytest.raises(ValueError):
            estimator.predict_interval(X=X, alpha=0.8)


class TestQuantileInterval:
    def test_initialization(self):
        """Test QuantileInterval initialization and properties"""
        intervals = [(0.1, 0.9), (0.25, 0.75), (0.4, 0.6)]

        for lower, upper in intervals:
            interval = QuantileInterval(lower_quantile=lower, upper_quantile=upper)
            assert interval.lower_quantile == lower
            assert interval.upper_quantile == upper


class TestSingleFitQuantileConformalEstimator:
    @pytest.mark.parametrize(
        "estimator_architecture", ["qrf"]  # Reduced to one architecture
    )
    def test_initialization(self, estimator_architecture):
        """Test SingleFitQuantileConformalEstimator initialization"""
        estimator = SingleFitQuantileConformalEstimator(
            quantile_estimator_architecture=estimator_architecture,
            n_pre_conformal_trials=5,  # Reduced from 20 to 5
        )
        assert estimator.quantile_estimator_architecture == estimator_architecture
        assert estimator.n_pre_conformal_trials == 5  # Updated assertion
        assert estimator.quantile_estimator is None
        assert estimator.nonconformity_scores == {}
        assert estimator.fitted_quantiles is None

    def test_interval_key(self):
        """Test _interval_key private method"""
        estimator = SingleFitQuantileConformalEstimator(
            quantile_estimator_architecture="qrf",
            n_pre_conformal_trials=5,  # Reduced from 20 to 5
        )
        interval = QuantileInterval(lower_quantile=0.1, upper_quantile=0.9)
        key = estimator._interval_key(interval)
        assert key == "0.1_0.9"

        # Explicitly delete estimator to free resources
        del estimator
        gc.collect()

    @pytest.mark.parametrize(
        "estimator_architecture", ["qrf"]  # Reduced to one architecture
    )
    def test_fit_and_predict_interval(
        self, estimator_architecture, dummy_fixed_quantile_dataset
    ):
        """Test complete fit and predict_interval workflow"""
        estimator = SingleFitQuantileConformalEstimator(
            quantile_estimator_architecture=estimator_architecture,
            n_pre_conformal_trials=5,  # Reduced from 20 to 5
        )

        # Prepare data - use smaller subset
        X, y = (
            dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
            dummy_fixed_quantile_dataset[:, 1],
        )

        # Use a smaller subset to reduce memory usage
        max_samples = min(len(X), 100)  # Limit to maximum 100 samples
        X = X[:max_samples]
        y = y[:max_samples]

        train_split = 0.8
        X_train, y_train = (
            X[: round(len(X) * train_split), :],
            y[: round(len(y) * train_split)],
        )
        X_val, y_val = (
            X[round(len(X) * train_split) :, :],
            y[round(len(y) * train_split) :],
        )

        # Create intervals for testing - reduced to one interval
        intervals = [
            QuantileInterval(lower_quantile=0.1, upper_quantile=0.9),
        ]

        # Fit the estimator
        estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            intervals=intervals,
            tuning_iterations=0,
            random_state=42,
        )

        # Verify estimator is fitted
        assert estimator.quantile_estimator is not None
        assert estimator.fitted_quantiles is not None
        assert len(estimator.fitted_quantiles) == 2  # Unique quantiles: 0.1, 0.9
        assert estimator.training_time is not None
        assert estimator.primary_estimator_error is not None

        # Test predict_interval for both intervals
        for interval in intervals:
            lower_bound, upper_bound = estimator.predict_interval(
                X=X_val, interval=interval
            )

            # Check shapes and types
            assert isinstance(lower_bound, np.ndarray)
            assert isinstance(upper_bound, np.ndarray)
            assert lower_bound.shape[0] == X_val.shape[0]
            assert upper_bound.shape[0] == X_val.shape[0]

            # Check that lower bounds are <= upper bounds
            assert np.all(lower_bound <= upper_bound)

            # Check interval coverage (approximate)
            target_coverage = interval.upper_quantile - interval.lower_quantile
            actual_coverage = np.mean((y_val >= lower_bound) & (y_val <= upper_bound))
            assert abs(actual_coverage - target_coverage) < 0.2

        # Explicitly delete estimator to free resources
        del estimator
        gc.collect()

    def test_predict_interval_error(self):
        """Test error handling in predict_interval"""
        estimator = SingleFitQuantileConformalEstimator(
            quantile_estimator_architecture="qrf",
            n_pre_conformal_trials=5,  # Reduced from 20 to 5
        )
        X = np.random.rand(10, 1)
        interval = QuantileInterval(lower_quantile=0.1, upper_quantile=0.9)

        with pytest.raises(ValueError):
            estimator.predict_interval(X=X, interval=interval)

        # Explicitly delete estimator to free resources
        del estimator
        gc.collect()


class TestMultiFitQuantileConformalEstimator:
    @pytest.mark.parametrize(
        "estimator_architecture", MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES
    )  # Reduced to one architecture
    def test_initialization(self, estimator_architecture):
        """Test MultiFitQuantileConformalEstimator initialization"""
        interval = QuantileInterval(lower_quantile=0.1, upper_quantile=0.9)
        estimator = MultiFitQuantileConformalEstimator(
            quantile_estimator_architecture=estimator_architecture,
            interval=interval,
            n_pre_conformal_trials=5,  # Reduced from 20 to 5
        )
        assert estimator.quantile_estimator_architecture == estimator_architecture
        assert estimator.interval == interval
        assert estimator.n_pre_conformal_trials == 5  # Updated assertion
        assert estimator.quantile_estimator is None
        assert estimator.nonconformity_scores is None
        assert estimator.conformalize_predictions is False

        # Explicitly delete estimator to free resources
        del estimator
        gc.collect()

    @pytest.mark.parametrize(
        "estimator_architecture",
        MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,  # Reduced to one architecture
    )
    def test_fit_and_predict_interval(
        self, estimator_architecture, dummy_fixed_quantile_dataset
    ):
        """Test complete fit and predict_interval workflow"""
        interval = QuantileInterval(lower_quantile=0.1, upper_quantile=0.9)
        estimator = MultiFitQuantileConformalEstimator(
            quantile_estimator_architecture=estimator_architecture,
            interval=interval,
            n_pre_conformal_trials=5,  # Reduced from 20 to 5
        )

        # Prepare data
        X, y = (
            dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
            dummy_fixed_quantile_dataset[:, 1],
        )

        # Use a smaller subset to reduce memory usage
        max_samples = min(len(X), 100)  # Limit to maximum 100 samples
        X = X[:max_samples]
        y = y[:max_samples]

        train_split = 0.8
        X_train, y_train = (
            X[: round(len(X) * train_split), :],
            y[: round(len(y) * train_split)],
        )
        X_val, y_val = (
            X[round(len(X) * train_split) :, :],
            y[round(len(y) * train_split) :],
        )

        # Fit the estimator
        estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=0,
            random_state=42,
        )

        # Verify estimator is fitted
        assert estimator.quantile_estimator is not None
        assert estimator.training_time is not None
        assert estimator.primary_estimator_error is not None

        # Test predict_interval
        lower_bound, upper_bound = estimator.predict_interval(X=X_val)

        # Check shapes and types
        assert isinstance(lower_bound, np.ndarray)
        assert isinstance(upper_bound, np.ndarray)
        assert lower_bound.shape[0] == X_val.shape[0]
        assert upper_bound.shape[0] == X_val.shape[0]

        # Check that lower bounds are <= upper bounds
        assert np.all(lower_bound <= upper_bound)

        # Check interval coverage (approximate)
        interval = estimator.interval
        target_coverage = interval.upper_quantile - interval.lower_quantile
        actual_coverage = np.mean((y_val >= lower_bound) & (y_val <= upper_bound))
        assert abs(actual_coverage - target_coverage) < 0.2

        # Explicitly delete estimator to free resources
        del estimator
        gc.collect()

    def test_predict_interval_error(self):
        """Test error handling in predict_interval"""
        interval = QuantileInterval(lower_quantile=0.1, upper_quantile=0.9)
        estimator = MultiFitQuantileConformalEstimator(
            quantile_estimator_architecture=MULTI_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES[
                0
            ],
            interval=interval,
            n_pre_conformal_trials=5,  # Reduced from 20 to 5
        )
        X = np.random.rand(10, 1)

        with pytest.raises(ValueError):
            estimator.predict_interval(X=X)

        # Explicitly delete estimator to free resources
        del estimator
        gc.collect()
