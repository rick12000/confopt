import numpy as np
import pytest
from copy import deepcopy

# Remove scipy imports and add the proper range types
from confopt.ranges import IntRange, FloatRange

from confopt.estimation import (
    initialize_point_estimator,
    initialize_quantile_estimator,
    cross_validate_configurations,
    average_scores_across_folds,
    tune,
    SEARCH_MODEL_DEFAULT_CONFIGURATIONS,
    SEARCH_MODEL_TUNING_SPACE,
)
from confopt.config import (
    GBM_NAME,
    RF_NAME,
    QGBM_NAME,
    QLGBM_NAME,
    KNN_NAME,
    LGBM_NAME,
)


class TestEstimatorInitialization:
    @pytest.mark.parametrize("architecture", [GBM_NAME, RF_NAME, LGBM_NAME])
    def test_point_estimator_initialization_reproducibility(self, architecture):
        """Test that point estimators initialized with the same random state produce the same predictions"""
        # Setup
        config = deepcopy(SEARCH_MODEL_DEFAULT_CONFIGURATIONS[architecture])
        X = np.random.rand(100, 5)

        # Create two estimators with the same random state
        estimator1 = initialize_point_estimator(
            estimator_architecture=architecture,
            initialization_params=config,
            random_state=42,
        )
        estimator2 = initialize_point_estimator(
            estimator_architecture=architecture,
            initialization_params=config,
            random_state=42,
        )

        # Train both on the same data
        y = np.random.rand(100)
        estimator1.fit(X, y)
        estimator2.fit(X, y)

        # Check that predictions are identical
        X_test = np.random.rand(20, 5)
        pred1 = estimator1.predict(X_test)
        pred2 = estimator2.predict(X_test)

        assert np.array_equal(pred1, pred2)

    @pytest.mark.parametrize("architecture", [QGBM_NAME, QLGBM_NAME])
    def test_quantile_estimator_initialization_reproducibility(self, architecture):
        """Test that quantile estimators initialized with the same random state produce the same predictions"""
        # Setup
        config = deepcopy(SEARCH_MODEL_DEFAULT_CONFIGURATIONS[architecture])
        X = np.random.rand(100, 5)
        quantiles = [0.25, 0.75]

        # Create two estimators with the same random state
        estimator1 = initialize_quantile_estimator(
            estimator_architecture=architecture,
            initialization_params=config,
            pinball_loss_alpha=quantiles,
            random_state=42,
        )
        estimator2 = initialize_quantile_estimator(
            estimator_architecture=architecture,
            initialization_params=config,
            pinball_loss_alpha=quantiles,
            random_state=42,
        )

        # Train both on the same data
        y = np.random.rand(100)
        estimator1.fit(X, y)
        estimator2.fit(X, y)

        # Check that predictions are identical
        X_test = np.random.rand(20, 5)
        pred1 = estimator1.predict(X_test)
        pred2 = estimator2.predict(X_test)

        assert np.array_equal(pred1, pred2)

    def test_point_estimator_config_respect(self):
        """Test that point estimators respect the configuration parameters provided"""
        # Test a few key parameters for GBM
        special_config = {"n_estimators": 123, "learning_rate": 0.07, "max_depth": 7}

        estimator = initialize_point_estimator(
            estimator_architecture=GBM_NAME,
            initialization_params=special_config,
            random_state=42,
        )

        # Verify key parameters were respected
        assert estimator.n_estimators == 123
        assert estimator.learning_rate == 0.07
        assert estimator.max_depth == 7


class TestCrossValidation:
    def test_average_scores_across_folds(self):
        """Test that average_scores_across_folds correctly aggregates scores"""
        # Setup test data
        configs = [{"param": 1}, {"param": 2}, {"param": 1}]
        scores = [0.1, 0.2, 0.3]

        # Call the function
        aggregated_configs, aggregated_scores = average_scores_across_folds(
            configs, scores
        )

        # Verify results
        assert len(aggregated_configs) == 2  # Unique configurations
        assert len(aggregated_scores) == 2  # One score per unique config

        # Check the actual aggregation
        if aggregated_configs[0] == {"param": 1}:
            assert abs(aggregated_scores[0] - 0.2) < 1e-5  # (0.1 + 0.3) / 2
            assert abs(aggregated_scores[1] - 0.2) < 1e-5  # Just 0.2
        else:
            assert abs(aggregated_scores[1] - 0.2) < 1e-5  # (0.1 + 0.3) / 2
            assert abs(aggregated_scores[0] - 0.2) < 1e-5  # Just 0.2

    def test_cross_validate_configurations_reproducibility(self):
        """Test that cross validation with the same random state produces the same results"""
        # Setup
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        configs = [
            {"n_estimators": 50, "max_features": 0.8},
            {"n_estimators": 100, "max_features": 0.5},
        ]

        # Run cross-validation twice with the same random state
        scored_configs1, scores1 = cross_validate_configurations(
            configurations=configs,
            estimator_architecture=RF_NAME,
            X=X,
            y=y,
            k_fold_splits=3,
            random_state=42,
        )

        scored_configs2, scores2 = cross_validate_configurations(
            configurations=configs,
            estimator_architecture=RF_NAME,
            X=X,
            y=y,
            k_fold_splits=3,
            random_state=42,
        )

        # Verify results are identical
        assert scored_configs1 == scored_configs2
        assert scores1 == scores2

    def test_cross_validate_quantile_estimators(self):
        """Test cross-validation with quantile estimators"""
        # Setup
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        configs = [
            {
                "n_estimators": 50,
                "learning_rate": 0.1,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_depth": 3,
            },
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_depth": 5,
            },
        ]
        quantiles = [0.25, 0.75]

        # Run cross-validation
        scored_configs, scores = cross_validate_configurations(
            configurations=configs,
            estimator_architecture=QGBM_NAME,
            X=X,
            y=y,
            k_fold_splits=2,
            quantiles=quantiles,
            random_state=42,
        )

        # Verify results make sense
        assert len(scored_configs) == len(scores)
        assert all(score > 0 for score in scores)  # Pinball loss should be positive


class TestTuning:
    def test_tune_finds_best_configuration(self):
        """Test that tune returns the configuration with the lowest cross-validation score"""
        # Create synthetic data where a specific configuration should work better
        np.random.seed(42)
        X = np.random.rand(100, 5)
        # Make y strongly correlated with the first feature
        y = 3 * X[:, 0] + 0.5 * np.random.randn(100)

        # Mock the tuning space for testing
        original_tuning_space = SEARCH_MODEL_TUNING_SPACE[KNN_NAME]
        SEARCH_MODEL_TUNING_SPACE[KNN_NAME] = {
            "n_neighbors": IntRange(min_value=1, max_value=10)
        }

        try:
            # Run tuning
            best_config = tune(
                X=X, y=y, estimator_architecture=KNN_NAME, n_searches=3, random_state=42
            )

            # For this specific problem, the best configuration should have a lower n_neighbors
            assert best_config["n_neighbors"] <= 5  # We expect 1 or 2 to be best
        finally:
            # Restore the original tuning space
            SEARCH_MODEL_TUNING_SPACE[KNN_NAME] = original_tuning_space

    def test_tune_reproducibility(self):
        """Test that tuning with the same random state produces the same results"""
        # Setup
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        # Store original tuning space
        original_tuning_space = SEARCH_MODEL_TUNING_SPACE[GBM_NAME]
        # Create a test tuning space with custom parameter ranges
        test_tuning_space = {
            "n_estimators": IntRange(min_value=50, max_value=100),
            "learning_rate": FloatRange(min_value=0.01, max_value=0.1),
            "max_depth": IntRange(min_value=3, max_value=7),
        }
        SEARCH_MODEL_TUNING_SPACE[GBM_NAME] = test_tuning_space

        try:
            # Run tuning twice with the same random state
            best_config1 = tune(
                X=X,
                y=y,
                estimator_architecture=GBM_NAME,
                n_searches=5,  # Small number for faster testing
                random_state=42,
            )

            best_config2 = tune(
                X=X, y=y, estimator_architecture=GBM_NAME, n_searches=5, random_state=42
            )

            # Verify results are identical
            assert best_config1 == best_config2
        finally:
            # Restore original tuning space
            SEARCH_MODEL_TUNING_SPACE[GBM_NAME] = original_tuning_space


def test_end_to_end_model_selection():
    """Test the complete model selection process from tuning to initialization"""
    # Setup synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.exp(X[:, 0] + 0.5 * X[:, 1]) + 0.1 * np.random.randn(100)

    # Split into train/test
    split_idx = 80
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, _ = y[:split_idx], y[split_idx:]

    # Create a smaller search space for faster testing using proper parameter ranges
    test_tuning_space = {
        "n_estimators": IntRange(min_value=50, max_value=100),
        "learning_rate": FloatRange(min_value=0.05, max_value=0.1),
        "max_depth": IntRange(min_value=3, max_value=5),
    }

    original_tuning_space = SEARCH_MODEL_TUNING_SPACE[GBM_NAME]
    SEARCH_MODEL_TUNING_SPACE[GBM_NAME] = test_tuning_space

    try:
        # Step 1: Tune hyperparameters
        best_config = tune(
            X=X_train,
            y=y_train,
            estimator_architecture=GBM_NAME,
            n_searches=4,  # All combinations in test_tuning_space
            random_state=42,
        )

        # Step 2: Initialize the model with best config
        model = initialize_point_estimator(
            estimator_architecture=GBM_NAME,
            initialization_params=best_config,
            random_state=42,
        )

        # Step 3: Train and evaluate
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Verify predictions make sense
        assert predictions.shape == (X_test.shape[0],)
        assert not np.any(np.isnan(predictions))

        # Verify model has the tuned parameters
        for param, value in best_config.items():
            assert getattr(model, param) == value
    finally:
        # Restore the original tuning space
        SEARCH_MODEL_TUNING_SPACE[GBM_NAME] = original_tuning_space
