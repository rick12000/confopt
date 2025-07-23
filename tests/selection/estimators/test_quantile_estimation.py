import pytest
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from unittest.mock import Mock
from confopt.selection.estimators.quantile_estimation import (
    QuantileLasso,
    QuantileGBM,
    QuantileLightGBM,
    QuantileForest,
    QuantileKNN,
    GaussianProcessQuantileEstimator,
    QuantileLeaf,
    QuantRegWrapper,
)


def assess_quantile_quality(
    y_true: np.ndarray, predictions: np.ndarray, quantiles: List[float]
) -> Dict[str, Any]:
    """Comprehensive quality assessment for quantile predictions."""
    n_samples, n_quantiles = predictions.shape

    # Pinball losses
    pinball_losses = []
    for i, q in enumerate(quantiles):
        loss = mean_pinball_loss(y_true, predictions[:, i], alpha=q)
        pinball_losses.append(loss)

    # Monotonicity violations
    hard_violations = 0
    soft_violations = 0
    violation_magnitudes = []
    tolerance = 1e-6

    for i in range(n_samples):
        pred_row = predictions[i, :]
        for j in range(n_quantiles - 1):
            diff = pred_row[j + 1] - pred_row[j]
            if diff < -tolerance:
                hard_violations += 1
                violation_magnitudes.append(abs(diff))
            elif abs(diff) <= tolerance:
                soft_violations += 1

    total_comparisons = n_samples * (n_quantiles - 1)

    # Coverage errors
    coverage_errors = []
    for i, q in enumerate(quantiles):
        empirical_coverage = np.mean(y_true <= predictions[:, i])
        coverage_error = abs(empirical_coverage - q)
        coverage_errors.append(coverage_error)

    return {
        "pinball_losses": pinball_losses,
        "mean_pinball_loss": np.mean(pinball_losses),
        "hard_violations": hard_violations,
        "soft_violations": soft_violations,
        "hard_rate": hard_violations / total_comparisons,
        "soft_rate": soft_violations / total_comparisons,
        "mean_violation_magnitude": np.mean(violation_magnitudes)
        if violation_magnitudes
        else 0.0,
        "coverage_errors": coverage_errors,
        "mean_coverage_error": np.mean(coverage_errors),
        "total_comparisons": total_comparisons,
    }


# Quality thresholds
QUALITY_THRESHOLDS = {
    "single_fit": {
        "max_hard_violation_rate": 0.0,
        "max_soft_violation_rate": 0.10,
        "max_mean_pinball_loss": 2.0,
        "max_coverage_error": 0.15,
    },
    "multi_fit": {
        "max_hard_violation_rate": 0.08,
        "max_soft_violation_rate": 0.18,
        "max_mean_pinball_loss": 3.0,
        "max_coverage_error": 0.20,
    },
}

# Estimator-specific adjustments for challenging cases
ESTIMATOR_ADJUSTMENTS = {
    "QuantileForest": {
        "challenging_monotonicity_data": {"max_soft_violation_rate": 0.20},
        "skewed_noise_data": {"max_coverage_error": 0.18},
    },
    "QuantileLeaf": {
        "challenging_monotonicity_data": {"max_soft_violation_rate": 0.12}
    },
    "QuantileGBM": {"high_dimensional_sparse_data": {"max_hard_violation_rate": 0.20}},
    "QuantileLasso": {
        "challenging_monotonicity_data": {"max_hard_violation_rate": 0.15}
    },
}

# Dataset-specific adjustments
DATASET_ADJUSTMENTS = {
    "challenging_monotonicity_data": {
        "multi_fit": {"max_hard_violation_rate": 0.12, "max_soft_violation_rate": 0.30}
    },
    "skewed_noise_data": {
        "multi_fit": {"max_hard_violation_rate": 0.10, "max_mean_pinball_loss": 4.0}
    },
    "high_dimensional_sparse_data": {
        "multi_fit": {"max_hard_violation_rate": 0.15, "max_soft_violation_rate": 0.25}
    },
}


@pytest.mark.parametrize(
    "data_fixture_name",
    [
        "linear_regression_data",
        "heteroscedastic_data",
        "multimodal_data",
        "skewed_noise_data",
        "high_dimensional_sparse_data",
        "challenging_monotonicity_data",
    ],
)
@pytest.mark.parametrize(
    "estimator_class,estimator_params,estimator_type",
    [
        # Single-fit estimators
        (
            GaussianProcessQuantileEstimator,
            {"kernel": "matern", "random_state": 42},
            "single_fit",
        ),
        (
            QuantileForest,
            {"n_estimators": 30, "max_depth": 6, "random_state": 42},
            "single_fit",
        ),
        (
            QuantileLeaf,
            {"n_estimators": 30, "max_depth": 6, "random_state": 42},
            "single_fit",
        ),
        (QuantileKNN, {"n_neighbors": 8}, "single_fit"),
        # Multi-fit estimators
        (
            QuantileGBM,
            {
                "learning_rate": 0.1,
                "n_estimators": 30,
                "min_samples_split": 8,
                "min_samples_leaf": 4,
                "max_depth": 4,
                "random_state": 42,
            },
            "multi_fit",
        ),
        (
            QuantileLightGBM,
            {"learning_rate": 0.1, "n_estimators": 30, "random_state": 42},
            "multi_fit",
        ),
        (
            QuantileLasso,
            {"max_iter": 1000, "p_tol": 1e-6, "random_state": 42},
            "multi_fit",
        ),
    ],
)
def test_quantile_estimator_comprehensive_quality(
    request,
    data_fixture_name,
    estimator_class,
    estimator_params,
    estimator_type,
    comprehensive_test_quantiles,
):
    """Comprehensive test for quantile estimator quality across all datasets and estimators."""
    X, y = request.getfixturevalue(data_fixture_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    quantiles = comprehensive_test_quantiles
    estimator = estimator_class(**estimator_params)

    # Fit and predict
    try:
        estimator.fit(X_train, y_train, quantiles)
        predictions = estimator.predict(X_test)
    except Exception as e:
        pytest.fail(
            f"Estimator {estimator_class.__name__} failed on {data_fixture_name}: {str(e)}"
        )

    # Basic validity checks
    assert predictions.shape == (
        len(X_test),
        len(quantiles),
    ), f"Wrong prediction shape: {predictions.shape}"
    assert not np.any(np.isnan(predictions)), "Predictions contain NaN values"
    assert not np.any(np.isinf(predictions)), "Predictions contain infinite values"

    # Get adjusted thresholds
    base_thresholds = QUALITY_THRESHOLDS[estimator_type].copy()

    # Apply dataset adjustments
    if (
        data_fixture_name in DATASET_ADJUSTMENTS
        and estimator_type in DATASET_ADJUSTMENTS[data_fixture_name]
    ):
        base_thresholds.update(DATASET_ADJUSTMENTS[data_fixture_name][estimator_type])

    # Apply estimator-specific adjustments
    if (
        estimator_class.__name__ in ESTIMATOR_ADJUSTMENTS
        and data_fixture_name in ESTIMATOR_ADJUSTMENTS[estimator_class.__name__]
    ):
        base_thresholds.update(
            ESTIMATOR_ADJUSTMENTS[estimator_class.__name__][data_fixture_name]
        )

    # Quality assessment and assertions
    quality_stats = assess_quantile_quality(y_test, predictions, quantiles)

    assert quality_stats["hard_rate"] <= base_thresholds["max_hard_violation_rate"], (
        f"{estimator_class.__name__} on {data_fixture_name}: "
        f"Hard violation rate {quality_stats['hard_rate']:.1%} exceeds threshold "
        f"{base_thresholds['max_hard_violation_rate']:.1%}. "
        f"Hard violations: {quality_stats['hard_violations']}/{quality_stats['total_comparisons']}"
    )

    assert quality_stats["soft_rate"] <= base_thresholds["max_soft_violation_rate"], (
        f"{estimator_class.__name__} on {data_fixture_name}: "
        f"Soft violation rate {quality_stats['soft_rate']:.1%} exceeds threshold "
        f"{base_thresholds['max_soft_violation_rate']:.1%}. "
        f"Soft violations: {quality_stats['soft_violations']}/{quality_stats['total_comparisons']}"
    )

    assert (
        quality_stats["mean_pinball_loss"] <= base_thresholds["max_mean_pinball_loss"]
    ), (
        f"{estimator_class.__name__} on {data_fixture_name}: "
        f"Mean pinball loss {quality_stats['mean_pinball_loss']:.4f} exceeds threshold "
        f"{base_thresholds['max_mean_pinball_loss']:.4f}. "
        f"Individual losses: {[f'{loss:.4f}' for loss in quality_stats['pinball_losses']]}"
    )

    assert (
        quality_stats["mean_coverage_error"] <= base_thresholds["max_coverage_error"]
    ), (
        f"{estimator_class.__name__} on {data_fixture_name}: "
        f"Mean coverage error {quality_stats['mean_coverage_error']:.4f} exceeds threshold "
        f"{base_thresholds['max_coverage_error']:.4f}. "
        f"Coverage errors: {[f'{err:.4f}' for err in quality_stats['coverage_errors']]}"
    )


def test_quantreg_wrapper_with_intercept():
    """Test QuantRegWrapper handles intercept correctly."""
    mock_results = Mock()
    mock_results.params = np.array([1.0, 2.0, 3.0])  # intercept + 2 features

    wrapper = QuantRegWrapper(mock_results, has_intercept=True)
    X_test = np.array([[1, 2], [3, 4]])

    predictions = wrapper.predict(X_test)
    expected = np.array(
        [1 + 1 * 2 + 2 * 3, 1 + 3 * 2 + 4 * 3]
    )  # intercept + feature products

    np.testing.assert_array_equal(predictions, expected)


def test_quantreg_wrapper_without_intercept():
    """Test QuantRegWrapper handles no intercept correctly."""
    mock_results = Mock()
    mock_results.params = np.array([2.0, 3.0])  # 2 features only

    wrapper = QuantRegWrapper(mock_results, has_intercept=False)
    X_test = np.array([[1, 2], [3, 4]])

    predictions = wrapper.predict(X_test)
    expected = np.array([1 * 2 + 2 * 3, 3 * 2 + 4 * 3])  # feature products only

    np.testing.assert_array_equal(predictions, expected)
