import pytest
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from unittest.mock import Mock
from confopt.selection.estimators.quantile_estimation import (
    QuantileLasso,
    QuantileGBM,
    QuantileForest,
    QuantileKNN,
    QuantileGP,
    QuantileLeaf,
    QuantRegWrapper,
)


def assess_quantile_quality(
    y_true: np.ndarray, predictions: np.ndarray, quantiles: List[float]
) -> Dict[str, Any]:
    n_samples, n_quantiles = predictions.shape

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

    coverage_errors = []
    for i, q in enumerate(quantiles):
        empirical_coverage = np.mean(y_true <= predictions[:, i])
        coverage_error = abs(empirical_coverage - q)
        coverage_errors.append(coverage_error)

    return {
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


QUALITY_THRESHOLDS = {
    "single_fit": {
        "max_hard_violation_rate": 0.0,
        "max_soft_violation_rate": 0.10,
        "max_coverage_error": 0.20,
    },
    "multi_fit": {
        "max_hard_violation_rate": 0.08,
        "max_soft_violation_rate": 0.18,
        "max_coverage_error": 0.20,
    },
}


@pytest.mark.parametrize(
    "data_fixture_name",
    [
        "linear_regression_data",
        "heteroscedastic_data",
        "diabetes_data",
    ],
)
@pytest.mark.parametrize(
    "estimator_class,estimator_params,estimator_type",
    [
        (
            QuantileGP,
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
    X, y = request.getfixturevalue(data_fixture_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Standardize features to avoid penalizing scale-sensitive estimators
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    quantiles = comprehensive_test_quantiles
    estimator = estimator_class(**estimator_params)

    try:
        estimator.fit(X_train, y_train, quantiles)
        predictions = estimator.predict(X_test)
    except Exception as e:
        pytest.fail(
            "Estimator {} failed on {}: {}".format(
                estimator_class.__name__, data_fixture_name, str(e)
            )
        )

    assert predictions.shape == (len(X_test), len(quantiles))
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))

    base_thresholds = QUALITY_THRESHOLDS[estimator_type].copy()

    quality_stats = assess_quantile_quality(y_test, predictions, quantiles)

    assert quality_stats["hard_rate"] <= base_thresholds["max_hard_violation_rate"]
    assert quality_stats["soft_rate"] <= base_thresholds["max_soft_violation_rate"]
    assert quality_stats["mean_coverage_error"] <= base_thresholds["max_coverage_error"]


def test_quantreg_wrapper_with_intercept():
    mock_results = Mock()
    mock_results.params = np.array([1.0, 2.0, 3.0])

    wrapper = QuantRegWrapper(mock_results, has_intercept=True)
    X_test = np.array([[1, 2], [3, 4]])

    predictions = wrapper.predict(X_test)
    expected = np.array([1 + 1 * 2 + 2 * 3, 1 + 3 * 2 + 4 * 3])

    np.testing.assert_array_equal(predictions, expected)


def test_quantreg_wrapper_without_intercept():
    mock_results = Mock()
    mock_results.params = np.array([2.0, 3.0])

    wrapper = QuantRegWrapper(mock_results, has_intercept=False)
    X_test = np.array([[1, 2], [3, 4]])

    predictions = wrapper.predict(X_test)
    expected = np.array([1 * 2 + 2 * 3, 3 * 2 + 4 * 3])

    np.testing.assert_array_equal(predictions, expected)
