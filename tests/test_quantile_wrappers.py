import numpy as np
from confopt.selection.quantile_estimators import QuantRegressionWrapper, QuantileLasso


def test_quantreg_wrapper_intercept_handling():
    """Test that QuantRegressionWrapper correctly handles intercept columns."""
    # Create synthetic data
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(100, 3))  # 100 samples, 3 features
    beta = np.array([2.5, 1.0, -0.5])  # True coefficients
    epsilon = np.random.normal(0, 0.5, size=100)  # Random noise
    y = X @ beta + epsilon  # Linear model with noise

    # Test case 1: Data without intercept column
    model = QuantRegressionWrapper(alpha=0.5)  # 50th percentile (median)
    model.fit(X, y)
    predictions = model.predict(X)

    # Check that predictions have the right shape
    assert predictions.shape == (100,)

    # Test case 2: Data with intercept column already included
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    model2 = QuantRegressionWrapper(alpha=0.5)
    model2.fit(X_with_intercept, y)
    predictions2 = model2.predict(X_with_intercept)

    # Check shape and that predictions are similar in both cases
    assert predictions2.shape == (100,)
    assert np.allclose(predictions, predictions2, rtol=1e-2)


def test_quantile_lasso_different_shapes():
    """Test that QuantileLasso works with different input shapes in fit and predict."""
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.normal(0, 1, size=(100, 3))  # 100 samples, 3 features
    beta = np.array([2.5, 1.0, -0.5])  # True coefficients
    epsilon = np.random.normal(0, 0.5, size=100)  # Random noise
    y_train = X_train @ beta + epsilon  # Linear model with noise

    # Create test data with different number of samples
    X_test = np.random.normal(0, 1, size=(20, 3))  # 20 samples, same 3 features

    # Initialize and fit QuantileLasso
    quantiles = [0.1, 0.5, 0.9]  # 10th, 50th, and 90th percentiles
    lasso = QuantileLasso(alpha=0.1)
    lasso.fit(X_train, y_train, quantiles=quantiles)

    # Predict on test data with different dimensions
    predictions = lasso.predict(X_test)

    # Verify shape of predictions: (n_samples, n_quantiles)
    assert predictions.shape == (20, 3)

    # Check that predictions follow expected order (lower quantile < median < higher quantile)
    assert np.all(
        predictions[:, 0] <= predictions[:, 1]
    )  # 10th percentile <= 50th percentile
    assert np.all(
        predictions[:, 1] <= predictions[:, 2]
    )  # 50th percentile <= 90th percentile
