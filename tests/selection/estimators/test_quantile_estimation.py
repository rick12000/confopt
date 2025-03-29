import pytest
import numpy as np
from confopt.selection.estimators.quantile_estimation import (
    QuantileLasso,
    QuantileGBM,
    QuantileLightGBM,
    QuantileForest,
    QuantileKNN,
)

MODEL_CONFIGS = [
    (QuantileLasso, {}),
    (
        QuantileGBM,
        {
            "learning_rate": 0.1,
            "n_estimators": 200,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_depth": 4,
        },
    ),
    (QuantileLightGBM, {"learning_rate": 0.1, "n_estimators": 100}),
    (QuantileForest, {"n_estimators": 200, "max_depth": None, "random_state": 42}),
    (QuantileKNN, {"n_neighbors": 50}),
]


@pytest.fixture
def uniform_feature_data():
    np.random.seed(42)
    n_samples_train = 10000
    n_features = 3

    X_train = np.random.uniform(-1, 1, size=(n_samples_train, n_features))
    y_train = np.random.uniform(0, 1, size=n_samples_train)

    grid_points = np.linspace(-1, 1, 20)
    x1, x2, x3 = np.meshgrid(grid_points, grid_points, grid_points)
    X_test = np.column_stack([x1.flatten(), x2.flatten(), x3.flatten()])

    quantiles = [0.1, 0.9]
    expected_quantiles = {q: q for q in quantiles}

    return X_train, y_train, X_test, expected_quantiles


@pytest.mark.parametrize("model_class, model_params", MODEL_CONFIGS)
def test_predict(uniform_feature_data, model_class, model_params):
    X_train, y_train, X_test, expected_quantiles = uniform_feature_data
    quantiles = [0.1, 0.9]

    model = model_class(**model_params)
    model.fit(X_train, y_train, quantiles=quantiles)

    predictions = model.predict(X_test)

    assert predictions.shape == (len(X_test), len(quantiles))

    ordering_breaches = np.sum(predictions[:, 0] > predictions[:, 1])
    ordering_breach_pct = ordering_breaches / len(X_test)
    max_ordering_breach_pct = 0.05

    assert ordering_breach_pct <= max_ordering_breach_pct
    tolerance = 0.20
    max_deviation_breach_pct = 0.15

    for i, q in enumerate(quantiles):
        deviations = np.abs(predictions[:, i] - expected_quantiles[q])
        deviation_breaches = np.sum(deviations >= tolerance)
        deviation_breach_pct = deviation_breaches / len(X_test)

        assert deviation_breach_pct <= max_deviation_breach_pct
