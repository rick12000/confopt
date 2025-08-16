import pytest
import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from confopt.selection.estimators.ensembling import (
    PointEnsembleEstimator,
    QuantileEnsembleEstimator,
)
from confopt.selection.estimators.quantile_estimation import (
    QuantileGBM,
    QuantileKNN,
    QuantileLasso,
)


def create_diverse_quantile_estimators(random_state=42):
    return [
        QuantileGBM(
            learning_rate=0.1,
            n_estimators=50,
            min_samples_split=10,
            min_samples_leaf=5,
            max_depth=3,
            random_state=random_state,
        ),
        QuantileKNN(n_neighbors=15),
        QuantileLasso(
            max_iter=1000,
            p_tol=1e-6,
            random_state=random_state,
        ),
    ]


def create_diverse_point_estimators(random_state=42):
    return [
        LinearRegression(),
        RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=random_state,
        ),
    ]


def evaluate_quantile_performance(y_true, y_pred, quantiles):
    total_loss = 0.0
    for i, q in enumerate(quantiles):
        loss = mean_pinball_loss(y_true, y_pred[:, i], alpha=q)
        total_loss += loss
    return total_loss / len(quantiles)


def evaluate_point_performance(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def test_point_ensemble_get_stacking_training_data(toy_dataset, estimator1, estimator2):
    X, y = toy_dataset

    model = PointEnsembleEstimator(
        estimators=[estimator1, estimator2], cv=2, random_state=42, alpha=0.01
    )

    val_indices, val_targets, val_predictions = model._get_stacking_training_data(X, y)

    assert len(np.unique(val_indices)) == len(X)
    assert val_predictions.shape == (len(X), 2)
    assert np.array_equal(val_targets, y[val_indices])


@pytest.mark.parametrize("weighting_strategy", ["uniform", "linear_stack"])
def test_point_ensemble_compute_weights(
    toy_dataset, estimator1, competing_estimator, weighting_strategy
):
    X, y = toy_dataset

    model = PointEnsembleEstimator(
        estimators=[estimator1, competing_estimator],
        cv=2,
        weighting_strategy=weighting_strategy,
        random_state=42,
        alpha=0.01,
    )

    weights = model._compute_point_weights(X, y)

    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0)

    if weighting_strategy == "uniform":
        assert np.allclose(weights, np.array([0.5, 0.5]))


def test_point_ensemble_predict_with_uniform_weights(
    toy_dataset, estimator1, estimator2
):
    X, _ = toy_dataset

    model = PointEnsembleEstimator(
        estimators=[estimator1, estimator2],
        weighting_strategy="uniform",
        alpha=0.01,
    )
    model.weights = np.array([0.5, 0.5])

    predictions = model.predict(X)
    expected = np.array([3, 5, 7, 9])

    assert predictions[0] == 3
    assert predictions[-1] == 9
    assert np.array_equal(predictions, expected)


def test_quantile_ensemble_get_stacking_training_data(
    toy_dataset, quantiles, quantile_estimator1, quantile_estimator2
):
    X, y = toy_dataset

    model = QuantileEnsembleEstimator(
        estimators=[quantile_estimator1, quantile_estimator2],
        cv=2,
        random_state=42,
        alpha=0.01,
    )

    (
        val_indices,
        val_targets,
        val_predictions,
    ) = model._get_stacking_training_data(X, y, quantiles)

    assert len(val_indices) == len(val_targets) == len(X)
    assert val_predictions.shape[0] == len(X)
    assert val_predictions.shape[1] == 2 * len(quantiles)


@pytest.mark.parametrize("weighting_strategy", ["uniform", "linear_stack"])
def test_quantile_ensemble_compute_quantile_weights(
    toy_dataset,
    quantiles,
    quantile_estimator1,
    quantile_estimator2,
    weighting_strategy,
):
    X, y = toy_dataset

    model = QuantileEnsembleEstimator(
        estimators=[quantile_estimator1, quantile_estimator2],
        cv=2,
        weighting_strategy=weighting_strategy,
        random_state=42,
        alpha=0.01,
    )

    weights = model._compute_quantile_weights(X, y, quantiles)

    if weighting_strategy == "uniform":
        assert weights.shape == (len(quantiles), 2)
        for w in weights:
            assert np.isclose(np.sum(w), 1.0)
            assert np.all(w >= 0)
    elif weighting_strategy == "linear_stack":
        assert weights.shape == (len(quantiles), 2)
        for w in weights:
            assert np.isclose(np.sum(w), 1.0)
            assert np.all(w >= 0)


def test_quantile_ensemble_predict_quantiles(
    toy_dataset, quantiles, quantile_estimator1, quantile_estimator2
):
    X, _ = toy_dataset
    n_samples = len(X)

    model = QuantileEnsembleEstimator(
        estimators=[quantile_estimator1, quantile_estimator2],
        weighting_strategy="uniform",
        alpha=0.01,
    )
    model.quantiles = quantiles
    model.quantile_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    predictions = model.predict(X)
    expected = np.tile([3.0, 4.0, 5.0], (n_samples, 1))
    assert np.array_equal(predictions, expected)

    quantile_estimator1.predict.assert_called_with(X)
    quantile_estimator2.predict.assert_called_with(X)


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_fixture_name",
    [
        "heteroscedastic_data",
        "diabetes_data",
    ],
)
@pytest.mark.parametrize("weighting_strategy", ["linear_stack"])
def test_ensemble_outperforms_components_multiple_repetitions(
    request,
    data_fixture_name,
    weighting_strategy,
    ensemble_test_quantiles,
):
    X, y = request.getfixturevalue(data_fixture_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_repetitions = 10
    success_threshold = 0.51

    pinball_wins = 0

    for rep in range(n_repetitions):
        estimators = create_diverse_quantile_estimators(random_state=42 + rep)

        individual_losses = []
        for estimator in estimators:
            estimator.fit(X_train, y_train, quantiles=ensemble_test_quantiles)
            y_pred_individual = estimator.predict(X_test)
            loss = evaluate_quantile_performance(
                y_test, y_pred_individual, ensemble_test_quantiles
            )
            individual_losses.append(loss)

        best_individual_loss = min(individual_losses)

        ensemble = QuantileEnsembleEstimator(
            estimators=create_diverse_quantile_estimators(random_state=42 + rep),
            cv=5,
            weighting_strategy=weighting_strategy,
            random_state=42 + rep,
            alpha=0.1,
        )

        ensemble.fit(X_train, y_train, quantiles=ensemble_test_quantiles)
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_loss = evaluate_quantile_performance(
            y_test, y_pred_ensemble, ensemble_test_quantiles
        )

        if ensemble_loss <= best_individual_loss:
            pinball_wins += 1

    pinball_success_rate = pinball_wins / n_repetitions
    assert pinball_success_rate > success_threshold


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_fixture_name",
    [
        "heteroscedastic_data",
        "diabetes_data",
    ],
)
@pytest.mark.parametrize("weighting_strategy", ["linear_stack"])
def test_point_ensemble_outperforms_components_multiple_repetitions(
    request,
    data_fixture_name,
    weighting_strategy,
):
    X, y = request.getfixturevalue(data_fixture_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_repetitions = 10
    success_threshold = 0.51

    mse_wins = 0

    for rep in range(n_repetitions):
        estimators = create_diverse_point_estimators(random_state=42 + rep)

        individual_losses = []
        for estimator in estimators:
            estimator.fit(X_train, y_train)
            y_pred_individual = estimator.predict(X_test)
            loss = evaluate_point_performance(y_test, y_pred_individual)
            individual_losses.append(loss)

        best_individual_loss = min(individual_losses)

        ensemble = PointEnsembleEstimator(
            estimators=create_diverse_point_estimators(random_state=42 + rep),
            cv=5,
            weighting_strategy=weighting_strategy,
            random_state=42 + rep,
            alpha=0.1,
        )

        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_loss = evaluate_point_performance(y_test, y_pred_ensemble)

        if ensemble_loss <= best_individual_loss:
            mse_wins += 1

    mse_success_rate = mse_wins / n_repetitions
    assert mse_success_rate > success_threshold
