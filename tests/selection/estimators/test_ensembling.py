import pytest
import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from confopt.selection.estimators.ensembling import (
    PointEnsembleEstimator,
    QuantileEnsembleEstimator,
    calculate_quantile_error,
)
from confopt.selection.estimators.quantile_estimation import (
    QuantileGBM,
    QuantileKNN,
    GaussianProcessQuantileEstimator,
)


def test_calculate_quantile_error():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array(
        [[0.8, 1, 1.2], [1.8, 2, 2.2], [2.8, 3, 3.2], [3.8, 4, 4.2], [4.8, 5, 5.2]]
    )
    quantiles = [0.1, 0.5, 0.9]

    errors = calculate_quantile_error(y_pred, y_true, quantiles)

    assert len(errors) == len(quantiles)
    assert np.isclose(errors[1], 0.0)


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

    weights = model._compute_weights(X, y)

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
        val_predictions_by_quantile,
    ) = model._get_stacking_training_data(X, y, quantiles)

    assert len(val_indices) == len(val_targets) == len(X)
    assert len(val_predictions_by_quantile) == len(quantiles)
    for i, q_predictions in enumerate(val_predictions_by_quantile):
        assert q_predictions.shape == (len(X), 2)


@pytest.mark.parametrize(
    "weighting_strategy", ["uniform", "joint_shared", "joint_separate"]
)
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
        assert len(weights) == 2
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
        assert np.allclose(weights, np.array([0.5, 0.5]))
    elif weighting_strategy == "joint_shared":
        assert len(weights) == 2
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
    elif weighting_strategy == "joint_separate":
        assert len(weights) == len(quantiles)
        for w in weights:
            assert len(w) == 2
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
    model.quantile_weights = np.array([0.5, 0.5])

    predictions = model.predict(X)
    expected = np.tile([3.0, 4.0, 5.0], (n_samples, 1))
    assert np.array_equal(predictions, expected)

    quantile_estimator1.predict.assert_called_with(X)
    quantile_estimator2.predict.assert_called_with(X)


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
        GaussianProcessQuantileEstimator(
            kernel="rbf", random_state=random_state, alpha=1e-6
        ),
    ]


def calculate_breach_percentages(y_true, y_pred, quantiles):
    breach_percentages = []
    for i, q in enumerate(quantiles):
        below_quantile = np.sum(y_true <= y_pred[:, i])
        breach_percentage = below_quantile / len(y_true)
        breach_percentages.append(breach_percentage)
    return breach_percentages


def calculate_calibration_error(y_true, y_pred, quantiles):
    breach_percentages = calculate_breach_percentages(y_true, y_pred, quantiles)
    calibration_errors = [abs(bp - q) for bp, q in zip(breach_percentages, quantiles)]
    return np.mean(calibration_errors)


def evaluate_quantile_performance(y_true, y_pred, quantiles):
    total_loss = 0.0
    for i, q in enumerate(quantiles):
        loss = mean_pinball_loss(y_true, y_pred[:, i], alpha=q)
        total_loss += loss
    return total_loss / len(quantiles)


@pytest.mark.parametrize(
    "data_fixture_name",
    [
        "linear_regression_data",
        "heteroscedastic_data",
        "diabetes_data",
    ],
)
@pytest.mark.parametrize("weighting_strategy", ["joint_shared", "joint_separate"])
@pytest.mark.parametrize("regularization_target", ["uniform", "best_component"])
def test_ensemble_outperforms_components_multiple_repetitions(
    request,
    data_fixture_name,
    weighting_strategy,
    regularization_target,
    ensemble_test_quantiles,
):
    X, y = request.getfixturevalue(data_fixture_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Standardize features to avoid penalizing scale-sensitive estimators
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_repetitions = 5
    success_threshold = 0.6 if weighting_strategy == "joint_separate" else 0.8

    pinball_wins = 0
    calibration_wins = 0

    for rep in range(n_repetitions):
        estimators = create_diverse_quantile_estimators(random_state=42 + rep)

        individual_losses = []
        individual_calibrations = []
        for estimator in estimators:
            estimator.fit(X_train, y_train, quantiles=ensemble_test_quantiles)
            y_pred_individual = estimator.predict(X_test)
            loss = evaluate_quantile_performance(
                y_test, y_pred_individual, ensemble_test_quantiles
            )
            calibration = calculate_calibration_error(
                y_test, y_pred_individual, ensemble_test_quantiles
            )
            individual_losses.append(loss)
            individual_calibrations.append(calibration)

        best_individual_loss = min(individual_losses)
        best_individual_calibration_error = min(individual_calibrations)

        ensemble = QuantileEnsembleEstimator(
            estimators=create_diverse_quantile_estimators(random_state=42 + rep),
            cv=5,  # More folds for better stability
            weighting_strategy=weighting_strategy,
            regularization_target=regularization_target,
            random_state=42 + rep,
            alpha=0.1,
        )

        ensemble.fit(X_train, y_train, quantiles=ensemble_test_quantiles)
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_loss = evaluate_quantile_performance(
            y_test, y_pred_ensemble, ensemble_test_quantiles
        )
        ensemble_calibration_error = calculate_calibration_error(
            y_test, y_pred_ensemble, ensemble_test_quantiles
        )

        if ensemble_loss <= best_individual_loss:
            pinball_wins += 1
        if ensemble_calibration_error <= best_individual_calibration_error:
            calibration_wins += 1

        # Monotonicity not enforced for multi-fit ensemble models (design choice)

        assert ensemble_calibration_error <= 0.4

    pinball_success_rate = pinball_wins / n_repetitions
    calibration_success_rate = calibration_wins / n_repetitions

    assert pinball_success_rate >= success_threshold
    assert calibration_success_rate >= success_threshold
