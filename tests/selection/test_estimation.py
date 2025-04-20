import numpy as np
import pytest

from confopt.selection.estimation import (
    initialize_estimator,
    average_scores_across_folds,
    PointTuner,
    QuantileTuner,
)
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from sklearn.model_selection import train_test_split

from confopt.selection.estimator_configuration import ESTIMATOR_REGISTRY


def test_initialize_estimator_with_params():
    estimator = initialize_estimator(
        estimator_architecture="gbm",
        initialization_params={"random_state": 42},
        random_state=42,
    )
    assert estimator.random_state == 42


def test_average_scores_across_folds_duplicates():
    configs = [
        {"param_1": 1, "param_2": "a"},
        {"param_1": 1, "param_2": "a"},
        {"param_1": 2, "param_2": "b"},
        {"param_1": 3, "param_2": "c"},
        {"param_1": 3, "param_2": "c"},
    ]
    scores = [0.5, 0.3, 0.7, 0.2, 0.9]

    unique_configs, unique_scores = average_scores_across_folds(configs, scores)
    assert len(unique_configs) == 3

    expected_scores = [0.4, 0.7, 0.55]
    assert np.allclose(unique_scores, expected_scores)


def evaluate_point_model(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)


def evaluate_quantile_model(
    model, X_val: np.ndarray, y_val: np.ndarray, quantiles: list
) -> float:
    preds = model.predict(X_val)
    scores = []
    for i, q in enumerate(quantiles):
        scores.append(mean_pinball_loss(y_val, preds[:, i], alpha=q))
    return sum(scores) / len(scores)


def setup_test_data(seed=42):
    np.random.seed(seed)
    X = np.random.rand(50, 5)
    y = X.sum(axis=1) + np.random.normal(0, 0.1, 50)
    return train_test_split(X, y, test_size=0.25, random_state=seed)


def create_and_evaluate_point_model(
    estimator_architecture, params, X_train, y_train, X_val, y_val
):
    model = initialize_estimator(
        estimator_architecture, initialization_params=params, random_state=42
    )
    model.fit(X_train, y_train)
    error = evaluate_point_model(model, X_val, y_val)
    return model, error


def create_and_evaluate_quantile_model(
    estimator_architecture, params, X_train, y_train, X_val, y_val, quantiles
):
    model = initialize_estimator(
        estimator_architecture, initialization_params=params, random_state=42
    )
    model.fit(X_train, y_train, quantiles=quantiles)
    error = evaluate_quantile_model(model, X_val, y_val, quantiles)
    return model, error


def get_default_parameters(estimator_architecture):
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]
    default_estimator = initialize_estimator(estimator_architecture, random_state=42)
    return {
        param: getattr(default_estimator, param)
        for param in estimator_config.estimator_parameter_space.keys()
        if hasattr(default_estimator, param)
    }


def setup_point_tuner():
    return PointTuner(random_state=42)


def setup_quantile_tuner():
    quantiles = [0.1, 0.9]
    return QuantileTuner(quantiles=quantiles, random_state=42), quantiles


@pytest.mark.parametrize("split_type", ["k_fold", "ordinal_split"])
def test_random_tuner_better_than_default(split_type):
    results_for_this_split = []

    for estimator_architecture in list(ESTIMATOR_REGISTRY.keys()):
        X_train, X_val, y_train, y_val = setup_test_data()
        estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]
        default_params = get_default_parameters(estimator_architecture) or {}

        if estimator_config.is_quantile_estimator():
            tuner, quantiles = setup_quantile_tuner()

            _, baseline_error = create_and_evaluate_quantile_model(
                estimator_architecture,
                default_params,
                X_train,
                y_train,
                X_val,
                y_val,
                quantiles,
            )

            best_config = tuner.tune(
                X_train,
                y_train,
                estimator_architecture,
                n_searches=3,
                train_split=0.5,
                split_type=split_type,
                forced_param_configurations=[default_params] if default_params else [],
            )

            _, tuned_error = create_and_evaluate_quantile_model(
                estimator_architecture,
                best_config,
                X_train,
                y_train,
                X_val,
                y_val,
                quantiles,
            )
        else:
            tuner = setup_point_tuner()

            _, baseline_error = create_and_evaluate_point_model(
                estimator_architecture, default_params, X_train, y_train, X_val, y_val
            )

            best_config = tuner.tune(
                X_train,
                y_train,
                estimator_architecture,
                n_searches=5,
                train_split=0.5,
                split_type=split_type,
                forced_param_configurations=[default_params] if default_params else [],
            )

            _, tuned_error = create_and_evaluate_point_model(
                estimator_architecture, best_config, X_train, y_train, X_val, y_val
            )

        results_for_this_split.append(tuned_error <= baseline_error)

    assert len(results_for_this_split) > 0

    success_rate = np.mean(results_for_this_split)
    assert success_rate > 0.5


@pytest.mark.parametrize("split_type", ["k_fold", "ordinal_split"])
@pytest.mark.parametrize("estimator_architecture", list(ESTIMATOR_REGISTRY.keys()))
def test_tuning_with_default_params_matches_baseline(
    estimator_architecture, split_type
):
    X_train, X_val, y_train, y_val = setup_test_data(seed=42)
    estimator_config = ESTIMATOR_REGISTRY[estimator_architecture]
    default_params = estimator_config.default_params

    if estimator_config.is_quantile_estimator():
        tuner, quantiles = setup_quantile_tuner()

        _, baseline_error = create_and_evaluate_quantile_model(
            estimator_architecture,
            default_params,
            X_train,
            y_train,
            X_val,
            y_val,
            quantiles,
        )

        best_config = tuner.tune(
            X_train,
            y_train,
            estimator_architecture,
            n_searches=1,
            train_split=0.5,
            split_type=split_type,
            forced_param_configurations=[default_params],
        )

        assert best_config == default_params

        _, tuned_error = create_and_evaluate_quantile_model(
            estimator_architecture,
            best_config,
            X_train,
            y_train,
            X_val,
            y_val,
            quantiles,
        )
    else:
        tuner = setup_point_tuner()

        _, baseline_error = create_and_evaluate_point_model(
            estimator_architecture, default_params, X_train, y_train, X_val, y_val
        )

        best_config = tuner.tune(
            X_train,
            y_train,
            estimator_architecture,
            n_searches=1,
            train_split=0.5,
            split_type=split_type,
            forced_param_configurations=[default_params],
        )

        assert best_config == default_params

        _, tuned_error = create_and_evaluate_point_model(
            estimator_architecture, best_config, X_train, y_train, X_val, y_val
        )

    assert np.isclose(tuned_error, baseline_error, atol=1e-5)
