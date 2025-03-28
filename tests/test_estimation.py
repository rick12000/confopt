import numpy as np

from confopt.selection.estimation import (
    initialize_estimator,
    average_scores_across_folds,
)


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
