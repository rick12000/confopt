import pytest
import numpy as np

from confopt.selection.estimators.ensembling import (
    PointEnsembleEstimator,
    QuantileEnsembleEstimator,
    calculate_quantile_error,
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


class TestPointEnsembleEstimator:
    def test_get_stacking_training_data(self, toy_dataset, estimator1, estimator2):
        X, y = toy_dataset

        model = PointEnsembleEstimator(
            estimators=[estimator1, estimator2], cv=2, random_state=42
        )

        val_indices, val_targets, val_predictions = model._get_stacking_training_data(
            X, y
        )

        assert len(np.unique(val_indices)) == len(X)

        assert val_predictions.shape == (len(X), 2)

        assert np.array_equal(val_targets, y[val_indices])

    @pytest.mark.parametrize("weighting_strategy", ["uniform", "linear_stack"])
    def test_compute_weights(
        self, toy_dataset, estimator1, competing_estimator, weighting_strategy
    ):
        X, y = toy_dataset

        model = PointEnsembleEstimator(
            estimators=[estimator1, competing_estimator],
            cv=2,
            weighting_strategy=weighting_strategy,
            random_state=42,
        )

        weights = model._compute_weights(X, y)

        assert len(weights) == 2
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)

        if weighting_strategy == "uniform":
            assert np.allclose(weights, np.array([0.5, 0.5]))

    def test_predict_with_uniform_weights(self, toy_dataset, estimator1, estimator2):
        X, _ = toy_dataset

        model = PointEnsembleEstimator(
            estimators=[estimator1, estimator2],
            weighting_strategy="uniform",
        )
        model.weights = np.array([0.5, 0.5])

        predictions = model.predict(X)

        expected = np.array([3, 5, 7, 9])

        assert predictions[0] == 3
        assert predictions[-1] == 9

        assert np.array_equal(predictions, expected)


class TestQuantileEnsembleEstimator:
    def test_get_stacking_training_data(
        self, toy_dataset, quantiles, quantile_estimator1, quantile_estimator2
    ):
        X, y = toy_dataset

        model = QuantileEnsembleEstimator(
            estimators=[quantile_estimator1, quantile_estimator2], cv=2, random_state=42
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

    @pytest.mark.parametrize("weighting_strategy", ["uniform", "linear_stack"])
    def test_compute_quantile_weights(
        self,
        toy_dataset,
        quantiles,
        quantile_estimator1,
        quantile_estimator2,
        weighting_strategy,
    ):
        X, y = toy_dataset

        model_uniform = QuantileEnsembleEstimator(
            estimators=[quantile_estimator1, quantile_estimator2],
            cv=2,
            weighting_strategy=weighting_strategy,
            random_state=42,
        )

        weights = model_uniform._compute_quantile_weights(X, y, quantiles)

        assert len(weights) == len(quantiles)

        for w in weights:
            assert len(w) == 2
            assert np.isclose(np.sum(w), 1.0)
            assert np.all(w >= 0)
            if weighting_strategy == "uniform":
                assert np.allclose(w, np.array([0.5, 0.5]))

    def test_predict_quantiles(
        self, toy_dataset, quantiles, quantile_estimator1, quantile_estimator2
    ):
        X, _ = toy_dataset
        n_samples = len(X)

        model = QuantileEnsembleEstimator(
            estimators=[quantile_estimator1, quantile_estimator2],
            weighting_strategy="uniform",
        )
        model.quantiles = quantiles
        model.quantile_weights = [np.array([0.5, 0.5]) for _ in quantiles]

        predictions = model.predict(X)

        # Expected values: average of quantile_estimator1 and quantile_estimator2
        # For each quantile:
        # q0.1: (2 + 4) / 2 = 3
        # q0.5: (4 + 4) / 2 = 4
        # q0.9: (6 + 4) / 2 = 5
        expected = np.tile([3.0, 4.0, 5.0], (n_samples, 1))
        assert np.array_equal(predictions, expected)

        quantile_estimator1.predict.assert_called_with(X)
        quantile_estimator2.predict.assert_called_with(X)
