import pytest
import numpy as np
from unittest.mock import patch
from confopt.selection.acquisition import (
    calculate_ucb_predictions,
    calculate_thompson_predictions,
)


def test_calculate_ucb_predictions():
    lower_bound = np.array([0.5, 0.7, 0.3, 0.9])
    interval_width = np.array([0.2, 0.1, 0.3, 0.05])
    beta = 0.5

    result = calculate_ucb_predictions(
        lower_bound=lower_bound, interval_width=interval_width, beta=beta
    )
    expected = np.array([0.4, 0.65, 0.15, 0.875])

    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "enable_optimistic, point_predictions",
    [(False, None), (True, np.array([0.05, 0.35, 0.75, 0.25, 0.95]))],
)
def test_calculate_thompson_predictions(
    conformal_bounds, enable_optimistic, point_predictions
):
    fixed_indices = np.array([0, 1, 2, 0, 1])

    with patch.object(np.random, "choice", return_value=fixed_indices):
        result = calculate_thompson_predictions(
            predictions_per_interval=conformal_bounds,
            enable_optimistic_sampling=enable_optimistic,
            point_predictions=point_predictions,
        )

    lower_bounds = np.array(
        [
            conformal_bounds[0].lower_bounds[0],
            conformal_bounds[1].lower_bounds[1],
            conformal_bounds[2].lower_bounds[2],
            conformal_bounds[0].lower_bounds[3],
            conformal_bounds[1].lower_bounds[4],
        ]
    )

    if enable_optimistic:
        expected = np.minimum(lower_bounds, point_predictions)
    else:
        expected = lower_bounds

    np.testing.assert_array_almost_equal(result, expected)
