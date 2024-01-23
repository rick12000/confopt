from typing import Dict

import numpy as np
import pytest

from confopt.config import GBM_NAME, RF_NAME, QGBM_NAME, QRF_NAME
from confopt.estimation import (
    QuantileConformalRegression,
    LocallyWeightedConformalRegression,
    initialize_point_estimator,
    initialize_quantile_estimator,
    cross_validate_configurations,
)

DEFAULT_SEED = 1234


def get_discretized_quantile_dict(
    X: np.array, y: np.array, quantile_level: float
) -> Dict:
    """
    Helper function to obtain dictionary of quantiles per X value.

    Parameters
    ----------
    X :
        Explanatory variables.
    y :
        Target variable.
    quantile_level :
        Desired quantile to take.

    Returns
    -------
    quantile_dict :
        Dictionary relating X values to their quantile.
    """
    quantile_dict = {}
    for discrete_x_coordinate in np.unique(X):
        conditional_y_at_x = y[X == discrete_x_coordinate]
        quantile_dict[discrete_x_coordinate] = np.quantile(
            conditional_y_at_x, quantile_level
        )
    return quantile_dict


def test_initialize_point_estimator():
    dummy_estimator_name = GBM_NAME

    initialized_model = initialize_point_estimator(
        estimator_architecture=dummy_estimator_name,
        initialization_params={},
        random_state=DEFAULT_SEED,
    )

    assert hasattr(initialized_model, "predict")


def test_initialize_point_estimator__reproducibility():
    dummy_estimator_name = GBM_NAME

    initialized_model_first_call = initialize_point_estimator(
        estimator_architecture=dummy_estimator_name,
        initialization_params={},
        random_state=DEFAULT_SEED,
    )
    initialized_model_second_call = initialize_point_estimator(
        estimator_architecture=dummy_estimator_name,
        initialization_params={},
        random_state=DEFAULT_SEED,
    )
    assert (
        initialized_model_first_call.random_state
        == initialized_model_second_call.random_state
    )


def test_initialize_quantile_estimator():
    dummy_estimator_name = QRF_NAME
    dummy_pinball_loss_alpha = [0.25, 0.75]

    initialized_model = initialize_quantile_estimator(
        estimator_architecture=dummy_estimator_name,
        initialization_params={},
        pinball_loss_alpha=dummy_pinball_loss_alpha,
        random_state=DEFAULT_SEED,
    )

    assert hasattr(initialized_model, "predict")


def test_initialize_quantile_estimator__reproducibility():
    dummy_estimator_name = QRF_NAME
    dummy_pinball_loss_alpha = [0.25, 0.75]

    initialized_model_first_call = initialize_quantile_estimator(
        estimator_architecture=dummy_estimator_name,
        initialization_params={},
        pinball_loss_alpha=dummy_pinball_loss_alpha,
        random_state=DEFAULT_SEED,
    )
    initialized_model_second_call = initialize_quantile_estimator(
        estimator_architecture=dummy_estimator_name,
        initialization_params={},
        pinball_loss_alpha=dummy_pinball_loss_alpha,
        random_state=DEFAULT_SEED,
    )
    assert (
        initialized_model_first_call.random_state
        == initialized_model_second_call.random_state
    )


def test_cross_validate_estimator_configurations__point_estimator_mse(
    dummy_gbm_configurations, dummy_stationary_gaussian_dataset
):
    estimator_type = GBM_NAME
    X, y = (
        dummy_stationary_gaussian_dataset[:, 0].reshape(-1, 1),
        dummy_stationary_gaussian_dataset[:, 1],
    )
    scored_configurations, scores = cross_validate_configurations(
        configurations=dummy_gbm_configurations,
        estimator_architecture=estimator_type,
        X=X,
        y=y,
        k_fold_splits=3,
        quantiles=None,
        random_state=DEFAULT_SEED,
    )

    assert len(scored_configurations) == len(scores)
    assert len(scored_configurations) == len(dummy_gbm_configurations)

    stringified_scored_configurations = []
    for configuration in scored_configurations:
        stringified_scored_configurations.append(
            str(dict(sorted(configuration.items())))
        )
    assert sorted(list(set(stringified_scored_configurations))) == sorted(
        stringified_scored_configurations
    )

    for score in scores:
        assert score >= 0


def test_cross_validate_estimator_configurations__point_estimator_mse__reproducibility(
    dummy_gbm_configurations, dummy_stationary_gaussian_dataset
):
    estimator_type = GBM_NAME
    X, y = (
        dummy_stationary_gaussian_dataset[:, 0].reshape(-1, 1),
        dummy_stationary_gaussian_dataset[:, 1],
    )

    (
        scored_configurations_first_call,
        scores_first_call,
    ) = cross_validate_configurations(
        configurations=dummy_gbm_configurations,
        estimator_architecture=estimator_type,
        X=X,
        y=y,
        k_fold_splits=3,
        quantiles=None,
        random_state=DEFAULT_SEED,
    )
    (
        scored_configurations_second_call,
        scores_second_call,
    ) = cross_validate_configurations(
        configurations=dummy_gbm_configurations,
        estimator_architecture=estimator_type,
        X=X,
        y=y,
        k_fold_splits=3,
        quantiles=None,
        random_state=DEFAULT_SEED,
    )

    assert scored_configurations_first_call == scored_configurations_second_call
    assert scores_first_call == scores_second_call


@pytest.mark.parametrize("confidence_level", [0.2, 0.8])
@pytest.mark.parametrize("tuning_param_combinations", [0, 1, 3])
@pytest.mark.parametrize("quantile_estimator_architecture", [QGBM_NAME, QRF_NAME])
def test_quantile_regression_tune_fit(
    dummy_fixed_quantile_dataset,
    confidence_level,
    tuning_param_combinations,
    quantile_estimator_architecture,
):
    random_state = DEFAULT_SEED
    X, y = (
        dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
        dummy_fixed_quantile_dataset[:, 1],
    )
    train_split = 0.8
    X_train, y_train = (
        X[: round(len(X) * train_split), :],
        y[: round(len(y) * train_split)],
    )
    X_val, y_val = X[round(len(X) * train_split) :, :], y[round(len(y) * train_split) :]

    qcr = QuantileConformalRegression(
        quantile_estimator_architecture=quantile_estimator_architecture,
    )
    qcr.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        confidence_level=confidence_level,
        tuning_iterations=tuning_param_combinations,
        random_state=random_state,
    )

    assert qcr.nonconformity_scores is not None
    assert qcr.quantile_estimator is not None


@pytest.mark.parametrize("confidence_level", [0.2, 0.8])
@pytest.mark.parametrize("tuning_param_combinations", [5])
@pytest.mark.parametrize("quantile_estimator_architecture", [QGBM_NAME, QRF_NAME])
def test_quantile_regression_predict(
    dummy_fixed_quantile_dataset,
    confidence_level,
    tuning_param_combinations,
    quantile_estimator_architecture,
):
    random_state = DEFAULT_SEED
    X, y = (
        dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
        dummy_fixed_quantile_dataset[:, 1],
    )
    train_split = 0.8
    X_train, y_train = (
        X[: round(len(X) * train_split), :],
        y[: round(len(y) * train_split)],
    )
    X_val, y_val = X[round(len(X) * train_split) :, :], y[round(len(y) * train_split) :]
    qcr = QuantileConformalRegression(
        quantile_estimator_architecture=quantile_estimator_architecture,
    )
    qcr.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        confidence_level=confidence_level,
        tuning_iterations=tuning_param_combinations,
        random_state=random_state,
    )

    y_low_bounds, y_high_bounds = qcr.predict(X_val, confidence_level=confidence_level)

    # Check lower bound is always lower than higher bound:
    for y_low, y_high in zip(y_low_bounds, y_high_bounds):
        assert y_low <= y_high

    # Compute observed quantiles per X slice during training (only works for univariate dummy datasets):
    low_quantile_dict_train = get_discretized_quantile_dict(
        X_train.reshape(
            -1,
        ),
        y_train,
        confidence_level + ((1 - confidence_level) / 2),
    )
    high_quantile_dict_train = get_discretized_quantile_dict(
        X_train.reshape(
            -1,
        ),
        y_train,
        (1 - confidence_level) / 2,
    )

    # Check that predictions return observed quantiles during training:
    y_error_margin = 1
    margin_breach_tolerance = 0.3
    low_margin_breaches, high_margin_breaches = 0, 0
    for x_obs, y_low, y_high in zip(
        X_train.reshape(
            -1,
        ),
        y_low_bounds,
        y_high_bounds,
    ):
        if abs(y_low - low_quantile_dict_train[x_obs]) > y_error_margin:
            low_margin_breaches += 1
        if abs(y_high - high_quantile_dict_train[x_obs]) > y_error_margin:
            high_margin_breaches += 1
    assert low_margin_breaches < len(X_train) * margin_breach_tolerance
    assert high_margin_breaches < len(X_train) * margin_breach_tolerance

    # Collect breach events on validation data (this is not unseen data for qcr, as it's used to calibrate conformal
    # intervals, this is a basic check of function to see if calibration is correct, but doesn't check actual out of
    # sample coverage):
    interval_breach_states = []
    for y_obs, y_low, y_high in zip(y_val, y_low_bounds, y_high_bounds):
        is_interval_breach = 0 if y_high > y_obs > y_low else 1
        interval_breach_states.append(is_interval_breach)

    interval_breach_rate = sum(interval_breach_states) / len(interval_breach_states)
    breach_margin = 0.01
    assert (
        (confidence_level - breach_margin)
        <= (1 - interval_breach_rate)
        <= (confidence_level + breach_margin)
    )


@pytest.mark.parametrize("confidence_level", [0.2, 0.8])
@pytest.mark.parametrize("tuning_param_combinations", [0, 1, 3])
@pytest.mark.parametrize("point_estimator_architecture", [GBM_NAME, RF_NAME])
@pytest.mark.parametrize("demeaning_estimator_architecture", [GBM_NAME])
@pytest.mark.parametrize("variance_estimator_architecture", [GBM_NAME])
def test_locally_weighted_regression_tune_fit(
    dummy_fixed_quantile_dataset,
    confidence_level,
    tuning_param_combinations,
    point_estimator_architecture,
    demeaning_estimator_architecture,
    variance_estimator_architecture,
):
    random_state = DEFAULT_SEED
    X, y = (
        dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
        dummy_fixed_quantile_dataset[:, 1],
    )
    train_split = 0.8
    X_train, y_train = (
        X[: round(len(X) * train_split), :],
        y[: round(len(y) * train_split)],
    )
    pe_split = 0.8
    X_pe, y_pe = (
        X_train[: round(len(X_train) * pe_split), :],
        y_train[: round(len(y_train) * pe_split)],
    )
    X_ve, y_ve = (
        X_train[round(len(X_train) * pe_split) :, :],
        y_train[round(len(y_train) * pe_split) :],
    )
    X_val, y_val = X[round(len(X) * train_split) :, :], y[round(len(y) * train_split) :]

    lwcr = LocallyWeightedConformalRegression(
        point_estimator_architecture=point_estimator_architecture,
        demeaning_estimator_architecture=demeaning_estimator_architecture,
        variance_estimator_architecture=variance_estimator_architecture,
    )
    lwcr.fit(
        X_pe=X_pe,
        y_pe=y_pe,
        X_ve=X_ve,
        y_ve=y_ve,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=tuning_param_combinations,
        random_state=random_state,
    )

    assert lwcr.nonconformity_scores is not None
    assert lwcr.pe_estimator is not None
    assert lwcr.ve_estimator is not None


@pytest.mark.parametrize("confidence_level", [0.2, 0.8])
@pytest.mark.parametrize("tuning_param_combinations", [5])
@pytest.mark.parametrize("point_estimator_architecture", [GBM_NAME, RF_NAME])
@pytest.mark.parametrize("demeaning_estimator_architecture", [GBM_NAME])
@pytest.mark.parametrize("variance_estimator_architecture", [GBM_NAME])
def test_locally_weighted_regression_predict(
    dummy_fixed_quantile_dataset,
    confidence_level,
    tuning_param_combinations,
    point_estimator_architecture,
    demeaning_estimator_architecture,
    variance_estimator_architecture,
):
    random_state = DEFAULT_SEED
    X, y = (
        dummy_fixed_quantile_dataset[:, 0].reshape(-1, 1),
        dummy_fixed_quantile_dataset[:, 1],
    )
    train_split = 0.8
    X_train, y_train = (
        X[: round(len(X) * train_split), :],
        y[: round(len(y) * train_split)],
    )
    pe_split = 0.8
    X_pe, y_pe = (
        X_train[: round(len(X_train) * pe_split), :],
        y_train[: round(len(y_train) * pe_split)],
    )
    X_ve, y_ve = (
        X_train[round(len(X_train) * pe_split) :, :],
        y_train[round(len(y_train) * pe_split) :],
    )
    X_val, y_val = X[round(len(X) * train_split) :, :], y[round(len(y) * train_split) :]

    lwcr = LocallyWeightedConformalRegression(
        point_estimator_architecture=point_estimator_architecture,
        demeaning_estimator_architecture=demeaning_estimator_architecture,
        variance_estimator_architecture=variance_estimator_architecture,
    )
    lwcr.fit(
        X_pe=X_pe,
        y_pe=y_pe,
        X_ve=X_ve,
        y_ve=y_ve,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=tuning_param_combinations,
        random_state=random_state,
    )

    y_low_bounds, y_high_bounds = lwcr.predict(X_val, confidence_level=confidence_level)

    # Check lower bound is always lower than higher bound:
    for y_low, y_high in zip(y_low_bounds, y_high_bounds):
        assert y_low <= y_high

    # Compute observed quantiles per X slice during training (only works for univariate dummy datasets):
    low_quantile_dict_train = get_discretized_quantile_dict(
        X_train.reshape(
            -1,
        ),
        y_train,
        confidence_level + ((1 - confidence_level) / 2),
    )
    high_quantile_dict_train = get_discretized_quantile_dict(
        X_train.reshape(
            -1,
        ),
        y_train,
        (1 - confidence_level) / 2,
    )

    # Check that predictions return observed quantiles during training:
    y_error_margin = 1
    margin_breach_tolerance = 0.3
    low_margin_breaches, high_margin_breaches = 0, 0
    for x_obs, y_low, y_high in zip(
        X_train.reshape(
            -1,
        ),
        y_low_bounds,
        y_high_bounds,
    ):
        if abs(y_low - low_quantile_dict_train[x_obs]) > y_error_margin:
            low_margin_breaches += 1
        if abs(y_high - high_quantile_dict_train[x_obs]) > y_error_margin:
            high_margin_breaches += 1
    assert low_margin_breaches < len(X_train) * margin_breach_tolerance
    assert high_margin_breaches < len(X_train) * margin_breach_tolerance

    # Collect breach events on validation data (this is not unseen data for lwcr, as it's used to calibrate conformal
    # intervals, this is a basic check of function to see if calibration is correct, but doesn't check actual out of
    # sample coverage):
    interval_breach_states = []
    for y_obs, y_low, y_high in zip(y_val, y_low_bounds, y_high_bounds):
        is_interval_breach = 0 if y_high > y_obs > y_low else 1
        interval_breach_states.append(is_interval_breach)

    interval_breach_rate = sum(interval_breach_states) / len(interval_breach_states)
    breach_margin = 0.01
    assert (
        (confidence_level - breach_margin)
        <= (1 - interval_breach_rate)
        <= (confidence_level + breach_margin)
    )
