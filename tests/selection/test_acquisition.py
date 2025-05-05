import pytest
import numpy as np
from confopt.selection.acquisition import (
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,
)
from confopt.selection.sampling import (
    PessimisticLowerBoundSampler,
    LowerBoundSampler,
    ThompsonSampler,
    ExpectedImprovementSampler,
    InformationGainSampler,
    MaxValueEntropySearchSampler,
)
from conftest import (
    POINT_ESTIMATOR_ARCHITECTURES,
    QUANTILE_ESTIMATOR_ARCHITECTURES,
    SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES,
)


@pytest.mark.parametrize(
    "sampler_class,sampler_kwargs",
    [
        (PessimisticLowerBoundSampler, {"interval_width": 0.8}),
        (LowerBoundSampler, {"interval_width": 0.8}),
        (ThompsonSampler, {"n_quantiles": 4}),
        (ExpectedImprovementSampler, {"n_quantiles": 4}),
        (InformationGainSampler, {"n_quantiles": 4}),
        (MaxValueEntropySearchSampler, {"n_quantiles": 4}),
    ],
)
@pytest.mark.parametrize("point_arch", POINT_ESTIMATOR_ARCHITECTURES[:1])
@pytest.mark.parametrize("variance_arch", POINT_ESTIMATOR_ARCHITECTURES[:1])
def test_locally_weighted_conformal_searcher(
    sampler_class, sampler_kwargs, point_arch, variance_arch, big_toy_dataset
):
    X, y = big_toy_dataset
    X_train, y_train = X[:7], y[:7]
    X_val, y_val = X[7:], y[7:]

    sampler = sampler_class(**sampler_kwargs)
    searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture=point_arch,
        variance_estimator_architecture=variance_arch,
        sampler=sampler,
    )

    searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )

    predictions = searcher.predict(X_val)
    assert len(predictions) == len(X_val)

    X_update = X_val[0].reshape(1, -1)
    y_update = y_val[0]
    initial_X_train_len = len(searcher.X_train)
    initial_y_train_len = len(searcher.y_train)

    searcher.update(X_update, y_update)

    assert len(searcher.X_train) == initial_X_train_len + 1
    assert len(searcher.y_train) == initial_y_train_len + 1
    assert np.array_equal(searcher.X_train[-1], X_update.flatten())
    assert searcher.y_train[-1] == y_update


@pytest.mark.parametrize(
    "sampler_class,sampler_kwargs",
    [
        (PessimisticLowerBoundSampler, {"interval_width": 0.8}),
        (LowerBoundSampler, {"interval_width": 0.8}),
        (ThompsonSampler, {"n_quantiles": 4}),
        (ExpectedImprovementSampler, {"n_quantiles": 4}),
        (InformationGainSampler, {"n_quantiles": 4}),
        (MaxValueEntropySearchSampler, {"n_quantiles": 4}),
    ],
)
@pytest.mark.parametrize(
    "quantile_arch",
    [
        QUANTILE_ESTIMATOR_ARCHITECTURES[0],
        SINGLE_FIT_QUANTILE_ESTIMATOR_ARCHITECTURES[0],
    ],
)
def test_quantile_conformal_searcher(
    sampler_class, sampler_kwargs, quantile_arch, big_toy_dataset
):
    X, y = big_toy_dataset
    X_train, y_train = X[:7], y[:7]
    X_val, y_val = X[7:], y[7:]

    sampler = sampler_class(**sampler_kwargs)
    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture=quantile_arch,
        sampler=sampler,
        n_pre_conformal_trials=5,
    )

    searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )

    predictions = searcher.predict(X_val)
    assert len(predictions) == len(X_val)

    X_update = X_val[0].reshape(1, -1)
    y_update = y_val[0]
    initial_X_train_len = len(searcher.X_train)
    initial_y_train_len = len(searcher.y_train)

    searcher.update(X_update, y_update)

    assert len(searcher.X_train) == initial_X_train_len + 1
    assert len(searcher.y_train) == initial_y_train_len + 1
    assert np.array_equal(searcher.X_train[-1], X_update.flatten())
    assert searcher.y_train[-1] == y_update


def test_locally_weighted_searcher_prediction_methods(big_toy_dataset):
    X, y = big_toy_dataset
    X_train, y_train = X[:7], y[:7]
    X_val, y_val = X[7:], y[7:]
    X_test = X_val

    lb_sampler = LowerBoundSampler(interval_width=0.8, beta_decay=None)
    lb_searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        sampler=lb_sampler,
    )
    lb_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    lb_predictions = lb_searcher.predict(X_test)
    assert len(lb_predictions) == len(X_test)

    thompson_sampler = ThompsonSampler(n_quantiles=4)
    thompson_searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        sampler=thompson_sampler,
    )
    thompson_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    thompson_predictions = thompson_searcher.predict(X_test)
    assert len(thompson_predictions) == len(X_test)

    ei_sampler = ExpectedImprovementSampler(n_quantiles=4, current_best_value=0.5)
    ei_searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        sampler=ei_sampler,
    )
    ei_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    ei_predictions = ei_searcher.predict(X_test)
    assert len(ei_predictions) == len(X_test)

    plb_sampler = PessimisticLowerBoundSampler(interval_width=0.8)
    plb_searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        sampler=plb_sampler,
    )
    plb_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    plb_predictions = plb_searcher.predict(X_test)
    assert len(plb_predictions) == len(X_test)

    assert not np.array_equal(lb_predictions, thompson_predictions)
    assert not np.array_equal(thompson_predictions, ei_predictions)
    assert not np.array_equal(ei_predictions, plb_predictions)


def test_locally_weighted_searcher_with_advanced_samplers(big_toy_dataset):
    X, y = big_toy_dataset
    X_train, y_train = X[:7], y[:7]
    X_val, y_val = X[7:], y[7:]
    X_test = X_val[:2]

    ig_sampler = InformationGainSampler(
        n_quantiles=4,
        n_paths=10,
        n_X_candidates=2,
        n_y_candidates_per_x=2,
        sampling_strategy="thompson",
    )
    ig_searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        sampler=ig_sampler,
    )
    ig_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    ig_predictions = ig_searcher.predict(X_test)
    assert len(ig_predictions) == len(X_test)

    mes_sampler = MaxValueEntropySearchSampler(
        n_quantiles=4,
        n_min_samples=10,
        n_y_samples=5,
    )
    mes_searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        variance_estimator_architecture=POINT_ESTIMATOR_ARCHITECTURES[0],
        sampler=mes_sampler,
    )
    mes_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    mes_predictions = mes_searcher.predict(X_test)
    assert len(mes_predictions) == len(X_test)


def test_quantile_searcher_prediction_methods(big_toy_dataset):
    X, y = big_toy_dataset
    X_train, y_train = X[:7], y[:7]
    X_val, y_val = X[7:], y[7:]
    X_test = X_val

    lb_sampler = LowerBoundSampler(interval_width=0.8, beta_decay=None)
    lb_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=lb_sampler,
        n_pre_conformal_trials=5,
    )
    lb_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    lb_predictions = lb_searcher.predict(X_test)
    assert len(lb_predictions) == len(X_test)

    thompson_sampler = ThompsonSampler(n_quantiles=4)
    thompson_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=thompson_sampler,
        n_pre_conformal_trials=5,
    )
    thompson_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    thompson_predictions = thompson_searcher.predict(X_test)
    assert len(thompson_predictions) == len(X_test)

    ei_sampler = ExpectedImprovementSampler(n_quantiles=4, current_best_value=0.5)
    ei_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=ei_sampler,
        n_pre_conformal_trials=5,
    )
    ei_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    ei_predictions = ei_searcher.predict(X_test)
    assert len(ei_predictions) == len(X_test)

    plb_sampler = PessimisticLowerBoundSampler(interval_width=0.8)
    plb_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=plb_sampler,
        n_pre_conformal_trials=5,
    )
    plb_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    plb_predictions = plb_searcher.predict(X_test)
    assert len(plb_predictions) == len(X_test)


def test_quantile_searcher_with_advanced_samplers(big_toy_dataset):
    X, y = big_toy_dataset
    X_train, y_train = X[:7], y[:7]
    X_val, y_val = X[7:], y[7:]
    X_test = X_val[:2]

    ig_sampler = InformationGainSampler(
        n_quantiles=4,
        n_paths=10,
        n_X_candidates=2,
        n_y_candidates_per_x=2,
        sampling_strategy="thompson",
    )
    ig_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=ig_sampler,
        n_pre_conformal_trials=5,
    )
    ig_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    ig_predictions = ig_searcher.predict(X_test)
    assert len(ig_predictions) == len(X_test)

    mes_sampler = MaxValueEntropySearchSampler(
        n_quantiles=4,
        n_min_samples=10,
        n_y_samples=5,
    )
    mes_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=mes_sampler,
        n_pre_conformal_trials=5,
    )
    mes_searcher.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        tuning_iterations=0,
        random_state=42,
    )
    mes_predictions = mes_searcher.predict(X_test)
    assert len(mes_predictions) == len(X_test)
