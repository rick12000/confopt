import pytest
import numpy as np
from confopt.selection.acquisition import (
    QuantileConformalSearcher,
)
from confopt.selection.sampling.bound_samplers import (
    PessimisticLowerBoundSampler,
    LowerBoundSampler,
)
from confopt.selection.sampling.thompson_samplers import ThompsonSampler
from confopt.selection.sampling.expected_improvement_samplers import (
    ExpectedImprovementSampler,
)
from confopt.selection.sampling.entropy_samplers import (
    MaxValueEntropySearchSampler,
)
from conftest import (
    QUANTILE_ESTIMATOR_ARCHITECTURES,
)


@pytest.mark.parametrize(
    "sampler_class,sampler_kwargs",
    [
        (PessimisticLowerBoundSampler, {"interval_width": 0.8}),
        (LowerBoundSampler, {"interval_width": 0.8}),
        (ThompsonSampler, {"n_quantiles": 4}),
        (ExpectedImprovementSampler, {"n_quantiles": 4}),
        (MaxValueEntropySearchSampler, {"n_quantiles": 4}),
    ],
)
@pytest.mark.parametrize("quantile_arch", QUANTILE_ESTIMATOR_ARCHITECTURES[:1])
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

    # Combine train and val data for new interface
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    searcher.fit(
        X=X_combined,
        y=y_combined,
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

    # Data doesn't change, only updates samplers and other states:
    assert len(searcher.X_train) == initial_X_train_len
    assert len(searcher.y_train) == initial_y_train_len


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
    # Combine train and val data for new interface
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    lb_searcher.fit(
        X=X_combined,
        y=y_combined,
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
        X=X_combined,
        y=y_combined,
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
        X=X_combined,
        y=y_combined,
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
        X=X_combined,
        y=y_combined,
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

    # Combine train and val data for new interface
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))

    mes_sampler = MaxValueEntropySearchSampler(
        n_quantiles=4,
        n_paths=10,
        n_y_candidates_per_x=5,
    )
    mes_searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=mes_sampler,
        n_pre_conformal_trials=5,
    )
    mes_searcher.fit(
        X=X_combined,
        y=y_combined,
        tuning_iterations=0,
        random_state=42,
    )
    mes_predictions = mes_searcher.predict(X_test)
    assert len(mes_predictions) == len(X_test)


@pytest.mark.parametrize("current_best_value", [0.0, 0.5, 1.0, 10.0])
def test_expected_improvement_best_value_update(current_best_value, big_toy_dataset):
    """Test that Expected Improvement properly tracks and updates best values."""
    X, y = big_toy_dataset
    X_train, y_train = X[:10], y[:10]
    X_val, y_val = X[10:20], y[10:20]

    sampler = ExpectedImprovementSampler(
        n_quantiles=4, current_best_value=current_best_value
    )
    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=sampler,
        n_pre_conformal_trials=5,
    )

    # Combine train and val data for new interface
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    searcher.fit(X=X_combined, y=y_combined, tuning_iterations=0, random_state=42)

    # Test that sampler has correct initial best value
    assert sampler.current_best_value == current_best_value

    # Test update with better value (remember: we minimize, so lower is better)
    new_value = current_best_value - 1.0
    searcher.update(X_val[0], new_value)
    assert sampler.current_best_value == new_value

    # Test update with worse value (should not change)
    worse_value = current_best_value + 1.0
    searcher.update(X_val[1], worse_value)
    assert sampler.current_best_value == new_value  # Should remain the better value


def test_adaptive_alpha_updating(big_toy_dataset):
    """Test that adaptive alpha updating works correctly for compatible samplers."""
    X, y = big_toy_dataset
    X_train, y_train = X[:15], y[:15]
    X_val, y_val = X[15:30], y[15:30]

    # Test with adaptive sampler
    sampler = LowerBoundSampler(interval_width=0.8, adapter="DtACI")
    searcher = QuantileConformalSearcher(
        quantile_estimator_architecture="ql",
        sampler=sampler,
        n_pre_conformal_trials=5,
    )

    # Combine train and val data for new interface
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    searcher.fit(X=X_combined, y=y_combined, tuning_iterations=0, random_state=42)

    # Store initial alpha values
    initial_alphas = searcher.sampler.fetch_alphas().copy()

    # Perform several updates
    for i in range(3):
        test_point = X_val[i]
        test_value = y_val[i]
        searcher.update(test_point, test_value)

    # Check that alphas change:
    final_alphas = searcher.sampler.fetch_alphas()
    assert len(final_alphas) == len(initial_alphas)
    assert all(0 < alpha < 1 for alpha in final_alphas)
    assert not np.array_equal(initial_alphas, final_alphas)
