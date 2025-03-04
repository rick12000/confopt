import random
from copy import deepcopy

import numpy as np
import pandas as pd

from confopt.tracking import RuntimeTracker, Trial
from confopt.tuning import (
    process_and_split_estimation_data,
    normalize_estimation_data,
)
from confopt.estimation import (
    LocallyWeightedConformalSearcher,
    UCBSampler,
)

DEFAULT_SEED = 1234


def test_process_and_split_estimation_data(dummy_configurations):
    train_split = 0.5
    dummy_searched_configurations = pd.DataFrame(dummy_configurations).to_numpy()
    stored_dummy_searched_configurations = deepcopy(dummy_searched_configurations)
    dummy_searched_performances = np.array(
        [random.random() for _ in range(len(dummy_configurations))]
    )
    stored_dummy_searched_performances = deepcopy(dummy_searched_performances)

    X_train, y_train, X_val, y_val = process_and_split_estimation_data(
        searched_configurations=dummy_searched_configurations,
        searched_performances=dummy_searched_performances,
        train_split=train_split,
        filter_outliers=False,
        outlier_scope=None,
        random_state=DEFAULT_SEED,
    )

    assert len(X_val) == len(y_val)
    assert len(X_train) == len(y_train)

    assert len(X_val) + len(X_train) == len(dummy_searched_configurations)

    assert (
        abs(len(X_train) - round(len(dummy_searched_configurations) * train_split)) <= 1
    )
    assert (
        abs(len(X_val) - round(len(dummy_searched_configurations) * (1 - train_split)))
        <= 1
    )

    # Assert there is no mutability of input:
    assert np.array_equal(
        dummy_searched_configurations, stored_dummy_searched_configurations
    )
    assert np.array_equal(
        dummy_searched_performances, stored_dummy_searched_performances
    )


def test_process_and_split_estimation_data__reproducibility(dummy_configurations):
    train_split = 0.5
    dummy_searched_configurations = pd.DataFrame(dummy_configurations).to_numpy()
    dummy_searched_performances = np.array(
        [random.random() for _ in range(len(dummy_configurations))]
    )

    (
        X_train_first_call,
        y_train_first_call,
        X_val_first_call,
        y_val_first_call,
    ) = process_and_split_estimation_data(
        searched_configurations=dummy_searched_configurations,
        searched_performances=dummy_searched_performances,
        train_split=train_split,
        filter_outliers=False,
        outlier_scope=None,
        random_state=DEFAULT_SEED,
    )
    (
        X_train_second_call,
        y_train_second_call,
        X_val_second_call,
        y_val_second_call,
    ) = process_and_split_estimation_data(
        searched_configurations=dummy_searched_configurations,
        searched_performances=dummy_searched_performances,
        train_split=train_split,
        filter_outliers=False,
        outlier_scope=None,
        random_state=DEFAULT_SEED,
    )

    assert np.array_equal(X_train_first_call, X_train_second_call)
    assert np.array_equal(y_train_first_call, y_train_second_call)
    assert np.array_equal(X_val_first_call, X_val_second_call)
    assert np.array_equal(y_val_first_call, y_val_second_call)


def test_normalize_estimation_data(dummy_configurations):
    # Proportion of all candidate configurations that
    # have already been searched:
    searched_split = 0.5
    # Split of searched configurations that is used as
    # training data for the search estimator:
    train_split = 0.5

    dummy_searched_configurations = dummy_configurations[
        : round(len(dummy_configurations) * searched_split)
    ]
    dummy_searchable_configurations = pd.DataFrame(
        dummy_configurations[round(len(dummy_configurations) * searched_split) :]
    ).to_numpy()
    stored_dummy_searchable_configurations = deepcopy(dummy_searchable_configurations)
    dummy_training_searched_configurations = pd.DataFrame(
        dummy_searched_configurations[
            : round(len(dummy_searched_configurations) * train_split)
        ]
    ).to_numpy()
    stored_dummy_training_searched_configurations = deepcopy(
        dummy_training_searched_configurations
    )
    dummy_validation_searched_configurations = pd.DataFrame(
        dummy_searched_configurations[
            round(len(dummy_searched_configurations) * train_split) :
        ]
    ).to_numpy()
    stored_dummy_validation_searched_configurations = deepcopy(
        dummy_validation_searched_configurations
    )

    (
        normalized_training_searched_configurations,
        normalized_validation_searched_configurations,
        normalized_searchable_configurations,
    ) = normalize_estimation_data(
        training_searched_configurations=dummy_training_searched_configurations,
        validation_searched_configurations=dummy_validation_searched_configurations,
        searchable_configurations=dummy_searchable_configurations,
    )

    assert len(normalized_training_searched_configurations) == len(
        dummy_training_searched_configurations
    )
    assert len(normalized_validation_searched_configurations) == len(
        normalized_validation_searched_configurations
    )
    assert len(normalized_searchable_configurations) == len(
        normalized_searchable_configurations
    )

    # Assert there is no mutability of inputs:
    assert np.array_equal(
        dummy_training_searched_configurations,
        stored_dummy_training_searched_configurations,
    )
    assert np.array_equal(
        dummy_validation_searched_configurations,
        stored_dummy_validation_searched_configurations,
    )
    assert np.array_equal(
        dummy_searchable_configurations, stored_dummy_searchable_configurations
    )


def test_get_tuning_configurations(
    dummy_initialized_objective_conformal_searcher__gbm_mse,
):
    stored_search_space = (
        dummy_initialized_objective_conformal_searcher__gbm_mse.search_space
    )

    tuning_configurations = (
        dummy_initialized_objective_conformal_searcher__gbm_mse._get_tuning_configurations()
    )

    for configuration in tuning_configurations:
        for param_name, param_value in configuration.items():
            # Check configuration only has parameter names from parameter grid prompt:
            assert param_name in stored_search_space.keys()
            # Check values in configuration come from range in parameter grid prompt:
            assert param_value in stored_search_space[param_name]
    # Test for mutability:
    assert (
        stored_search_space
        == dummy_initialized_objective_conformal_searcher__gbm_mse.search_space
    )


def test_get_tuning_configurations__reproducibility(
    dummy_initialized_objective_conformal_searcher__gbm_mse,
):
    tuning_configs_first_call = (
        dummy_initialized_objective_conformal_searcher__gbm_mse._get_tuning_configurations()
    )
    tuning_configs_second_call = (
        dummy_initialized_objective_conformal_searcher__gbm_mse._get_tuning_configurations()
    )
    assert tuning_configs_first_call == tuning_configs_second_call


def test_random_search(dummy_initialized_objective_conformal_searcher__gbm_mse):
    n_searches = 5
    dummy_initialized_objective_conformal_searcher__gbm_mse.search_timer = (
        RuntimeTracker()
    )

    rs_trials = dummy_initialized_objective_conformal_searcher__gbm_mse._random_search(
        n_searches=n_searches,
        max_runtime=30,
        verbose=False,
    )

    assert len(rs_trials) > 0
    assert len(rs_trials) == n_searches

    for trial in rs_trials:
        assert isinstance(trial, Trial)
        assert trial.acquisition_source == "rs"
        assert trial.configuration is not None
        assert trial.timestamp is not None


def test_random_search__reproducibility(
    dummy_initialized_objective_conformal_searcher__gbm_mse,
):
    n_searches = 5
    dummy_initialized_objective_conformal_searcher__gbm_mse.search_timer = (
        RuntimeTracker()
    )

    # Set numpy random seed for reproducibility
    np.random.seed(DEFAULT_SEED)
    rs_trials_first_call = (
        dummy_initialized_objective_conformal_searcher__gbm_mse._random_search(
            n_searches=n_searches,
            max_runtime=30,
            verbose=False,
        )
    )

    # Reset random seed
    np.random.seed(DEFAULT_SEED)
    rs_trials_second_call = (
        dummy_initialized_objective_conformal_searcher__gbm_mse._random_search(
            n_searches=n_searches,
            max_runtime=30,
            verbose=False,
        )
    )

    # Check that the same configurations were selected
    for first_trial, second_trial in zip(rs_trials_first_call, rs_trials_second_call):
        assert first_trial.configuration == second_trial.configuration
        assert first_trial.performance == second_trial.performance


def test_search(dummy_initialized_objective_conformal_searcher__gbm_mse):
    searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture="gbm",
        variance_estimator_architecture="gbm",
        sampler=UCBSampler(c=1, interval_width=0.8),
    )

    n_random_searches = 5
    max_iter = 8

    stored_search_space = (
        dummy_initialized_objective_conformal_searcher__gbm_mse.search_space
    )
    stored_tuning_configurations = (
        dummy_initialized_objective_conformal_searcher__gbm_mse.tuning_configurations
    )

    dummy_initialized_objective_conformal_searcher__gbm_mse.search(
        searcher=searcher,
        n_random_searches=n_random_searches,
        max_iter=max_iter,
        conformal_retraining_frequency=1,
        verbose=False,
        random_state=DEFAULT_SEED,
    )

    # Check that trials were recorded
    assert len(dummy_initialized_objective_conformal_searcher__gbm_mse.study.trials) > 0
    assert (
        len(dummy_initialized_objective_conformal_searcher__gbm_mse.study.trials)
        == max_iter
    )

    # Check that random search and conformal search trials are both present
    rs_trials = [
        t
        for t in dummy_initialized_objective_conformal_searcher__gbm_mse.study.trials
        if t.acquisition_source == "rs"
    ]
    conf_trials = [
        t
        for t in dummy_initialized_objective_conformal_searcher__gbm_mse.study.trials
        if t.acquisition_source != "rs"
    ]

    assert len(rs_trials) == n_random_searches
    assert len(conf_trials) == max_iter - n_random_searches

    # Test for mutability:
    assert (
        stored_search_space
        == dummy_initialized_objective_conformal_searcher__gbm_mse.search_space
    )
    assert (
        stored_tuning_configurations
        == dummy_initialized_objective_conformal_searcher__gbm_mse.tuning_configurations
    )


def test_search__reproducibility(
    dummy_initialized_objective_conformal_searcher__gbm_mse,
):
    searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture="gbm",
        variance_estimator_architecture="gbm",
        sampler=UCBSampler(c=1, interval_width=0.8),
    )

    n_random_searches = 5
    max_iter = 8

    # Create copies for two independent runs
    searcher_first_call = deepcopy(
        dummy_initialized_objective_conformal_searcher__gbm_mse
    )
    searcher_second_call = deepcopy(
        dummy_initialized_objective_conformal_searcher__gbm_mse
    )

    # Run with same random seed
    searcher_first_call.search(
        searcher=searcher,
        n_random_searches=n_random_searches,
        max_iter=max_iter,
        conformal_retraining_frequency=1,
        verbose=False,
        random_state=DEFAULT_SEED,
    )

    searcher_second_call.search(
        searcher=searcher,
        n_random_searches=n_random_searches,
        max_iter=max_iter,
        conformal_retraining_frequency=1,
        verbose=False,
        random_state=DEFAULT_SEED,
    )

    # Check that the same configurations were selected and performances match
    for first_trial, second_trial in zip(
        searcher_first_call.study.trials, searcher_second_call.study.trials
    ):
        assert first_trial.configuration == second_trial.configuration
        assert first_trial.performance == second_trial.performance
        assert first_trial.acquisition_source == second_trial.acquisition_source


def test_get_best_params(dummy_initialized_objective_conformal_searcher__gbm_mse):
    # Setup a simple trial with some sample configurations
    searcher = dummy_initialized_objective_conformal_searcher__gbm_mse
    config1 = {"param1": 1, "param2": 2}
    config2 = {"param1": 3, "param2": 4}

    trial1 = Trial(
        iteration=0,
        timestamp=pd.Timestamp.now(),
        configuration=config1,
        performance=10.0,
    )
    trial2 = Trial(
        iteration=1,
        timestamp=pd.Timestamp.now(),
        configuration=config2,
        performance=5.0,
    )

    searcher.study.batch_append_trials([trial1, trial2])

    # Test that get_best_params returns the config with the lowest performance
    best_params = searcher.get_best_params()
    assert best_params == config2


def test_get_best_value(dummy_initialized_objective_conformal_searcher__gbm_mse):
    # Setup a simple trial with some sample configurations
    searcher = dummy_initialized_objective_conformal_searcher__gbm_mse
    config1 = {"param1": 1, "param2": 2}
    config2 = {"param1": 3, "param2": 4}

    trial1 = Trial(
        iteration=0,
        timestamp=pd.Timestamp.now(),
        configuration=config1,
        performance=10.0,
    )
    trial2 = Trial(
        iteration=1,
        timestamp=pd.Timestamp.now(),
        configuration=config2,
        performance=5.0,
    )

    searcher.study.batch_append_trials([trial1, trial2])

    # Test that get_best_value returns the lowest performance value
    best_value = searcher.get_best_value()
    assert best_value == 5.0
