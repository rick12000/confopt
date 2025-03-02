import random
from copy import deepcopy

import numpy as np
import pandas as pd

from confopt.config import GBM_NAME
from confopt.tracking import RuntimeTracker
from confopt.tuning import (
    process_and_split_estimation_data,
    normalize_estimation_data,
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


def test_get_tuning_configurations(dummy_initialized_conformal_searcher__gbm_mse):
    stored_search_space = dummy_initialized_conformal_searcher__gbm_mse.search_space

    tuning_configurations = (
        dummy_initialized_conformal_searcher__gbm_mse._get_tuning_configurations()
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
        == dummy_initialized_conformal_searcher__gbm_mse.search_space
    )


def test_get_tuning_configurations__reproducibility(
    dummy_initialized_conformal_searcher__gbm_mse,
):
    assert (
        dummy_initialized_conformal_searcher__gbm_mse._get_tuning_configurations()
        == dummy_initialized_conformal_searcher__gbm_mse._get_tuning_configurations()
    )


def test_evaluate_configuration_performance(
    dummy_initialized_conformal_searcher__gbm_mse, dummy_gbm_configurations
):
    # Arbitrarily select the first configuration in the list:
    dummy_configuration = dummy_gbm_configurations[0]
    stored_dummy_configuration = deepcopy(dummy_configuration)

    performance = dummy_initialized_conformal_searcher__gbm_mse._evaluate_configuration_performance(
        configuration=dummy_configuration, random_state=DEFAULT_SEED
    )

    assert performance > 0
    # Test for mutability:
    assert stored_dummy_configuration == dummy_configuration


def test_evaluate_configuration_performance__reproducibility(
    dummy_initialized_conformal_searcher__gbm_mse, dummy_gbm_configurations
):
    # Arbitrarily select the first configuration in the list:
    dummy_configuration = dummy_gbm_configurations[0]

    assert dummy_initialized_conformal_searcher__gbm_mse._evaluate_configuration_performance(
        configuration=dummy_configuration, random_state=DEFAULT_SEED
    ) == dummy_initialized_conformal_searcher__gbm_mse._evaluate_configuration_performance(
        configuration=dummy_configuration, random_state=DEFAULT_SEED
    )


def test_random_search(dummy_initialized_conformal_searcher__gbm_mse):
    n_searches = 5
    max_runtime = 30
    dummy_initialized_conformal_searcher__gbm_mse.search_timer = RuntimeTracker()

    (
        searched_configurations,
        searched_performances,
        searched_timestamps,
        runtime_per_search,
    ) = dummy_initialized_conformal_searcher__gbm_mse._random_search(
        n_searches=n_searches,
        max_runtime=max_runtime,
        random_state=DEFAULT_SEED,
    )

    for performance in searched_performances:
        assert performance > 0
    assert len(searched_configurations) > 0
    assert len(searched_performances) > 0
    assert len(searched_timestamps) > 0
    assert (
        len(searched_configurations)
        == len(searched_performances)
        == len(searched_timestamps)
    )
    assert len(searched_configurations) == n_searches
    assert 0 < runtime_per_search < max_runtime


def test_random_search__reproducibility(
    dummy_initialized_conformal_searcher__gbm_mse,
):
    n_searches = 5
    max_runtime = 30
    dummy_initialized_conformal_searcher__gbm_mse.search_timer = RuntimeTracker()

    (
        searched_configurations_first_call,
        searched_performances_first_call,
        _,
        _,
    ) = dummy_initialized_conformal_searcher__gbm_mse._random_search(
        n_searches=n_searches,
        max_runtime=max_runtime,
        random_state=DEFAULT_SEED,
    )
    (
        searched_configurations_second_call,
        searched_performances_second_call,
        _,
        _,
    ) = dummy_initialized_conformal_searcher__gbm_mse._random_search(
        n_searches=n_searches,
        max_runtime=max_runtime,
        random_state=DEFAULT_SEED,
    )

    assert searched_configurations_first_call == searched_configurations_second_call
    assert searched_performances_first_call == searched_performances_second_call


def test_search(dummy_initialized_conformal_searcher__gbm_mse):
    # TODO: Below I hard coded a slice of possible inputs, but consider
    #  pytest parametrizing these (though test will be very heavy,
    #  so tag as slow and only run when necessary)
    confidence_level = 0.2
    conformal_model_type = GBM_NAME
    conformal_retraining_frequency = 1
    conformal_learning_rate = 0.01
    enable_adaptive_intervals = True
    max_runtime = 120
    min_training_iterations = 20

    stored_search_space = dummy_initialized_conformal_searcher__gbm_mse.search_space
    stored_tuning_configurations = (
        dummy_initialized_conformal_searcher__gbm_mse.tuning_configurations
    )

    dummy_initialized_conformal_searcher__gbm_mse.search(
        conformal_search_estimator=conformal_model_type,
        confidence_level=confidence_level,
        n_random_searches=min_training_iterations,
        runtime_budget=max_runtime,
        conformal_retraining_frequency=conformal_retraining_frequency,
        conformal_learning_rate=conformal_learning_rate,
        enable_adaptive_intervals=enable_adaptive_intervals,
        verbose=0,
    )

    assert (
        len(dummy_initialized_conformal_searcher__gbm_mse.searched_configurations) > 0
    )
    assert len(dummy_initialized_conformal_searcher__gbm_mse.searched_performances) > 0
    assert len(
        dummy_initialized_conformal_searcher__gbm_mse.searched_configurations
    ) == len(dummy_initialized_conformal_searcher__gbm_mse.searched_performances)
    # Test for mutability:
    assert (
        stored_search_space
        == dummy_initialized_conformal_searcher__gbm_mse.search_space
    )
    assert (
        stored_tuning_configurations
        == dummy_initialized_conformal_searcher__gbm_mse.tuning_configurations
    )


def test_search__reproducibility(dummy_initialized_conformal_searcher__gbm_mse):
    confidence_level = 0.2
    conformal_model_type = GBM_NAME
    conformal_retraining_frequency = 1
    conformal_learning_rate = 0.01
    enable_adaptive_intervals = True
    max_runtime = 120
    min_training_iterations = 20

    searcher_first_call = deepcopy(dummy_initialized_conformal_searcher__gbm_mse)
    searcher_second_call = deepcopy(dummy_initialized_conformal_searcher__gbm_mse)

    searcher_first_call.search(
        conformal_search_estimator=conformal_model_type,
        confidence_level=confidence_level,
        n_random_searches=min_training_iterations,
        runtime_budget=max_runtime,
        conformal_retraining_frequency=conformal_retraining_frequency,
        conformal_learning_rate=conformal_learning_rate,
        enable_adaptive_intervals=enable_adaptive_intervals,
        verbose=0,
        random_state=DEFAULT_SEED,
    )
    searcher_second_call.search(
        conformal_search_estimator=conformal_model_type,
        confidence_level=confidence_level,
        n_random_searches=min_training_iterations,
        runtime_budget=max_runtime,
        conformal_retraining_frequency=conformal_retraining_frequency,
        conformal_learning_rate=conformal_learning_rate,
        enable_adaptive_intervals=enable_adaptive_intervals,
        verbose=0,
        random_state=DEFAULT_SEED,
    )

    assert (
        searcher_first_call.searched_configurations
        == searcher_second_call.searched_configurations
    )
    assert (
        searcher_first_call.searched_performances
        == searcher_second_call.searched_performances
    )


def test_objective_search(dummy_initialized_objective_conformal_searcher__gbm_mse):
    # TODO: Below I hard coded a slice of possible inputs, but consider
    #  pytest parametrizing these (though test will be very heavy,
    #  so tag as slow and only run when necessary)
    confidence_level = 0.2
    conformal_model_type = GBM_NAME
    conformal_retraining_frequency = 1
    conformal_learning_rate = 0.01
    enable_adaptive_intervals = True
    max_runtime = 120
    min_training_iterations = 20

    stored_search_space = (
        dummy_initialized_objective_conformal_searcher__gbm_mse.search_space
    )
    stored_tuning_configurations = (
        dummy_initialized_objective_conformal_searcher__gbm_mse.tuning_configurations
    )

    dummy_initialized_objective_conformal_searcher__gbm_mse.search(
        conformal_search_estimator=conformal_model_type,
        confidence_level=confidence_level,
        n_random_searches=min_training_iterations,
        runtime_budget=max_runtime,
        conformal_retraining_frequency=conformal_retraining_frequency,
        conformal_learning_rate=conformal_learning_rate,
        enable_adaptive_intervals=enable_adaptive_intervals,
        verbose=0,
    )

    assert (
        len(
            dummy_initialized_objective_conformal_searcher__gbm_mse.searched_configurations
        )
        > 0
    )
    assert (
        len(
            dummy_initialized_objective_conformal_searcher__gbm_mse.searched_performances
        )
        > 0
    )
    assert len(
        dummy_initialized_objective_conformal_searcher__gbm_mse.searched_configurations
    ) == len(
        dummy_initialized_objective_conformal_searcher__gbm_mse.searched_performances
    )
    # Test for mutability:
    assert (
        stored_search_space
        == dummy_initialized_objective_conformal_searcher__gbm_mse.search_space
    )
    assert (
        stored_tuning_configurations
        == dummy_initialized_objective_conformal_searcher__gbm_mse.tuning_configurations
    )


def test_objective_search__reproducibility(
    dummy_initialized_objective_conformal_searcher__gbm_mse,
):
    confidence_level = 0.2
    conformal_model_type = GBM_NAME
    conformal_retraining_frequency = 1
    conformal_learning_rate = 0.01
    enable_adaptive_intervals = True
    max_runtime = 120
    min_training_iterations = 20

    searcher_first_call = deepcopy(
        dummy_initialized_objective_conformal_searcher__gbm_mse
    )
    searcher_second_call = deepcopy(
        dummy_initialized_objective_conformal_searcher__gbm_mse
    )

    searcher_first_call.search(
        conformal_search_estimator=conformal_model_type,
        confidence_level=confidence_level,
        n_random_searches=min_training_iterations,
        runtime_budget=max_runtime,
        conformal_retraining_frequency=conformal_retraining_frequency,
        conformal_learning_rate=conformal_learning_rate,
        enable_adaptive_intervals=enable_adaptive_intervals,
        verbose=0,
        random_state=DEFAULT_SEED,
    )
    searcher_second_call.search(
        conformal_search_estimator=conformal_model_type,
        confidence_level=confidence_level,
        n_random_searches=min_training_iterations,
        runtime_budget=max_runtime,
        conformal_retraining_frequency=conformal_retraining_frequency,
        conformal_learning_rate=conformal_learning_rate,
        enable_adaptive_intervals=enable_adaptive_intervals,
        verbose=0,
        random_state=DEFAULT_SEED,
    )

    assert (
        searcher_first_call.searched_configurations
        == searcher_second_call.searched_configurations
    )
    assert (
        searcher_first_call.searched_performances
        == searcher_second_call.searched_performances
    )
