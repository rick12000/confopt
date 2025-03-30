import pytest
from unittest.mock import MagicMock

from confopt.tuning import (
    calculate_tuning_count,
    check_early_stopping,
    ConformalTuner,
)
from confopt.utils.tracking import Trial


@pytest.mark.parametrize("searcher_tuning_framework", ["runtime", "fixed", None])
def test_calculate_tuning_count(searcher_tuning_framework):
    # Runtime framework
    count = calculate_tuning_count(
        searcher_tuning_framework=searcher_tuning_framework,
        target_model_runtime=10.0,
        search_model_runtime=2.0,
        conformal_retraining_frequency=5,
    )
    if searcher_tuning_framework == "runtime":
        assert isinstance(count, int) and count >= 0
    elif searcher_tuning_framework == "fixed":
        assert count == 10
    elif searcher_tuning_framework is None:
        assert count == 0


@pytest.mark.parametrize(
    "searchable_indices,current_runtime,runtime_budget,current_iter,max_iter,n_random_searches,expected",
    [
        (
            [],
            None,
            None,
            None,
            None,
            None,
            (True, "All configurations have been searched"),
        ),  # Empty searchable indices
        (
            [1, 2, 3],
            11.0,
            10.0,
            None,
            None,
            None,
            (True, "Runtime budget (10.0) exceeded"),
        ),  # Runtime budget exceeded
        (
            [1, 2, 3],
            None,
            None,
            15,
            20,
            5,
            (True, "Maximum iterations (20) reached"),
        ),  # Max iterations reached
        ([1, 2, 3], 5.0, 10.0, 10, 30, 5, False),  # Normal operation (no stopping)
    ],
)
def test_check_early_stopping(
    searchable_indices,
    current_runtime,
    runtime_budget,
    current_iter,
    max_iter,
    n_random_searches,
    expected,
):
    result = check_early_stopping(
        searchable_indices=searchable_indices,
        current_runtime=current_runtime,
        runtime_budget=runtime_budget,
        current_iter=current_iter,
        max_iter=max_iter,
        n_random_searches=n_random_searches,
    )
    assert result == expected


class TestConformalTuner:
    def test_process_warm_start_configurations(
        self, mock_constant_objective_function, dummy_parameter_grid
    ):
        """Test that warm start configurations are properly processed"""
        warm_start_configs = [
            ({"param_1": 0.5, "param_2": 5, "param_3": "option1"}, 0.8),
            ({"param_1": 1.0, "param_2": 10, "param_3": "option2"}, 0.6),
        ]

        # Create a custom tuner with warm start configurations
        tuner = ConformalTuner(
            objective_function=mock_constant_objective_function,
            search_space=dummy_parameter_grid,
            metric_optimization="minimize",
            n_candidate_configurations=100,
            warm_start_configurations=warm_start_configs,
        )

        # Initialize tuning resources which calls _process_warm_start_configurations
        tuner._initialize_tuning_resources()

        # Verify that warm start configs are properly processed
        assert (
            len(tuner.study.trials) == 2
        ), "Should have added two trials from warm start"

        # Check that the configurations in trials match the warm start configs
        for i, (config, _) in enumerate(warm_start_configs):
            assert tuner.study.trials[i].configuration == config

        # Check that searched indices and performances are updated
        assert len(tuner.searched_indices) == 2
        assert len(tuner.searched_performances) == 2

        # Check that searchable indices don't include the warm start indices
        for idx in tuner.searched_indices:
            assert idx not in tuner.searchable_indices

        # Check that the total number of indices is preserved
        assert len(tuner.searchable_indices) + len(tuner.searched_indices) == len(
            tuner.tuning_configurations
        )

    def test_update_search_state(self, tuner):
        # Initialize tuning resources
        tuner._initialize_tuning_resources()

        # Save the initial state
        initial_searchable_indices = tuner.searchable_indices.copy()
        initial_searched_indices = tuner.searched_indices.copy()
        initial_searched_performances = tuner.searched_performances.copy()

        # Select a config index to update
        config_idx = 5
        performance = 0.75

        # Call the method under test
        tuner._update_search_state(config_idx=config_idx, performance=performance)

        # Verify that config_idx was added to searched_indices
        assert config_idx in tuner.searched_indices
        assert len(tuner.searched_indices) == len(initial_searched_indices) + 1

        # Verify that performance was added to searched_performances
        assert performance in tuner.searched_performances
        assert (
            len(tuner.searched_performances) == len(initial_searched_performances) + 1
        )

        # Verify that config_idx was removed from searchable_indices
        assert config_idx not in tuner.searchable_indices
        assert len(tuner.searchable_indices) == len(initial_searchable_indices) - 1

    def test_random_search(self, tuner):
        tuner._initialize_tuning_resources()

        # Save the initial state
        initial_searchable_indices_count = len(tuner.searchable_indices)
        initial_searched_indices_count = len(tuner.searched_indices)

        # Call the method under test with a small number of searches
        n_searches = 3
        trials = tuner._random_search(n_searches=n_searches, verbose=False)

        # Verify that the correct number of trials were returned
        assert len(trials) == n_searches

        # Verify that the search state was updated correctly
        assert (
            len(tuner.searched_indices) == initial_searched_indices_count + n_searches
        )
        assert (
            len(tuner.searchable_indices)
            == initial_searchable_indices_count - n_searches
        )

        # Verify that each trial has the correct metadata
        for trial in trials:
            assert isinstance(trial, Trial)
            assert trial.acquisition_source == "rs"
            assert trial.performance == 2

    def test_random_search_early_stopping(self, tuner):
        """Test that random search stops when runtime budget is exceeded."""
        tuner._initialize_tuning_resources()

        # Mock the search timer to return a runtime that exceeds the budget
        tuner.search_timer = MagicMock()
        tuner.search_timer.return_runtime = MagicMock(return_value=11.0)

        # Verify that RuntimeError is raised when budget is exceeded
        with pytest.raises(RuntimeError):
            tuner._random_search(n_searches=5, verbose=False, max_runtime=10.0)

    def test_tune_with_default_searcher(self, tuner):
        tuner.tune(n_random_searches=20, max_iter=30, verbose=False)

        assert len(tuner.study.trials) == 30
