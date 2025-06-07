import pytest
from unittest.mock import MagicMock

from confopt.tuning import (
    check_early_stopping,
    ConformalTuner,
)
from confopt.utils.tracking import Trial


@pytest.mark.parametrize(
    "searchable_count,current_runtime,runtime_budget,current_iter,max_iter,n_random_searches,expected",
    [
        (
            0,
            None,
            None,
            None,
            None,
            None,
            (True, "All configurations have been searched"),
        ),  # Empty searchable indices
        (
            3,
            11.0,
            10.0,
            None,
            None,
            None,
            (True, "Runtime budget (10.0) exceeded"),
        ),  # Runtime budget exceeded
        (
            3,
            None,
            None,
            15,
            20,
            5,
            (True, "Maximum iterations (20) reached"),
        ),  # Max iterations reached
        (3, 5.0, 10.0, 10, 30, 5, False),  # Normal operation (no stopping)
    ],
)
def test_check_early_stopping(
    searchable_count,
    current_runtime,
    runtime_budget,
    current_iter,
    max_iter,
    n_random_searches,
    expected,
):
    result = check_early_stopping(
        searchable_count=searchable_count,
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

        # Check that searched configs and performances are updated
        assert len(tuner.searched_configs) == 2
        assert len(tuner.searched_performances) == 2

        # Check that the configs are in the searched_configs_set
        from confopt.tuning import create_config_hash

        for config, _ in warm_start_configs:
            config_hash = create_config_hash(config)
            assert config_hash in tuner.searched_configs_set

        # Check that warm start configs aren't in searchable configs
        for config, _ in warm_start_configs:
            # Check it's not in searchable configurations
            assert config not in tuner.searchable_configs

    def test_update_search_state(self, tuner):
        # Initialize tuning resources
        tuner._initialize_tuning_resources()

        # Save the initial state
        initial_searchable_count = len(tuner.searchable_configs)
        initial_searched_count = len(tuner.searched_configs)
        initial_searched_performances = tuner.searched_performances.copy()

        # Select a config to update
        config = tuner.searchable_configs[0]
        performance = 0.75

        # Call the method under test
        tuner._update_search_state(config=config, performance=performance)

        # Verify that config was added to searched_configs
        assert config in tuner.searched_configs
        assert len(tuner.searched_configs) == initial_searched_count + 1

        # Verify that performance was added to searched_performances
        assert performance in tuner.searched_performances
        assert (
            len(tuner.searched_performances) == len(initial_searched_performances) + 1
        )

        # Verify that config was removed from searchable_configs
        assert config not in tuner.searchable_configs
        assert len(tuner.searchable_configs) == initial_searchable_count - 1

    def test_random_search(self, tuner):
        tuner._initialize_tuning_resources()

        # Save the initial state
        initial_searchable_count = len(tuner.searchable_configs)
        initial_searched_count = len(tuner.searched_configs)

        # Call the method under test with a small number of searches
        n_searches = 3
        trials = tuner._random_search(n_searches=n_searches, verbose=False)

        # Verify that the correct number of trials were returned
        assert len(trials) == n_searches

        # Verify that the search state was updated correctly
        assert len(tuner.searched_configs) == initial_searched_count + n_searches
        assert len(tuner.searchable_configs) == initial_searchable_count - n_searches

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

    @pytest.mark.parametrize(
        "searcher_tuning_framework", ["reward_cost", "fixed", None]
    )
    def test_tune_with_default_searcher(self, tuner, searcher_tuning_framework):
        tuner.tune(
            n_random_searches=30,
            max_iter=35,
            verbose=False,
            searcher_tuning_framework=searcher_tuning_framework,
        )

        assert len(tuner.study.trials) == 35

    def test_reproducibility_with_fixed_random_state(
        self, mock_constant_objective_function, dummy_parameter_grid
    ):
        common_params = {
            "objective_function": mock_constant_objective_function,
            "search_space": dummy_parameter_grid,
            "metric_optimization": "minimize",
            "n_candidate_configurations": 100,
        }
        tune_params = {
            "n_random_searches": 10,
            "max_iter": 35,
            "verbose": False,
            "random_state": 42,
        }

        tuner1 = ConformalTuner(**common_params)
        tuner1.tune(**tune_params)

        tuner2 = ConformalTuner(**common_params)
        tuner2.tune(**tune_params)

        assert len(tuner1.study.trials) == len(tuner2.study.trials)
        for trial1, trial2 in zip(tuner1.study.trials, tuner2.study.trials):
            assert trial1.configuration == trial2.configuration
            assert trial1.performance == trial2.performance
