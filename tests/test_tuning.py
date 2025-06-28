import pytest
from unittest.mock import MagicMock

from confopt.tuning import (
    check_early_stopping,
    ConformalTuner,
    create_config_hash,
)
from confopt.utils.tracking import Trial


@pytest.mark.parametrize(
    "searchable_count,current_runtime,runtime_budget,current_iter,max_iter,expected",
    [
        (
            0,
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
            (True, "Runtime budget (10.0) exceeded"),
        ),  # Runtime budget exceeded
        (
            3,
            None,
            None,
            20,
            20,
            (True, "Maximum iterations (20) reached"),
        ),  # Max iterations reached (when current_iter >= max_iter)
        (
            3,
            5.0,
            10.0,
            10,
            30,
            (False, "No stopping condition met"),
        ),  # Normal operation (no stopping)
    ],
)
def test_check_early_stopping(
    searchable_count,
    current_runtime,
    runtime_budget,
    current_iter,
    max_iter,
    expected,
):
    result = check_early_stopping(
        searchable_count=searchable_count,
        current_runtime=current_runtime,
        runtime_budget=runtime_budget,
        current_iter=current_iter,
        max_iter=max_iter,
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
        for config, _ in warm_start_configs:
            config_hash = create_config_hash(config)
            assert config_hash in tuner.searched_configs_set

        # Check that warm start configs aren't in searchable configs (static mode)
        if not tuner.dynamic_sampling:
            for config, _ in warm_start_configs:
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

    def test_primary_estimator_error_not_nan(self, tuner):
        # Run a short tuning session
        tuner.tune(n_random_searches=15, max_iter=30, verbose=False)
        # Collect all primary_estimator_error values from trials
        errors = [trial.primary_estimator_error for trial in tuner.study.trials]
        print(errors)
        # Check that at least one is not None and not NaN
        assert any(
            (e is not None and not (isinstance(e, float) and (e != e))) for e in errors
        ), "At least one primary_estimator_error should be set and not NaN in the trials output."


class TestDynamicSamplingIntegration:
    """Integration tests for dynamic sampling using the main tune() method"""

    def test_dynamic_sampling_no_duplicate_evaluations(self, dynamic_tuner):
        """Integration test: Ensure no already-searched configurations are ever evaluated"""
        # Run a short tuning session (need at least 5 random searches for conformal phase)
        dynamic_tuner.tune(
            n_random_searches=5,
            max_iter=10,
            verbose=False,
        )

        # Verify all evaluated configurations are unique
        all_hashes = [
            create_config_hash(config) for config in dynamic_tuner.searched_configs
        ]
        assert len(all_hashes) == len(
            set(all_hashes)
        ), "Duplicate configurations were evaluated"

        # Verify we completed the expected number of trials
        assert len(dynamic_tuner.study.trials) == 10
        assert len(dynamic_tuner.searched_configs) == 10

    def test_dynamic_sampling_state_consistency_during_tuning(self, dynamic_tuner):
        """Integration test: Verify state consistency throughout the tuning process"""
        # Run tuning (need at least 5 random searches for conformal phase)
        dynamic_tuner.tune(
            n_random_searches=5,
            max_iter=8,
            verbose=False,
        )

        # Verify final state consistency
        assert len(dynamic_tuner.searched_configs) == len(
            dynamic_tuner.searched_performances
        )
        assert len(dynamic_tuner.searched_configs) == len(
            dynamic_tuner.searched_configs_set
        )
        assert len(dynamic_tuner.study.trials) == len(dynamic_tuner.searched_configs)

        # Verify all searched configs are in the set
        for config in dynamic_tuner.searched_configs:
            config_hash = create_config_hash(config)
            assert config_hash in dynamic_tuner.searched_configs_set

    def test_dynamic_sampling_reaches_target_iterations(self, dynamic_tuner):
        """Integration test: Verify dynamic sampling can reach target iterations beyond n_candidate_configurations"""
        target_iterations = 12  # More than n_candidate_configurations (5)

        dynamic_tuner.tune(
            n_random_searches=5,  # Need at least 5 for conformal phase
            max_iter=target_iterations,
            verbose=False,
        )

        # Should reach target iterations despite small candidate count
        assert len(dynamic_tuner.study.trials) == target_iterations
        assert len(dynamic_tuner.searched_configs) == target_iterations


class TestStaticSamplingIntegration:
    """Integration tests for static sampling using the main tune() method"""

    def test_static_sampling_no_duplicate_evaluations(self, static_tuner):
        """Integration test: Ensure no already-searched configurations are ever evaluated in static mode"""
        # Run tuning (need at least 5 random searches for conformal phase)
        static_tuner.tune(
            n_random_searches=5,
            max_iter=10,
            verbose=False,
        )

        # Verify all evaluated configurations are unique
        all_hashes = [
            create_config_hash(config) for config in static_tuner.searched_configs
        ]
        assert len(all_hashes) == len(
            set(all_hashes)
        ), "Duplicate configurations were evaluated"

    def test_static_sampling_with_warm_start_integration(
        self, mock_constant_objective_function, small_parameter_grid
    ):
        """Integration test: Verify static sampling with warm start configurations"""
        warm_start_configs = [
            ({"x": 0.5, "y": 2, "z": "A"}, 1.0),
            ({"x": 0.8, "y": 1, "z": "B"}, 2.0),
        ]

        tuner = ConformalTuner(
            objective_function=mock_constant_objective_function,
            search_space=small_parameter_grid,
            metric_optimization="minimize",
            n_candidate_configurations=8,
            dynamic_sampling=False,
            warm_start_configurations=warm_start_configs,
        )

        # Run tuning (need at least 5 random searches for conformal phase)
        tuner.tune(
            n_random_searches=5,
            max_iter=8,
            verbose=False,
        )

        # Verify warm start configs are included in final results
        assert len(tuner.study.trials) == 8
        assert len(tuner.searched_configs) == 8

        # Verify warm start configs are in the searched configs
        warm_start_hashes = {
            create_config_hash(config) for config, _ in warm_start_configs
        }
        searched_hashes = {
            create_config_hash(config) for config in tuner.searched_configs
        }
        assert warm_start_hashes.issubset(
            searched_hashes
        ), "Warm start configs missing from results"


class TestConfigurationSamplingIsolated:
    """Isolated unit tests for individual configuration sampling methods"""

    def test_sample_configurations_for_iteration_dynamic_count(self, dynamic_tuner):
        """Isolated test: _sample_configurations_for_iteration returns correct count in dynamic mode"""
        dynamic_tuner._initialize_tuning_resources()

        configs = dynamic_tuner._sample_configurations_for_iteration()
        assert len(configs) == dynamic_tuner.n_candidate_configurations

    def test_sample_configurations_for_iteration_static_count(self, static_tuner):
        """Isolated test: _sample_configurations_for_iteration returns correct count in static mode"""
        static_tuner._initialize_tuning_resources()

        configs = static_tuner._sample_configurations_for_iteration()
        # Should return all available configs (up to n_candidate_configurations)
        assert len(configs) <= static_tuner.n_candidate_configurations

    def test_update_search_state_isolated(self, dynamic_tuner):
        """Isolated test: _update_search_state correctly updates all data structures"""
        dynamic_tuner._initialize_tuning_resources()

        test_config = {"x": 0.5, "y": 2, "z": "A"}
        test_performance = 1.5

        initial_searched_count = len(dynamic_tuner.searched_configs)

        dynamic_tuner._update_search_state(test_config, test_performance)

        # Verify updates
        assert len(dynamic_tuner.searched_configs) == initial_searched_count + 1
        assert test_config in dynamic_tuner.searched_configs
        assert test_performance in dynamic_tuner.searched_performances

        config_hash = create_config_hash(test_config)
        assert config_hash in dynamic_tuner.searched_configs_set

    def test_get_tabularized_configs_isolated(self, dynamic_tuner):
        """Isolated test: _get_tabularized_configs correctly transforms configurations"""
        dynamic_tuner._initialize_tuning_resources()

        test_configs = [
            {"x": 0.5, "y": 2, "z": "A"},
            {"x": 0.8, "y": 1, "z": "B"},
        ]

        tabularized = dynamic_tuner._get_tabularized_configs(test_configs)

        # Should return numpy array with correct shape
        assert tabularized.shape[0] == len(test_configs)
        assert tabularized.shape[1] > 0  # Should have features


class TestConfigurationHashing:
    """Isolated unit tests for configuration hashing functionality"""

    def test_config_hash_consistency(self):
        """Test that identical configurations produce identical hashes"""
        config1 = {"x": 1.0, "y": 2, "z": "A"}
        config2 = {"x": 1.0, "y": 2, "z": "A"}
        config3 = {"z": "A", "y": 2, "x": 1.0}  # Different order

        hash1 = create_config_hash(config1)
        hash2 = create_config_hash(config2)
        hash3 = create_config_hash(config3)

        assert (
            hash1 == hash2 == hash3
        ), "Identical configurations should produce identical hashes"

    def test_config_hash_uniqueness(self):
        """Test that different configurations produce different hashes"""
        configs = [
            {"x": 1.0, "y": 2, "z": "A"},
            {"x": 1.0, "y": 2, "z": "B"},
            {"x": 1.0, "y": 3, "z": "A"},
            {"x": 2.0, "y": 2, "z": "A"},
        ]

        hashes = [create_config_hash(config) for config in configs]

        assert len(hashes) == len(
            set(hashes)
        ), "Different configurations should produce different hashes"

    def test_config_hash_type_handling(self):
        """Test that config hashing handles different data types correctly"""
        config_with_types = {
            "float_param": 1.5,
            "int_param": 42,
            "bool_param": True,
            "str_param": "test",
        }

        # Should not raise an exception
        hash_result = create_config_hash(config_with_types)
        assert isinstance(hash_result, str)
        assert len(hash_result) > 0
