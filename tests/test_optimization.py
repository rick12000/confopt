from acho.optimization import derive_optimal_tuning_count, RuntimeTracker
import time
import pytest


def test_time_logger__return_runtime():
    dummy_tracker = RuntimeTracker()
    sleep_time = 5
    time.sleep(sleep_time)
    time_elapsed = dummy_tracker.return_runtime()
    assert sleep_time - 1 < round(time_elapsed) < sleep_time + 1


def test_time_logger__pause_runtime():
    dummy_tracker = RuntimeTracker()
    dummy_tracker.pause_runtime()
    sleep_time = 5
    time.sleep(sleep_time)
    dummy_tracker.resume_runtime()
    time_elapsed = dummy_tracker.return_runtime()
    assert time_elapsed < 1


@pytest.mark.parametrize("base_model_runtime", [1, 100])
@pytest.mark.parametrize("search_model_runtime", [1, 100])
@pytest.mark.parametrize("search_to_base_runtime_ratio", [0.5, 2])
@pytest.mark.parametrize("search_retraining_freq", [1, 10])
def test_get_n_search_estimator_tunings(
    base_model_runtime,
    search_model_runtime,
    search_to_base_runtime_ratio,
    search_retraining_freq,
):
    n_iterations = derive_optimal_tuning_count(
        base_model_runtime=base_model_runtime,
        search_model_runtime=search_model_runtime,
        search_to_base_runtime_ratio=search_to_base_runtime_ratio,
        search_retraining_freq=search_retraining_freq,
    )
    assert n_iterations >= 1
    assert isinstance(n_iterations, int)


def test_get_n_search_estimator_tunings__no_iterations():
    n_iterations = derive_optimal_tuning_count(
        base_model_runtime=1,
        search_model_runtime=1,
        search_to_base_runtime_ratio=1,
        search_retraining_freq=1,
    )
    assert n_iterations == 1