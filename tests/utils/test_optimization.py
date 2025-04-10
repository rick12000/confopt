import pytest
import numpy as np
from confopt.utils.optimization import BayesianTuner, FixedSurrogateTuner


@pytest.fixture
def bayesian_tuner():
    """Fixture to create a BayesianTuner instance."""
    return BayesianTuner(
        max_tuning_count=10,
        max_tuning_interval=10,
        conformal_retraining_frequency=2,
        min_observations=5,
        exploration_weight=0.1,
        random_state=42,
    )


def test_bayesian_tuner_initialization():
    """Test that the BayesianTuner initializes correctly."""
    tuner = BayesianTuner(max_tuning_interval=6, conformal_retraining_frequency=3)
    assert tuner.valid_intervals == [3, 6]

    tuner = BayesianTuner(max_tuning_interval=2, conformal_retraining_frequency=3)
    assert tuner.valid_intervals == [3]


def test_bayesian_tuner_update_and_fit_model(bayesian_tuner):
    """Test updating the tuner with observations and fitting the model."""
    observations = [
        (0, 5, 2, 0.8, 0.2),
        (1, 3, 4, 0.7, 0.3),
        (2, 7, 6, 0.9, 0.4),
        (3, 2, 2, 0.6, 0.2),
        (4, 10, 8, 0.5, 0.5),
        (5, 4, 2, 0.7, 0.3),
    ]

    for search_iter, tuning_count, interval, reward, cost in observations:
        bayesian_tuner.update(
            arm=(tuning_count, interval),
            reward=reward,
            cost=cost,
            search_iter=search_iter,
        )

    assert len(bayesian_tuner.X_observed) == len(observations)
    assert len(bayesian_tuner.y_observed) == len(observations)
    assert bayesian_tuner.model_trained
    assert bayesian_tuner.current_iter == observations[-1][0]


def test_bayesian_tuner_select_arm_with_insufficient_data(bayesian_tuner):
    """Test arm selection with insufficient data (should return random arm)."""
    arm = bayesian_tuner.select_arm()
    assert 1 <= arm[0] <= bayesian_tuner.max_tuning_count
    assert arm[1] in bayesian_tuner.valid_intervals

    for i in range(3):  # Less than min_observations (5)
        bayesian_tuner.update(
            arm=(5, 2),
            reward=0.8,
            cost=0.2,
            search_iter=i,
        )

    arm = bayesian_tuner.select_arm()
    assert 1 <= arm[0] <= bayesian_tuner.max_tuning_count
    assert arm[1] in bayesian_tuner.valid_intervals


def test_bayesian_tuner_select_arm_with_sufficient_data(bayesian_tuner):
    """Test arm selection with sufficient data (should use the model)."""
    observations = [
        (0, 3, 2, 0.6, 0.3),
        (1, 5, 2, 0.8, 0.2),
        (2, 7, 2, 0.9, 0.15),
        (3, 3, 4, 0.6, 0.6),
        (4, 5, 4, 0.8, 0.4),
        (5, 7, 4, 0.9, 0.3),
    ]

    for search_iter, tuning_count, interval, reward, cost in observations:
        bayesian_tuner.update(
            arm=(tuning_count, interval),
            reward=reward,
            cost=cost,
            search_iter=search_iter,
        )

    assert bayesian_tuner.model_trained

    arm = bayesian_tuner.select_arm()
    assert 1 <= arm[0] <= bayesian_tuner.max_tuning_count
    assert arm[1] in bayesian_tuner.valid_intervals


def test_bayesian_tuner_expected_improvement(bayesian_tuner):
    """Test the expected improvement calculation."""
    mean = np.array([0.5, 0.6, 0.7])
    std = np.array([0.1, 0.2, 0.05])
    best_f = 0.6

    ei = bayesian_tuner._expected_improvement(mean, std, best_f)
    assert np.argmax(ei) == 2

    best_f = 0.8
    ei = bayesian_tuner._expected_improvement(mean, std, best_f)
    assert np.argmax(ei) == 1


@pytest.fixture
def fixed_surrogate_tuner():
    """Fixture to create a FixedSurrogateTuner instance."""
    return FixedSurrogateTuner(n_tuning_episodes=8, tuning_interval=6)


def test_fixed_surrogate_tuner_initialization():
    """Test initialization of FixedSurrogateTuner."""
    tuner = FixedSurrogateTuner(tuning_interval=7, conformal_retraining_frequency=3)
    assert tuner.fixed_interval == 6


def test_fixed_surrogate_tuner_select_arm(fixed_surrogate_tuner):
    """Test that select_arm returns the fixed values."""
    arm = fixed_surrogate_tuner.select_arm()
    assert arm == (8, 6)


def test_fixed_surrogate_tuner_update(fixed_surrogate_tuner):
    """Test that update method doesn't change behavior."""
    fixed_surrogate_tuner.update(
        arm=(5, 2),
        reward=0.8,
        cost=0.2,
        search_iter=10,
    )

    arm = fixed_surrogate_tuner.select_arm()
    assert arm == (8, 6)
