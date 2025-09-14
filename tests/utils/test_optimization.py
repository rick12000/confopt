import pytest
from confopt.utils.optimization import FixedSearcherOptimizer, DecayingSearcherOptimizer


@pytest.fixture
def fixed_surrogate_tuner():
    """Fixture to create a FixedSurrogateTuner instance."""
    return FixedSearcherOptimizer(n_tuning_episodes=8, tuning_interval=6)


def test_fixed_surrogate_tuner_initialization():
    """Test initialization of FixedSurrogateTuner."""
    tuner = FixedSearcherOptimizer(tuning_interval=7)
    assert tuner.fixed_interval == 7


def test_fixed_surrogate_tuner_select_arm(fixed_surrogate_tuner):
    """Test that select_arm returns the fixed values."""
    arm = fixed_surrogate_tuner.select_arm()
    assert arm == (8, 6)


def test_fixed_surrogate_tuner_update(fixed_surrogate_tuner):
    """Test that update method doesn't change behavior."""
    fixed_surrogate_tuner.update(
        search_iter=10,
    )

    arm = fixed_surrogate_tuner.select_arm()
    assert arm == (8, 6)


@pytest.fixture
def decaying_tuner():
    """Fixture to create a DecayingSearcherOptimizer instance."""
    return DecayingSearcherOptimizer(
        n_tuning_episodes=10,
        initial_tuning_interval=2,
        decay_rate=0.5,
        decay_type="linear",
        max_tuning_interval=20,
    )


def test_decaying_tuner_initialization():
    """Test that the DecayingSearcherOptimizer initializes correctly."""
    tuner = DecayingSearcherOptimizer(
        initial_tuning_interval=3,
    )
    assert tuner.initial_tuning_interval == 3

    tuner = DecayingSearcherOptimizer(
        initial_tuning_interval=4,
    )
    assert tuner.initial_tuning_interval == 4


def test_decaying_tuner_invalid_decay_type():
    """Test that invalid decay_type raises ValueError."""
    with pytest.raises(ValueError, match="decay_type must be one of"):
        DecayingSearcherOptimizer(decay_type="invalid")


def test_decaying_tuner_linear_decay(decaying_tuner):
    """Test linear decay calculation."""
    # At iteration 0
    decaying_tuner.update(search_iter=0)
    arm = decaying_tuner.select_arm()
    assert arm[0] == 10  # n_tuning_episodes should remain constant
    assert arm[1] == 2  # initial_tuning_interval

    # At iteration 2: interval = 2 + 0.5 * 2 = 3, rounded to 3
    decaying_tuner.update(search_iter=2)
    arm = decaying_tuner.select_arm()
    assert arm[0] == 10
    assert arm[1] == 3

    # At iteration 10: interval = 2 + 0.5 * 10 = 7, rounded to 7
    decaying_tuner.update(search_iter=10)
    arm = decaying_tuner.select_arm()
    assert arm[0] == 10
    assert arm[1] == 7


def test_decaying_tuner_exponential_decay():
    """Test exponential decay calculation."""
    tuner = DecayingSearcherOptimizer(
        n_tuning_episodes=5,
        initial_tuning_interval=2,
        decay_rate=0.1,
        decay_type="exponential",
        max_tuning_interval=20,
    )

    # At iteration 0
    tuner.update(search_iter=0)
    arm = tuner.select_arm()
    assert arm[0] == 5
    assert arm[1] == 2  # initial_tuning_interval

    # At iteration 5: interval = 2 * (1.1)^5 ≈ 3.22, rounded to 3
    tuner.update(search_iter=5)
    arm = tuner.select_arm()
    assert arm[0] == 5
    assert arm[1] == 3


def test_decaying_tuner_logarithmic_decay():
    """Test logarithmic decay calculation."""
    tuner = DecayingSearcherOptimizer(
        n_tuning_episodes=8,
        initial_tuning_interval=2,
        decay_rate=2.0,
        decay_type="logarithmic",
        max_tuning_interval=20,
    )

    # At iteration 0
    tuner.update(search_iter=0)
    arm = tuner.select_arm()
    assert arm[0] == 8
    assert arm[1] == 2  # initial_tuning_interval

    # At iteration 4: interval = 2 + 2.0 * log(5) ≈ 5.22, rounded to 5
    tuner.update(search_iter=4)
    arm = tuner.select_arm()
    assert arm[0] == 8
    assert arm[1] == 5


def test_decaying_tuner_max_interval_cap(decaying_tuner):
    """Test that tuning interval is capped at max_tuning_interval."""
    # Set a very high iteration to exceed max_tuning_interval
    decaying_tuner.update(search_iter=100)
    arm = decaying_tuner.select_arm()
    assert arm[0] == 10
    assert arm[1] == 20  # Should be capped at max_tuning_interval
