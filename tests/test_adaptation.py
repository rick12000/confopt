import numpy as np
import pytest
from confopt.adaptation import ACI, DtACI  # , pinball_loss

COVERAGE_TOLERANCE: float = 0.05


@pytest.mark.parametrize("breach", [True, False])
@pytest.mark.parametrize("alpha", [0.2, 0.8])
@pytest.mark.parametrize("gamma", [0.01, 0.1])
def test_update_adaptive_interval(breach, alpha, gamma):
    aci = ACI(alpha=alpha, gamma=gamma)
    stored_alpha = aci.alpha
    updated_alpha = aci.update(breach_indicator=breach)

    assert 0 < updated_alpha < 1
    if breach:
        assert updated_alpha <= alpha
    else:
        assert updated_alpha >= alpha

    assert stored_alpha == aci.alpha


# Test pinball loss
# def test_pinball_loss():
#     # Test when beta < alpha (under coverage)
#     beta, alpha, target_alpha = 0.05, 0.1, 0.1
#     loss = pinball_loss(beta, alpha, target_alpha)
#     assert loss == 0  # loss is 0 when alpha equals target_alpha

#     # Test when beta < alpha with alpha != target_alpha
#     beta, alpha, target_alpha = 0.05, 0.2, 0.1
#     loss = pinball_loss(beta, alpha, target_alpha)
#     assert loss == (alpha - target_alpha)  # loss when we're too conservative

#     # Test when beta >= alpha
#     beta, alpha, target_alpha = 0.15, 0.1, 0.1
#     loss = pinball_loss(beta, alpha, target_alpha)
#     assert loss == 0  # loss is 0 when alpha equals target_alpha

#     # Test when beta >= alpha with alpha != target_alpha
#     beta, alpha, target_alpha = 0.15, 0.05, 0.1
#     loss = pinball_loss(beta, alpha, target_alpha)
#     assert loss == (target_alpha - alpha)  # loss when we're too aggressive

# Improved fixtures for time series data
@pytest.fixture
def stable_data():
    """
    Generate data with stable distribution (no distribution shift).

    Returns:
        ndarray: true_values generated as sin of normal noise (stable variance)
    """
    np.random.seed(42)
    n = 1000

    # Generate with stable variance - sin of normal noise
    noise = np.random.normal(0, 1, n)
    true_values = np.sin(noise)

    return true_values


@pytest.fixture
def shifting_data():
    """
    Generate data with heteroskedastic variance that grows with n.

    Returns:
        ndarray: true_values with increasing variance as n increases
    """
    np.random.seed(42)
    n = 1000

    # Create noise with variance that grows with n
    n_steps = np.arange(n)
    noise = np.random.normal(0, 1, n)
    true_values = np.sin(
        n_steps**2 * noise / 100
    )  # Divide by 100 to moderate the growth

    return true_values


# ACI tests with time series data
@pytest.mark.parametrize("target_alpha", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_aci_adaptation(stable_data, shifting_data, target_alpha):
    for data_name, true_values in [
        ("stable_data", stable_data),
        ("shifting_data", shifting_data),
    ]:
        aci = ACI(alpha=target_alpha, gamma=0.01)

        alpha = target_alpha
        breaches = 0

        # Start after we have enough data to calculate meaningful quantiles
        for i in range(20, len(true_values)):
            # Use the quantile of previously observed values to make prediction
            prev_values = true_values[: i - 1]

            # Calculate differences between consecutive values to better model changes
            diffs = np.diff(prev_values)

            # Calculate prediction interval using quantiles of absolute differences
            quantile_value = np.quantile(np.abs(diffs), 1 - alpha)

            # Make interval wider by applying a safety factor
            interval_width = quantile_value * 1.5

            # Center interval on previous value
            interval_low = true_values[i - 1] - interval_width
            interval_high = true_values[i - 1] + interval_width

            # Check if true value falls within interval
            breach = not (interval_low <= true_values[i] <= interval_high)
            if breach:
                breaches += 1

            # Update alpha_t
            aci.update(breach_indicator=int(breach))

            # Update alpha for next iteration
            alpha = aci.alpha_t

        # Calculate empirical coverage
        empirical_coverage = 1 - (breaches / (len(true_values) - 20))
        target_coverage = 1 - target_alpha

        # Check if coverage is near target with custom error message
        assert (
            abs(empirical_coverage - target_coverage) < COVERAGE_TOLERANCE
        ), f"Coverage test failed for {data_name} with target_alpha={target_alpha}: expected {target_coverage:.4f}, got {empirical_coverage:.4f}"


# DtACI test with time series data
@pytest.mark.parametrize("target_alpha", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_dtaci_adaptation(stable_data, shifting_data, target_alpha):
    for data_name, true_values in [
        ("stable_data", stable_data),
        ("shifting_data", shifting_data),
    ]:
        dtaci = DtACI(alpha=target_alpha)

        breaches = 0
        for i in range(20, len(true_values)):
            # Use the history of values to construct intervals
            prev_values = true_values[: i - 1]

            # Calculate differences between consecutive values
            diffs = np.diff(prev_values)

            # Create separate interval for each expert based on their individual alphas
            breach_indicators = []
            for j, alpha in enumerate(dtaci.alpha_t_values):
                # Calculate interval width using quantiles of the differences
                quantile_value = np.quantile(np.abs(diffs), 1 - alpha)

                # Make interval wider with safety factor
                interval_width = quantile_value * 1.5

                # Create prediction interval
                interval_low = true_values[i - 1] - interval_width
                interval_high = true_values[i - 1] + interval_width

                # Check if true value falls within interval for this expert
                breach = not (interval_low <= true_values[i] <= interval_high)
                breach_indicators.append(int(breach))

            # For tracking overall performance, use the DtACI's current alpha_t
            dtaci_quantile = np.quantile(np.abs(diffs), 1 - dtaci.alpha_t)
            dtaci_width = dtaci_quantile * 1.5
            dtaci_low = true_values[i - 1] - dtaci_width
            dtaci_high = true_values[i - 1] + dtaci_width
            dtaci_breach = not (dtaci_low <= true_values[i] <= dtaci_high)
            if dtaci_breach:
                breaches += 1

            # Update DtACI with individual breach indicators from each expert
            dtaci.update(breach_indicators=breach_indicators)

        # Calculate empirical coverage
        empirical_coverage = 1 - (breaches / (len(true_values) - 20))
        target_coverage = 1 - target_alpha

        # Check if coverage is near target with custom error message
        assert (
            abs(empirical_coverage - target_coverage) < COVERAGE_TOLERANCE
        ), f"Coverage test failed for {data_name} with target_alpha={target_alpha}: expected {target_coverage:.4f}, got {empirical_coverage:.4f}"


# Comparative test to evaluate coverage performance
@pytest.mark.parametrize("target_alpha", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_adaptation_methods_comparison(shifting_data, target_alpha):
    """
    Test that DtACI has better coverage than ACI, which has better coverage
    than using no adaptation, especially in scenarios with distribution shift.
    """
    true_values = shifting_data
    target_coverage = 1 - target_alpha

    # Initialize methods with the same gamma value to verify convergence
    gamma = 0.01
    dtaci = DtACI(alpha=target_alpha)
    aci = ACI(alpha=target_alpha, gamma=gamma)

    # Track breaches for each method
    dtaci_breaches = 0
    aci_breaches = 0
    no_adapt_breaches = 0

    starting_training_samples = 20
    # Start after we have enough data to calculate meaningful quantiles
    for i in range(starting_training_samples, len(true_values)):
        prev_values = true_values[: i - 1]
        diffs = np.diff(prev_values)

        # 1. No Adaptation - Fixed alpha at target_alpha
        fixed_quantile = np.quantile(np.abs(diffs), 1 - target_alpha)
        fixed_width = fixed_quantile * 1.5
        fixed_low = true_values[i - 1] - fixed_width
        fixed_high = true_values[i - 1] + fixed_width
        fixed_breach = not (fixed_low <= true_values[i] <= fixed_high)
        if fixed_breach:
            no_adapt_breaches += 1

        # 2. ACI
        aci_quantile = np.quantile(np.abs(diffs), 1 - aci.alpha_t)
        aci_width = aci_quantile * 1.5
        aci_low = true_values[i - 1] - aci_width
        aci_high = true_values[i - 1] + aci_width
        aci_breach = not (aci_low <= true_values[i] <= aci_high)
        if aci_breach:
            aci_breaches += 1

        # Update ACI with the breach
        aci.update(breach_indicator=int(aci_breach))

        # 3. DtACI
        # Calculate breach indicators for each expert (just one in this case)
        breach_indicators = []
        for alpha in dtaci.alpha_t_values:
            expert_quantile = np.quantile(np.abs(diffs), 1 - alpha)
            expert_width = expert_quantile * 1.5
            expert_low = true_values[i - 1] - expert_width
            expert_high = true_values[i - 1] + expert_width
            expert_breach = not (expert_low <= true_values[i] <= expert_high)
            breach_indicators.append(int(expert_breach))

        # Calculate DtACI interval and check breach
        dtaci_quantile = np.quantile(np.abs(diffs), 1 - dtaci.alpha_t)
        dtaci_width = dtaci_quantile * 1.5
        dtaci_low = true_values[i - 1] - dtaci_width
        dtaci_high = true_values[i - 1] + dtaci_width
        dtaci_breach = not (dtaci_low <= true_values[i] <= dtaci_high)
        if dtaci_breach:
            dtaci_breaches += 1

        # Update DtACI with individual breach indicators
        dtaci.update(breach_indicators=breach_indicators)

    # Calculate coverage for each method
    samples_processed = len(true_values) - 20
    no_adapt_coverage = 1 - (no_adapt_breaches / samples_processed)
    aci_coverage = 1 - (aci_breaches / samples_processed)
    dtaci_coverage = 1 - (dtaci_breaches / samples_processed)

    # Check that DtACI coverage is better than ACI coverage
    dtaci_error = abs(dtaci_coverage - target_coverage)
    aci_error = abs(aci_coverage - target_coverage)
    no_adapt_error = abs(no_adapt_coverage - target_coverage)

    # Log coverage information
    print(f"\nTarget alpha: {target_alpha}, Target coverage: {target_coverage:.4f}")
    print(f"DtACI coverage: {dtaci_coverage:.4f}, error: {dtaci_error:.4f}")
    print(f"ACI coverage: {aci_coverage:.4f}, error: {aci_error:.4f}")
    print(
        f"No adaptation coverage: {no_adapt_coverage:.4f}, error: {no_adapt_error:.4f}"
    )

    # Assert that DtACI has better coverage than ACI which has better coverage than no adaptation
    # Using error relative to target coverage as the comparison metric
    assert (
        dtaci_error <= aci_error
    ), f"With target_alpha={target_alpha}: DtACI (error={dtaci_error:.4f}) should have better coverage than ACI (error={aci_error:.4f})"

    assert (
        aci_error <= no_adapt_error
    ), f"With target_alpha={target_alpha}: ACI (error={aci_error:.4f}) should have better coverage than no adaptation (error={no_adapt_error:.4f})"

    # Special check when DtACI has a single gamma equal to ACI's gamma
    if len(dtaci.gamma_values) == 1 and dtaci.gamma_values[0] == gamma:
        # They should converge to similar coverage (within a small margin)
        assert (
            abs(dtaci_coverage - aci_coverage) < 0.02
        ), f"With same gamma={gamma}: DtACI ({dtaci_coverage:.4f}) and ACI ({aci_coverage:.4f}) should converge to similar coverage"
