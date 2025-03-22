import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from confopt.adaptation import ACI, DtACI


COVERAGE_TOLERANCE: float = 0.03


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


# Create fixtures for testing with regression-based conformal prediction
@pytest.fixture
def linear_data_stable():
    """
    Generate stable linear data with constant noise level.
    """
    np.random.seed(42)
    n = 500
    X = np.linspace(0, 10, n).reshape(-1, 1)
    y = 2 * X.flatten() + 5 + np.random.normal(0, 1, n)
    return X, y


@pytest.fixture
def linear_data_drift():
    """
    Generate linear data with distributional shift:
    - Increasing noise level
    - Change in relationship slope
    - Jump in relationship
    """
    np.random.seed(42)
    n = 500
    X = np.linspace(0, 10, n).reshape(-1, 1)

    # Create noise with increasing variance
    noise_level = np.linspace(0.5, 3, n)
    noise = np.random.normal(0, 1, n) * noise_level

    # Create y with changing relationships
    y = np.zeros(n)

    # First segment: y = 2x + 5
    first_segment = int(0.3 * n)
    y[:first_segment] = 2 * X[:first_segment].flatten() + 5 + noise[:first_segment]

    # Second segment: y = 3x + 2 (slope change)
    second_segment = int(0.6 * n)
    y[first_segment:second_segment] = (
        3 * X[first_segment:second_segment].flatten()
        + 2
        + noise[first_segment:second_segment]
    )

    # Third segment: y = 2.5x + 8 (jump and different slope)
    y[second_segment:] = 2.5 * X[second_segment:].flatten() + 8 + noise[second_segment:]

    return X, y


def calculate_beta_t(residual, cal_residuals):
    """
    Calculate beta_t as the percentile rank of the residual among the calibration residuals.

    Parameters:
    - residual: The residual of the current observation
    - cal_residuals: Array of residuals from the calibration set

    Returns:
    - beta_t: The percentile rank (0 to 1)
    """
    # Calculate what percentile the residual is in the calibration set
    return np.mean(cal_residuals >= residual)


# Test ACI and DtACI with regression-based conformal prediction
@pytest.mark.parametrize("target_alpha", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_regression_conformal_adaptation(
    linear_data_stable, linear_data_drift, target_alpha
):
    """Test ACI and DtACI with regression-based conformal prediction using rolling window."""

    # Test both tabular data and time series data
    for data_name, data in [
        ("stable_data", linear_data_stable),
        ("drift_data", linear_data_drift),
    ]:
        # Initialize methods
        aci = ACI(alpha=target_alpha, gamma=0.01)
        dtaci = DtACI(
            alpha=target_alpha, gamma_values=[0.01, 0.05], deterministic=False
        )

        # Define initial training window size
        initial_window = (
            30 if "data" in data_name else 20
        )  # smaller window for time series

        # Create lists to track breaches
        no_adapt_breaches = []
        aci_breaches = []
        dtaci_breaches = []

        X, y = data

        # Process data using expanding window
        for i in range(
            initial_window, len(X) - (0 if data_name == "time_series" else 1)
        ):
            # Use all data up to current point for training & calibration
            X_hist = X[: i - 1]
            y_hist = y[: i - 1]

            # Proper split: use 70% for training, 30% for calibration
            n_cal = max(int(len(X_hist) * 0.3), 5)  # Ensure minimum calibration points

            # Split historical data into train and calibration sets
            X_train, X_cal = X_hist[:-n_cal], X_hist[-n_cal:]
            y_train, y_cal = y_hist[:-n_cal], y_hist[-n_cal:]

            # The next point is our test point
            x_test = X[i].reshape(1, -1)
            y_test = y[i]

            # Train model on training data only
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Calculate residuals on calibration set (not training data)
            y_cal_pred = model.predict(X_cal)
            cal_residuals = np.abs(y_cal - y_cal_pred)

            # Make prediction for test point
            y_pred = model.predict(x_test)[0]

            # Calculate residual for this point
            residual = np.abs(y_test - y_pred)

            # Calculate beta_t (percentile of current residual)
            beta_t = calculate_beta_t(residual, cal_residuals)

            # 1. No adaptation (fixed alpha)
            fixed_quantile = np.quantile(cal_residuals, 1 - target_alpha)
            fixed_lower = y_pred - fixed_quantile
            fixed_upper = y_pred + fixed_quantile
            fixed_breach = not (fixed_lower <= y_test <= fixed_upper)
            no_adapt_breaches.append(int(fixed_breach))

            # 2. ACI
            aci_quantile = np.quantile(cal_residuals, 1 - aci.alpha_t)
            aci_lower = y_pred - aci_quantile
            aci_upper = y_pred + aci_quantile
            aci_breach = not (aci_lower <= y_test <= aci_upper)
            aci_breaches.append(int(aci_breach))

            # Update ACI
            aci.update(breach_indicator=int(aci_breach))

            # 3. DtACI - calculate breach indicators for each expert
            dtaci_breach_indicators = []
            for alpha in dtaci.alpha_t_values:
                expert_quantile = np.quantile(cal_residuals, 1 - alpha)
                expert_lower = y_pred - expert_quantile
                expert_upper = y_pred + expert_quantile
                expert_breach = not (expert_lower <= y_test <= expert_upper)
                dtaci_breach_indicators.append(int(expert_breach))

            # DtACI current interval
            dtaci_quantile = np.quantile(cal_residuals, 1 - dtaci.alpha_t)
            dtaci_lower = y_pred - dtaci_quantile
            dtaci_upper = y_pred + dtaci_quantile
            dtaci_breach = not (dtaci_lower <= y_test <= dtaci_upper)
            dtaci_breaches.append(int(dtaci_breach))

            # Update DtACI
            dtaci.update(beta_t=beta_t)

        # Calculate empirical coverage
        no_adapt_coverage = 1 - np.mean(no_adapt_breaches)
        aci_coverage = 1 - np.mean(aci_breaches)
        dtaci_coverage = 1 - np.mean(dtaci_breaches)

        target_coverage = 1 - target_alpha

        # Calculate errors
        no_adapt_error = abs(no_adapt_coverage - target_coverage)
        aci_error = abs(aci_coverage - target_coverage)

        # Print results
        # print(f"\nData: {data_name}, Target coverage: {target_coverage:.4f}")
        # print(f"No adaptation: {no_adapt_coverage:.4f}, error: {no_adapt_error:.4f}")
        # print(f"ACI: {aci_coverage:.4f}, error: {aci_error:.4f}")
        # print(f"DtACI: {dtaci_coverage:.4f}, error: {dtaci_error:.4f}")

        # Check coverage (with more tolerance for the drift and time series cases)
        data_tolerance = (
            COVERAGE_TOLERANCE
            if data_name == "stable_data"
            else COVERAGE_TOLERANCE * 1.5
        )

        # Assert coverage is within tolerance
        assert (
            abs(dtaci_coverage - target_coverage) < data_tolerance
        ), f"DtACI coverage error too large: {abs(dtaci_coverage - target_coverage):.4f}"

        # Check that ACI performs better than no adaptation
        assert (
            aci_error <= no_adapt_error * 1.1
        ), f"{data_name}: ACI error ({aci_error:.4f}) should be better than no adaptation ({no_adapt_error:.4f})"
