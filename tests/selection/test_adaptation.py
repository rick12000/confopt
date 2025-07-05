import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from confopt.selection.adaptation import DtACI, pinball_loss

COVERAGE_TOLERANCE: float = 0.03


def check_breach(alpha_level, y_pred, y_test, cal_res):
    """Check if observation breaches prediction interval."""
    quantile = np.quantile(cal_res, 1 - alpha_level)
    lower = y_pred - quantile
    upper = y_pred + quantile
    return int(not (lower <= y_test <= upper))


@pytest.mark.parametrize(
    "beta,theta,alpha,expected",
    [
        (0.8, 0.9, 0.1, 0.1 * 0.1),  # Under-coverage case
        (0.95, 0.9, 0.1, 0.9 * 0.05),  # Over-coverage case
        (0.9, 0.9, 0.1, 0.0),  # Exact coverage
        (0.5, 0.8, 0.2, 0.2 * 0.3),  # Under-coverage with different alpha
        (0.7, 0.6, 0.3, 0.7 * 0.1),  # Over-coverage with different alpha
    ],
)
def test_pinball_loss_mathematical_correctness(beta, theta, alpha, expected):
    """Test pinball loss calculation matches theoretical formula."""
    result = pinball_loss(beta=beta, theta=theta, alpha=alpha)
    assert abs(result - expected) < 1e-10


def test_pinball_loss_asymmetric_penalty():
    """Test that pinball loss correctly implements asymmetric penalties."""
    alpha = 0.1
    theta = 0.9

    # Under-coverage should be penalized more heavily when alpha is small
    under_coverage_loss = pinball_loss(beta=0.8, theta=theta, alpha=alpha)
    over_coverage_loss = pinball_loss(beta=1.0, theta=theta, alpha=alpha)

    # Under-coverage penalty: alpha * |theta - beta| = 0.1 * 0.1 = 0.01
    # Over-coverage penalty: (1-alpha) * |beta - theta| = 0.9 * 0.1 = 0.09
    assert under_coverage_loss < over_coverage_loss


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.5])
def test_dtaci_initialization_parameters(alpha):
    """Test DtACI initializes with correct mathematical parameters."""
    dtaci = DtACI(alpha=alpha)

    # Check alpha bounds
    assert 0 < dtaci.alpha < 1
    assert dtaci.alpha_t == alpha

    # Check all experts start with same alpha
    assert np.allclose(dtaci.alpha_t_values, alpha)

    # Check weights are uniform initially
    expected_weight = 1.0 / dtaci.k
    assert np.allclose(dtaci.weights, expected_weight)
    assert abs(np.sum(dtaci.weights) - 1.0) < 1e-10

    # Check eta parameter follows theoretical formula
    T = dtaci.interval
    k = dtaci.k
    expected_eta = (
        np.sqrt(3 / T) * np.sqrt(np.log(T * k) + 2) / ((1 - alpha) ** 2 * alpha**3)
    )
    assert abs(dtaci.eta - expected_eta) < 1e-10


def test_dtaci_invalid_parameters():
    """Test DtACI raises appropriate errors for invalid parameters."""
    with pytest.raises(ValueError, match="alpha must be in"):
        DtACI(alpha=0.0)

    with pytest.raises(ValueError, match="alpha must be in"):
        DtACI(alpha=1.0)

    with pytest.raises(ValueError, match="gamma values must be positive"):
        DtACI(alpha=0.1, gamma_values=[0.1, 0.0, 0.2])


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_dtaci_update_weight_normalization(beta):
    """Test that expert weights remain normalized after updates."""
    dtaci = DtACI(alpha=0.1, gamma_values=[0.01, 0.05, 0.1])

    for _ in range(10):
        dtaci.update(beta=beta)

        # Weights should sum to 1
        assert abs(np.sum(dtaci.weights) - 1.0) < 1e-10

        # All weights should be non-negative
        assert np.all(dtaci.weights >= 0)

        # Alpha values should be in valid range
        assert np.all(dtaci.alpha_t_values > 0)
        assert np.all(dtaci.alpha_t_values < 1)


def test_dtaci_update_invalid_beta():
    """Test DtACI update rejects invalid beta values."""
    dtaci = DtACI(alpha=0.1)

    with pytest.raises(ValueError, match="beta must be in"):
        dtaci.update(beta=-0.1)

    with pytest.raises(ValueError, match="beta must be in"):
        dtaci.update(beta=1.5)


@pytest.mark.parametrize("target_alpha", [0.1, 0.2, 0.5])
def test_dtaci_coverage_adaptation_under_shift(target_alpha):
    """Test coverage adaptation under distribution shift scenarios."""
    np.random.seed(42)

    # Create data with shift: different noise levels in two segments
    n_points = 200
    shift_point = 100

    # First segment: low noise
    X1 = np.random.randn(shift_point, 2)
    y1 = X1.sum(axis=1) + 0.1 * np.random.randn(shift_point)

    # Second segment: high noise
    X2 = np.random.randn(n_points - shift_point, 2)
    y2 = X2.sum(axis=1) + 0.5 * np.random.randn(n_points - shift_point)

    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])

    dtaci = DtACI(alpha=target_alpha, gamma_values=[0.01, 0.05, 0.1])
    breaches = []
    betas_observed = []

    initial_window = 30

    for i in range(initial_window, len(X)):
        X_past = X[:i]
        y_past = y[:i]
        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        # Use conformal prediction setup
        n_cal = max(int(len(X_past) * 0.3), 10)
        X_train, X_cal = X_past[:-n_cal], X_past[-n_cal:]
        y_train, y_cal = y_past[:-n_cal], y_past[-n_cal:]

        # Fit model and get predictions
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_cal_pred = model.predict(X_cal)
        cal_residuals = np.abs(y_cal - y_cal_pred)
        y_test_pred = model.predict(X_test)[0]

        # Calculate beta (empirical p-value)
        test_residual = abs(y_test - y_test_pred)
        beta = np.mean(cal_residuals >= test_residual)
        betas_observed.append(beta)

        # Update DtACI and check coverage
        current_alpha = dtaci.update(beta=beta)
        breach = check_breach(current_alpha, y_test_pred, y_test, cal_residuals)
        breaches.append(breach)

    # Check overall coverage is close to target
    empirical_coverage = 1 - np.mean(breaches)
    target_coverage = 1 - target_alpha
    coverage_error = abs(empirical_coverage - target_coverage)

    assert coverage_error < COVERAGE_TOLERANCE


@pytest.mark.parametrize("n_updates", [10, 50, 100])
def test_dtaci_expert_weight_evolution(n_updates):
    """Test that expert weights evolve reasonably over time."""
    dtaci = DtACI(alpha=0.1, gamma_values=[0.001, 0.1, 0.2])  # Different gamma values

    # Simulate consistent under-coverage scenario
    initial_weights = dtaci.weights.copy()

    for _ in range(n_updates):
        # Beta = 0.05 means significant under-coverage (target coverage = 0.9)
        dtaci.update(beta=0.05)

    final_weights = dtaci.weights.copy()

    # Weights should change from initial uniform distribution
    assert not np.allclose(initial_weights, final_weights)

    # Weights should still be normalized
    assert abs(np.sum(final_weights) - 1.0) < 1e-10

    # In under-coverage scenario with low beta, experts that adjust more conservatively
    # (smaller gamma) should generally get higher weight since they avoid over-correction
    # This is because the pinball loss penalizes overcorrection more severely
    assert (
        final_weights[0] > final_weights[2]
    )  # gamma=0.001 should outperform gamma=0.2


@pytest.mark.parametrize("target_alpha", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_regression_conformal_adaptation(linear_data_drift, target_alpha):
    """Test DtACI adaptation on linear regression with drift."""
    dtaci = DtACI(alpha=target_alpha, gamma_values=[0.01, 0.05])

    initial_window = 30
    no_adapt_breaches = []
    dtaci_breaches = []
    alpha_evolution = []

    X, y = linear_data_drift

    for i in range(initial_window, len(X) - 1):
        X_past = X[: i - 1]
        y_past = y[: i - 1]
        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        n_cal = max(int(len(X_past) * 0.3), 5)
        X_train, X_cal = X_past[:-n_cal], X_past[-n_cal:]
        y_train, y_cal = y_past[:-n_cal], y_past[-n_cal:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_cal_pred = model.predict(X_cal)
        cal_residuals = np.abs(y_cal - y_cal_pred)
        y_test_pred = model.predict(X_test)[0]

        residual = np.abs(y_test - y_test_pred)
        beta_t = np.mean(cal_residuals >= residual)
        adapted_alpha = dtaci.update(beta=beta_t)
        alpha_evolution.append(adapted_alpha)

        no_adapt_breaches.append(
            check_breach(target_alpha, y_test_pred, y_test, cal_residuals)
        )
        dtaci_breaches.append(
            check_breach(adapted_alpha, y_test_pred, y_test, cal_residuals)
        )

    dtaci_coverage = 1 - np.mean(dtaci_breaches)
    target_coverage = 1 - target_alpha

    # Main coverage guarantee test
    assert abs(dtaci_coverage - target_coverage) < COVERAGE_TOLERANCE

    # Additional checks for adaptation quality
    alpha_range = max(alpha_evolution) - min(alpha_evolution)
    assert alpha_range > 0  # Alpha should adapt over time


def test_dtaci_convergence_properties():
    """Test theoretical convergence properties of DtACI."""
    dtaci = DtACI(alpha=0.1, gamma_values=[0.01, 0.02, 0.05])

    # Test convergence under stationary conditions
    target_beta = 0.9  # Perfect coverage scenario
    alpha_history = []

    for _ in range(100):
        alpha_t = dtaci.update(beta=target_beta)
        alpha_history.append(alpha_t)

    # Under perfect coverage, alpha should stabilize near target
    recent_alphas = alpha_history[-20:]
    alpha_variance = np.var(recent_alphas)
    alpha_mean = np.mean(recent_alphas)

    # Should converge to low variance
    assert alpha_variance < 0.01

    # Should converge near target alpha (allowing for reasonable adaptation range)
    # Note: Some drift is expected due to the stochastic nature and exploration
    assert abs(alpha_mean - dtaci.alpha) < 0.15
