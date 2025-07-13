import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from confopt.selection.adaptation import DtACI, pinball_loss


class SimpleACI:
    """Simplified ACI implementation from Gibbs & Candès (2021) paper.

    This implements the basic adaptive conformal inference algorithm with the simple update:
    α_{t+1} = α_t + γ(α - err_t)

    where err_t = 1 if β_t < α_t (breach), 0 if β_t ≥ α_t (coverage).
    This follows the exact formula from equation (2) in the paper.
    This is used only for testing equivalence with DTACI when using a single gamma value.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        """Initialize Simple ACI.

        Args:
            alpha: Target miscoverage level (α ∈ (0,1))
            gamma: Learning rate for alpha updates
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self.alpha_history = []

    def update(self, beta: float) -> float:
        """Update alpha based on empirical coverage feedback.

        Args:
            beta: Empirical coverage (proportion of calibration scores >= test score)

        Returns:
            Updated miscoverage level α_t+1
        """
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        # Convert beta to error indicator: err_t = 1 if breach (beta < alpha_t), 0 if coverage
        err_t = float(beta < self.alpha_t)

        # Simple ACI update from paper: α_{t+1} = α_t + γ(α - err_t)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err_t)
        self.alpha_t = np.clip(self.alpha_t, 0.001, 0.999)

        self.alpha_history.append(self.alpha_t)
        return self.alpha_t


def run_dtaci_performance_test(X, y, target_alpha, gamma_values=None):
    """Helper function to run DtACI performance tests and return metrics."""
    if gamma_values is None:
        gamma_values = [0.01, 0.05, 0.1]

    dtaci = DtACI(alpha=target_alpha, gamma_values=gamma_values)
    breaches = []
    alpha_evolution = []
    initial_window = 30

    for i in range(initial_window, len(X)):
        X_past = X[:i]
        y_past = y[:i]
        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        n_cal = max(int(len(X_past) * 0.3), 10)
        X_train, X_cal = X_past[:-n_cal], X_past[-n_cal:]
        y_train, y_cal = y_past[:-n_cal], y_past[-n_cal:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_cal_pred = model.predict(X_cal)
        cal_residuals = np.abs(y_cal - y_cal_pred)
        y_test_pred = model.predict(X_test)[0]

        test_residual = abs(y_test - y_test_pred)
        beta = np.mean(cal_residuals >= test_residual)

        current_alpha = dtaci.update(beta=beta)
        alpha_evolution.append(current_alpha)

        # Check breach
        quantile = np.quantile(cal_residuals, 1 - current_alpha)
        lower = y_test_pred - quantile
        upper = y_test_pred + quantile
        breach = int(not (lower <= y_test <= upper))
        breaches.append(breach)

    coverage = 1 - np.mean(breaches)
    target_coverage = 1 - target_alpha
    coverage_error = abs(coverage - target_coverage)

    return {
        "coverage_error": coverage_error,
        "alpha_variance": np.var(alpha_evolution),
        "alpha_range": max(alpha_evolution) - min(alpha_evolution),
        "alpha_evolution": alpha_evolution,
    }


@pytest.mark.parametrize("gamma", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("target_alpha", [0.1, 0.2])
def test_dtaci_simple_aci_equivalence(gamma, target_alpha):
    """Test that DTACI with single gamma produces identical results to simple ACI."""
    np.random.seed(42)

    # Initialize both algorithms with same parameters
    dtaci = DtACI(alpha=target_alpha, gamma_values=[gamma], use_weighted_average=True)
    simple_aci = SimpleACI(alpha=target_alpha, gamma=gamma)

    # Test with sequence of beta values
    beta_sequence = [0.85, 0.92, 0.88, 0.95, 0.80, 0.75, 0.93, 0.87, 0.91, 0.82]

    dtaci_alphas = []
    simple_aci_alphas = []

    for beta in beta_sequence:
        dtaci_alpha = dtaci.update(beta=beta)
        simple_aci_alpha = simple_aci.update(beta=beta)

        dtaci_alphas.append(dtaci_alpha)
        simple_aci_alphas.append(simple_aci_alpha)

    # Alpha updates should be identical
    assert np.allclose(dtaci_alphas, simple_aci_alphas, atol=1e-12)

    # Alpha histories should be identical
    assert np.allclose(dtaci.alpha_history, simple_aci.alpha_history, atol=1e-12)

    # Final alpha values should be identical
    assert abs(dtaci.alpha_t - simple_aci.alpha_t) < 1e-12


def test_simple_aci_basic_functionality():
    """Test basic functionality of SimpleACI class."""
    aci = SimpleACI(alpha=0.1, gamma=0.01)

    # Test initialization
    assert aci.alpha == 0.1
    assert aci.gamma == 0.01
    assert aci.alpha_t == 0.1
    assert len(aci.alpha_history) == 0

    # Test update with breach (beta < alpha_t)
    alpha_new = aci.update(beta=0.05)  # breach, err_t = 1
    expected_alpha = 0.1 + 0.01 * (0.1 - 1)  # 0.1 + 0.01 * (-0.9) = 0.091
    assert abs(alpha_new - expected_alpha) < 1e-12
    assert len(aci.alpha_history) == 1

    # Test update with coverage (beta >= alpha_t)
    alpha_new = aci.update(beta=0.95)  # coverage, err_t = 0
    expected_alpha = expected_alpha + 0.01 * (0.1 - 0)  # 0.091 + 0.01 * 0.1 = 0.092
    assert abs(alpha_new - expected_alpha) < 1e-12
    assert len(aci.alpha_history) == 2


def test_simple_aci_parameter_validation():
    """Test parameter validation for SimpleACI."""
    # Test invalid alpha
    with pytest.raises(ValueError, match="alpha must be in"):
        SimpleACI(alpha=0.0)

    with pytest.raises(ValueError, match="alpha must be in"):
        SimpleACI(alpha=1.0)

    # Test invalid gamma
    with pytest.raises(ValueError, match="gamma must be positive"):
        SimpleACI(alpha=0.1, gamma=0.0)

    with pytest.raises(ValueError, match="gamma must be positive"):
        SimpleACI(alpha=0.1, gamma=-0.01)

    # Test invalid beta in update
    aci = SimpleACI(alpha=0.1, gamma=0.01)
    with pytest.raises(ValueError, match="beta must be in"):
        aci.update(beta=-0.1)

    with pytest.raises(ValueError, match="beta must be in"):
        aci.update(beta=1.1)


def test_dtaci_simple_aci_comprehensive_equivalence():
    """Comprehensive test showing DTACI and SimpleACI produce identical results with same gamma."""
    np.random.seed(42)

    # Test parameters
    target_alpha = 0.1
    gamma = 0.05

    # Initialize both algorithms
    dtaci = DtACI(alpha=target_alpha, gamma_values=[gamma], use_weighted_average=True)
    simple_aci = SimpleACI(alpha=target_alpha, gamma=gamma)

    # Generate synthetic data for testing
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)

    # Track results
    dtaci_alphas = []
    simple_aci_alphas = []
    dtaci_coverage = []
    simple_aci_coverage = []

    # Simulate online conformal prediction
    for i in range(30, n_samples):
        # Split data
        X_past = X[:i]
        y_past = y[:i]
        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        # Use simple train/calibration split
        n_cal = 20
        X_train, X_cal = X_past[:-n_cal], X_past[-n_cal:]
        y_train, y_cal = y_past[:-n_cal], y_past[-n_cal:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_cal_pred = model.predict(X_cal)
        cal_residuals = np.abs(y_cal - y_cal_pred)
        y_test_pred = model.predict(X_test)[0]
        test_residual = abs(y_test - y_test_pred)

        # Compute beta (empirical coverage)
        beta = np.mean(cal_residuals >= test_residual)

        # Update both algorithms
        dtaci_alpha = dtaci.update(beta=beta)
        simple_aci_alpha = simple_aci.update(beta=beta)

        dtaci_alphas.append(dtaci_alpha)
        simple_aci_alphas.append(simple_aci_alpha)

        # Check coverage for both methods
        dtaci_quantile = np.quantile(cal_residuals, 1 - dtaci_alpha)
        simple_aci_quantile = np.quantile(cal_residuals, 1 - simple_aci_alpha)

        dtaci_covered = abs(y_test - y_test_pred) <= dtaci_quantile
        simple_aci_covered = abs(y_test - y_test_pred) <= simple_aci_quantile

        dtaci_coverage.append(dtaci_covered)
        simple_aci_coverage.append(simple_aci_covered)

    # Verify exact equivalence
    assert np.allclose(dtaci_alphas, simple_aci_alphas, atol=1e-12)
    assert np.array_equal(dtaci_coverage, simple_aci_coverage)

    # Verify coverage performance
    dtaci_empirical_coverage = np.mean(dtaci_coverage)
    simple_aci_empirical_coverage = np.mean(simple_aci_coverage)
    target_coverage = 1 - target_alpha

    assert abs(dtaci_empirical_coverage - simple_aci_empirical_coverage) < 1e-12
    # Both should achieve reasonable coverage
    assert abs(dtaci_empirical_coverage - target_coverage) < 0.1
    assert abs(simple_aci_empirical_coverage - target_coverage) < 0.1


@pytest.mark.parametrize(
    "beta,theta,alpha,expected",
    [
        (0.8, 0.9, 0.1, 0.09),
        (0.95, 0.9, 0.1, 0.005),
        (0.9, 0.9, 0.1, 0.0),
        (0.5, 0.8, 0.2, 0.24),
        (0.7, 0.6, 0.3, 0.03),
    ],
)
def test_pinball_loss_mathematical_correctness(beta, theta, alpha, expected):
    """Test pinball loss calculation matches theoretical formula from paper."""
    result = pinball_loss(beta=beta, theta=theta, alpha=alpha)
    assert abs(result - expected) < 1e-10


def test_pinball_loss_asymmetric_penalty():
    """Test pinball loss correctly implements asymmetric penalty structure."""
    alpha = 0.1
    theta = 0.9

    # Under-coverage case (beta < theta)
    under_coverage_loss = pinball_loss(beta=0.8, theta=theta, alpha=alpha)
    # Over-coverage case (beta > theta)
    over_coverage_loss = pinball_loss(beta=1.0, theta=theta, alpha=alpha)

    # Under-coverage: ℓ(0.8, 0.9) = 0.1*(0.8-0.9) - min{0, 0.8-0.9} = -0.01 - (-0.1) = 0.09
    # Over-coverage: ℓ(1.0, 0.9) = 0.1*(1.0-0.9) - min{0, 1.0-0.9} = 0.01 - 0 = 0.01
    assert abs(under_coverage_loss - 0.09) < 1e-10
    assert abs(over_coverage_loss - 0.01) < 1e-10
    # Under-coverage should be penalized more than over-coverage
    assert under_coverage_loss > over_coverage_loss


def test_pinball_loss_properties():
    """Test general mathematical properties of pinball loss function."""
    alpha = 0.1

    # Test non-negativity and zero at equality
    for beta in [0.0, 0.3, 0.5, 0.7, 1.0]:
        for theta in [0.1, 0.4, 0.6, 0.9]:
            loss = pinball_loss(beta, theta, alpha)
            assert loss >= 0

            if abs(beta - theta) < 1e-10:
                assert abs(loss) < 1e-10


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.5])
def test_dtaci_initialization_parameters(alpha):
    """Test DtACI initializes with correct theoretical parameters."""
    dtaci = DtACI(alpha=alpha)

    # Check theoretical parameter formulas
    expected_eta = (
        np.sqrt(3 / dtaci.interval)
        * np.sqrt(np.log(dtaci.interval * dtaci.k) + 2)
        / ((1 - alpha) ** 2 * alpha**2)
    )
    expected_sigma = 1 / (2 * dtaci.interval)

    assert abs(dtaci.eta - expected_eta) < 1e-10
    assert abs(dtaci.sigma - expected_sigma) < 1e-12
    assert np.allclose(dtaci.alpha_t_candidates, alpha)
    assert np.allclose(dtaci.weights, 1.0 / dtaci.k)
    assert abs(np.sum(dtaci.weights) - 1.0) < 1e-10


def test_dtaci_invalid_parameters():
    """Test DtACI raises appropriate errors for invalid parameters."""
    with pytest.raises(ValueError, match="alpha must be in"):
        DtACI(alpha=0.0)

    with pytest.raises(ValueError, match="alpha must be in"):
        DtACI(alpha=1.0)

    with pytest.raises(ValueError, match="gamma values must be positive"):
        DtACI(alpha=0.1, gamma_values=[0.1, 0.0, 0.2])


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_dtaci_update_weight_normalization(beta, dtaci_instance):
    """Test that expert weights remain valid and probabilities can be computed."""
    for _ in range(10):
        dtaci_instance.update(beta=beta)
        # Weights should be non-negative but not necessarily normalized
        assert np.all(dtaci_instance.weights >= 0)
        # Should be able to compute valid probabilities
        weight_sum = np.sum(dtaci_instance.weights)
        assert (
            weight_sum > 0
        ), "Weight sum should be positive for probability computation"
        probabilities = dtaci_instance.weights / weight_sum
        assert abs(np.sum(probabilities) - 1.0) < 1e-10
        assert np.all(probabilities >= 0)
        # Alpha values should remain in valid range
        assert np.all(dtaci_instance.alpha_t_candidates > 0)
        assert np.all(dtaci_instance.alpha_t_candidates < 1)


def test_dtaci_theoretical_weight_updates():
    """Test that weight updates follow theoretical exponential weighting scheme."""
    dtaci = DtACI(alpha=0.1, gamma_values=[0.01, 0.05])

    initial_weights = dtaci.weights.copy()
    initial_alphas = dtaci.alpha_t_candidates.copy()

    beta = 0.85
    dtaci.update(beta=beta)

    # Manually compute expected weight update following the paper's approach
    losses = np.array(
        [
            pinball_loss(beta=beta, theta=alpha_val, alpha=dtaci.alpha)
            for alpha_val in initial_alphas
        ]
    )

    updated_weights = initial_weights * np.exp(-dtaci.eta * losses)
    sum_of_updated_weights = np.sum(updated_weights)
    expected_regularized = (1 - dtaci.sigma) * updated_weights + (
        (dtaci.sigma * sum_of_updated_weights) / dtaci.k
    )

    assert np.allclose(dtaci.weights, expected_regularized, atol=1e-12)


def test_dtaci_expert_alpha_updates():
    """Test expert alpha values are updated correctly according to theoretical formula."""
    dtaci = DtACI(alpha=0.1, gamma_values=[0.01, 0.05])

    initial_alphas = dtaci.alpha_t_candidates.copy()
    beta = 0.85
    dtaci.update(beta=beta)

    # Verify alpha updates follow: α_t+1^i = α_t^i + γ_i * (α - err_t^i)
    for i, (initial_alpha, gamma) in enumerate(zip(initial_alphas, dtaci.gamma_values)):
        err_indicator = float(beta < initial_alpha)
        expected_alpha = initial_alpha + gamma * (dtaci.alpha - err_indicator)
        expected_alpha = np.clip(expected_alpha, 0.001, 0.999)

        assert abs(dtaci.alpha_t_candidates[i] - expected_alpha) < 1e-12


def test_dtaci_both_selection_methods():
    """Test that both random sampling and weighted average methods work correctly."""
    np.random.seed(42)
    target_alpha = 0.1

    for use_weighted_average in [True, False]:
        dtaci = DtACI(
            alpha=target_alpha,
            gamma_values=[0.01, 0.05],
            use_weighted_average=use_weighted_average,
        )

        # Test with series of beta values
        betas = [0.85, 0.92, 0.88, 0.95, 0.80]
        alphas = [dtaci.update(beta=beta) for beta in betas]

        # Both methods should produce valid alphas
        assert all(0.001 <= alpha <= 0.999 for alpha in alphas)
        # Should show adaptation behavior
        assert len(set(np.round(alphas, 6))) > 1


def test_dtaci_convergence_under_stationary_conditions():
    """Test DtACI behavior under stationary conditions."""
    dtaci = DtACI(alpha=0.1, gamma_values=[0.01, 0.02, 0.05])

    # Test under conditions where target coverage is achieved
    # Use a beta value that should lead to equilibrium near the target alpha
    target_beta = 0.1  # This should lead to equilibrium around alpha = 0.1
    alpha_history = []

    for _ in range(500):
        alpha_t = dtaci.update(beta=target_beta)
        alpha_history.append(alpha_t)

    # Under stationary conditions, alpha should be relatively stable
    recent_alphas = alpha_history[-100:]
    alpha_variance = np.var(recent_alphas)
    alpha_mean = np.mean(recent_alphas)

    assert alpha_variance < 0.01
    # With beta = alpha, the algorithm should converge to a value close to alpha
    assert abs(alpha_mean - dtaci.alpha) < 0.1


def test_dtaci_algorithm_behavior():
    """Test comprehensive DtACI algorithm behavior and theoretical correctness."""
    dtaci = DtACI(alpha=0.1, gamma_values=[0.01, 0.05])

    # Test algorithm components work as specified
    betas = [0.85, 0.92, 0.88, 0.95, 0.80]

    for beta in betas:
        prev_weights = dtaci.weights.copy()
        prev_alphas = dtaci.alpha_t_candidates.copy()

        alpha_t = dtaci.update(beta=beta)

        # Verify weights remain valid (non-negative and positive sum)
        assert np.all(dtaci.weights >= 0)
        assert np.sum(dtaci.weights) > 0

        # Verify weights change when losses differ
        losses = [
            pinball_loss(beta, alpha_val, dtaci.alpha) for alpha_val in prev_alphas
        ]
        if not np.allclose(losses, losses[0]):
            assert not np.allclose(dtaci.weights, prev_weights, atol=1e-10)

        # Verify alpha values are in valid range
        assert np.all(dtaci.alpha_t_candidates >= 0.001)
        assert np.all(dtaci.alpha_t_candidates <= 0.999)
        assert 0.001 <= alpha_t <= 0.999

    # Test algorithm adaptation over time
    alphas_sequence = [dtaci.update(beta=beta) for beta in betas]
    unique_alphas = len(set(np.round(alphas_sequence, 6)))
    assert unique_alphas > 1


@pytest.mark.parametrize("target_alpha", [0.1, 0.2, 0.5])
def test_dtaci_moderate_shift_performance(moderate_shift_data, target_alpha):
    """Test DtACI performance under moderate distribution shift."""
    X, y = moderate_shift_data
    results = run_dtaci_performance_test(X, y, target_alpha)

    tolerance = 0.05

    assert results["coverage_error"] < tolerance
    # Should show adaptation behavior
    assert results["alpha_variance"] > 0.00001
    assert results["alpha_range"] > 0.0001


@pytest.mark.parametrize("target_alpha", [0.1, 0.2, 0.5])
def test_dtaci_high_shift_performance(high_shift_data, target_alpha):
    """Test DtACI performance under high distribution shift."""
    X, y = high_shift_data
    results = run_dtaci_performance_test(X, y, target_alpha)

    tolerance = 0.05

    assert results["coverage_error"] < tolerance
    # Should show significant adaptation behavior under high shift
    assert results["alpha_variance"] > 0.00001
    assert results["alpha_range"] > 0.005
