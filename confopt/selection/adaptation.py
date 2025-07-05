import numpy as np
import logging

logger = logging.getLogger(__name__)


def pinball_loss(beta: float, theta: float, alpha: float) -> float:
    """Calculate the pinball loss for conformal prediction adaptation.

    The pinball loss is a key component of the adaptive conformal inference
    algorithm, measuring the cost of miscoverage based on the asymmetric
    penalty structure inherent in conformal prediction.

    Args:
        beta: Empirical coverage (proportion of calibration scores >= test score).
            This represents the p-value of the conformity test.
        theta: Target coverage level (1 - alpha_level).
            This is the desired coverage probability.
        alpha: Miscoverage level used for asymmetric penalty weighting.
            Controls the relative cost of over vs under-coverage.

    Returns:
        Pinball loss value, always non-negative.

    Mathematical Details:
        L(β, θ, α) = α × max(θ - β, 0) + (1-α) × max(β - θ, 0)

        This asymmetric loss function penalizes:
        - Under-coverage (β < θ) with weight α
        - Over-coverage (β > θ) with weight (1-α)

        The asymmetry reflects that under-coverage is typically more costly
        than over-coverage in conformal prediction applications.

    References:
        Gibbs & Candès (2021). "Conformal Inference for Online Prediction
        with Arbitrary Distribution Shifts". Section 3.2.
    """
    under_coverage_penalty = alpha * max(theta - beta, 0)
    over_coverage_penalty = (1 - alpha) * max(beta - theta, 0)
    return under_coverage_penalty + over_coverage_penalty


class DtACI:
    """Adaptive Conformal Inference with Distribution-free Tracking (Dt-ACI).

    Implements the Dt-ACI algorithm from Gibbs & Candès (2021) for online
    conformal prediction under distribution shift. The algorithm adaptively
    adjusts miscoverage levels (alpha) based on empirical coverage feedback
    to maintain target coverage despite changing data distributions.

    The algorithm maintains multiple candidate alpha values with different
    step sizes (gamma values) and uses an exponential weighting scheme to
    select among them based on their pinball loss performance.

    Args:
        alpha: Target miscoverage level in (0, 1). Coverage = 1 - alpha.
        gamma_values: Learning rates for different alpha candidates.
            If None, uses default exponentially spaced values.

    Attributes:
        alpha: Original target miscoverage level.
        alpha_t: Current adapted miscoverage level at time t.
        k: Number of candidate alpha values (experts).
        gamma_values: Learning rates for gradient updates.
        alpha_t_values: Current values of all k alpha candidates.
        interval: Window size for regret analysis (T in paper).
        sigma: Mixing parameter for expert weights regularization.
        eta: Learning rate for exponential weights algorithm.
        weights: Current probability distribution over k experts.

    Mathematical Foundation:
        The algorithm follows these key steps at each time t:
        1. Receive empirical coverage β_t from conformal predictor
        2. Compute pinball losses L_t^i for each expert i
        3. Update expert weights using exponential weighting:
           w̃_t^i ∝ w_{t-1}^i × exp(-η × L_t^i)
        4. Apply regularization: w_t^i = (1-σ)w̃_t^i + σ/k
        5. Update alpha values: α_t^i ← α_{t-1}^i + γ^i(α - I_{β_t < α_{t-1}^i})
        6. Sample current alpha: α_t ~ w_t

    Coverage Guarantee:
        Under mild assumptions, the algorithm achieves regret bound:
        R_T ≤ O(√(T log(T·k)))

        This ensures asymptotic coverage convergence to the target level.

    References:
        Gibbs, I. & Candès, E. (2021). "Conformal Inference for Online
        Prediction with Arbitrary Distribution Shifts". Section 3.
    """

    def __init__(self, alpha: float = 0.1, gamma_values: list[float] = None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        self.alpha = alpha
        self.alpha_t = alpha

        if gamma_values is None:
            # Default values from paper: exponentially spaced learning rates
            gamma_values = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

        if any(gamma <= 0 for gamma in gamma_values):
            raise ValueError("All gamma values must be positive")

        self.k = len(gamma_values)
        self.gamma_values = np.asarray(gamma_values)
        self.alpha_t_values = np.array([alpha] * len(gamma_values))

        # Algorithm parameters following the paper
        self.interval = 500  # T in the paper
        self.sigma = 1 / (2 * self.interval)  # Regularization parameter

        # Learning rate for exponential weights (Equation 8 in paper)
        self.eta = (
            np.sqrt(3 / self.interval)
            * np.sqrt(np.log(self.interval * self.k) + 2)
            / ((1 - alpha) ** 2 * alpha**3)
        )

        # Initialize uniform weights over experts
        self.weights = np.ones(self.k) / self.k

    def update(self, beta: float) -> float:
        """Update alpha values based on empirical coverage feedback.

        Implements one step of the Dt-ACI algorithm, updating expert weights
        and alpha values based on the observed empirical coverage (beta).

        Args:
            beta: Empirical coverage at current time step. This is the fraction
                of calibration nonconformity scores >= current test score.
                Should be in [0, 1].

        Returns:
            Updated alpha_t value for use in next prediction interval.

        Mathematical Details:
            1. Compute target coverage: θ = 1 - α (desired coverage level)
            2. Calculate pinball losses for each expert i:
               L_t^i = pinball_loss(β_t, α_t^i, α)
            3. Update unnormalized weights:
               w̃_t^i = w_{t-1}^i × exp(-η × L_t^i)
            4. Apply mixing regularization:
               w_t^i = (1-σ) × w̃_t^i / ||w̃_t||_1 + σ/k
            5. Update alpha values using gradient step:
               α_t^i ← clip(α_{t-1}^i + γ^i × (α - I_{β_t < α_{t-1}^i}), ε, 1-ε)
            6. Sample new alpha: α_t ~ Categorical(w_t)

        Implementation Notes:
            - Alpha values are clipped to [0.01, 0.99] for numerical stability
            - The indicator I_{β_t < α_{t-1}^i} equals 1 when coverage is below target
            - Weights are normalized after exponential update and regularization

        Raises:
            ValueError: If beta is not in [0, 1].
        """
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        # Compute pinball losses for each expert
        # Note: target coverage is (1 - alpha_t_values) for each expert
        target_coverages = 1 - self.alpha_t_values
        losses = np.array(
            [
                pinball_loss(beta=beta, theta=target_cov, alpha=self.alpha)
                for target_cov in target_coverages
            ]
        )

        # Update expert weights using exponential weighting (Equation 7 in paper)
        unnormalized_weights = self.weights * np.exp(-self.eta * losses)

        # Apply mixing regularization (Equation 9 in paper)
        sum_unnormalized = np.sum(unnormalized_weights)
        if sum_unnormalized > 0:
            normalized_weights = unnormalized_weights / sum_unnormalized
        else:
            # Fallback to uniform if all weights become zero
            normalized_weights = np.ones(self.k) / self.k
            logger.warning("All expert weights became zero, reverting to uniform")

        self.weights = (1 - self.sigma) * normalized_weights + self.sigma / self.k

        # Update alpha values using gradient ascent (Algorithm 1, line 8)
        # The gradient step: α_t^i ← α_{t-1}^i + γ^i × (α - I_{β_t < α_{t-1}^i})
        coverage_indicators = (beta < self.alpha_t_values).astype(float)
        gradient_updates = self.gamma_values * (self.alpha - coverage_indicators)

        self.alpha_t_values = np.clip(
            self.alpha_t_values + gradient_updates,
            0.01,  # Lower bound for numerical stability
            0.99,  # Upper bound for numerical stability
        )

        # Sample current alpha from expert distribution
        chosen_idx = np.random.choice(self.k, p=self.weights)
        self.alpha_t = self.alpha_t_values[chosen_idx]

        return self.alpha_t
