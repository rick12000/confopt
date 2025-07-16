import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def pinball_loss(beta: float, theta: float, alpha: float) -> float:
    """Calculate pinball loss for conformal prediction adaptation.

    Args:
        beta: Empirical coverage probability (proportion of calibration scores >= test score)
        theta: Parameter (in DtACI context, this is α_t^i, the expert's alpha value)
        alpha: Global target miscoverage level

    Returns:
        Pinball loss value

    Mathematical Details:
        From the paper: ℓ(β_t, θ) := α(β_t - θ) - min{0, β_t - θ}

        This is the theoretical pinball loss used in the DtACI algorithm.
        In the algorithm, θ = α_t^i (expert's alpha value) and α is the global target.

        Beta represents the empirical coverage probability of the new observation.
        High beta (> α) means the observation is "easy" (low nonconformity relative to
        calibration) and intervals should be tightened. Low beta (< α) means the
        observation is "hard" (high nonconformity) and intervals should be widened.
    """
    return alpha * (beta - theta) - min(0, beta - theta)


class DtACI:
    """Dynamically-tuned Adaptive Conformal Inference.

    Implements the DtACI algorithm from Gibbs & Candès (2021) with K experts using
    different learning rates γ_k. Each expert maintains its own miscoverage level α_t^k,
    combined using exponential weighting based on pinball loss performance.

    Mathematical Components from the Paper:
    1. Pinball loss: ℓ(β_t, α_t^i) := α(β_t - α_t^i) - min{0, β_t - α_t^i}
    2. Weight update: w_t+1^i ∝ w_t^i × exp(-η × ℓ(β_t, α_t^i))
    3. Expert update: α_t+1^i = α_t^i + γ_i × (α - err_t^i)
    4. Selection: α_t via weighted average or random sampling
    5. Regularization: w_t+1^i = (1-σ)w̄_t^i + σ/k
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma_values: Optional[list[float]] = None,
        use_weighted_average: bool = True,
    ):
        """Initialize DtACI with theoretical parameters.

        Args:
            alpha: Target miscoverage level (α ∈ (0,1))
            gamma_values: Learning rates for each expert. If single value provided,
                functions as simple ACI. If None, uses conservative multi-expert defaults.
            use_weighted_average: If True, uses deterministic weighted average (Algorithm 2).
                If False, uses random sampling (Algorithm 1).
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        self.alpha = alpha
        self.alpha_t = alpha
        self.use_weighted_average = use_weighted_average

        if gamma_values is None:
            gamma_values = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

        if any(gamma <= 0 for gamma in gamma_values):
            raise ValueError("All gamma values must be positive")

        self.k = len(gamma_values)
        self.gamma_values = np.asarray(gamma_values)
        self.alpha_t_candidates = np.array([alpha] * self.k)

        # Theoretical parameters from Algorithm 1 in the paper
        self.interval = 500
        self.sigma = 1 / (2 * self.interval)
        self.eta = (
            np.sqrt(3 / self.interval)
            * np.sqrt(np.log(self.interval * self.k) + 2)
            / ((1 - alpha) ** 2 * alpha**2)
        )

        self.weights = np.ones(self.k) / self.k
        self.update_count = 0
        self.beta_history = []
        self.alpha_history = []
        self.weight_history = []

    def update(self, beta: float) -> float:
        """Update alpha values based on empirical coverage feedback.

        Implements Algorithm 1 from Gibbs & Candès (2021):
        1. Compute pinball losses for each expert
        2. Update expert weights using exponential weighting
        3. Update each expert's alpha using gradient step
        4. Sample final alpha from weight distribution

        Args:
            beta: Empirical coverage feedback (β_t ∈ [0,1])

        Returns:
            Updated miscoverage level α_t+1
        """
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        self.update_count += 1
        self.beta_history.append(beta)

        # Compute pinball losses for each expert
        # From paper: ℓ(β_t, α_t^i) where β_t is empirical coverage and α_t^i is expert's alpha
        losses = np.array(
            [
                pinball_loss(beta=beta, theta=alpha_val, alpha=self.alpha)
                for alpha_val in self.alpha_t_candidates
            ]
        )

        updated_weights = self.weights * np.exp(-self.eta * losses)
        sum_of_updated_weights = np.sum(updated_weights)
        self.weights = (1 - self.sigma) * updated_weights + (
            (self.sigma * sum_of_updated_weights) / self.k
        )

        # Update each expert's alpha using gradient step
        # err_indicators = 1 if breach (beta < alpha), 0 if coverage (beta >= alpha)
        err_indicators = (beta < self.alpha_t_candidates).astype(float)
        self.alpha_t_candidates = self.alpha_t_candidates + self.gamma_values * (
            self.alpha - err_indicators
        )
        self.alpha_t_candidates = np.clip(self.alpha_t_candidates, 0.001, 0.999)

        if np.sum(self.weights) > 0:
            normalized_weights = self.weights / np.sum(self.weights)
        else:
            normalized_weights = np.ones(self.k) / self.k
            logger.warning("All expert weights became zero, reverting to uniform")

        if self.use_weighted_average:
            # Deterministic weighted average (Algorithm 2)
            self.alpha_t = np.sum(normalized_weights * self.alpha_t_candidates)
        else:
            # Random sampling (Algorithm 1)
            chosen_idx = np.random.choice(self.k, p=normalized_weights)
            self.alpha_t = self.alpha_t_candidates[chosen_idx]

        self.alpha_history.append(self.alpha_t)
        self.weight_history.append(normalized_weights.copy())

        return self.alpha_t
