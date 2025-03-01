import numpy as np
import random


class BaseACI:
    def __init__(self, alpha=0.1):
        """
        Base class for Adaptive Conformal Inference (ACI).

        Parameters:
        - alpha: Target coverage level (1 - alpha is the desired coverage).
        - gamma: Step-size parameter for updating alpha_t.
        """
        self.alpha = alpha
        self.alpha_t = alpha  # Initial confidence level

    def update(self, breach_indicator):
        """
        Update the confidence level alpha_t based on the breach indicator.

        Parameters:
        - breach_indicator: 1 if the previous prediction breached its interval, 0 otherwise.

        Returns:
        - alpha_t: Updated confidence level.
        """
        raise NotImplementedError("Subclasses must implement the `update` method.")


class ACI(BaseACI):
    def __init__(self, alpha=0.1, gamma=0.1):
        """
        Standard Adaptive Conformal Inference (ACI).

        Parameters:
        - alpha: Target coverage level (1 - alpha is the desired coverage).
        - gamma: Step-size parameter for updating alpha_t.
        """
        super().__init__(alpha)
        self.gamma = gamma

    def update(self, breach_indicator):
        """
        Update the confidence level alpha_t using the standard ACI update rule.

        Parameters:
        - breach_indicator: 1 if the previous prediction breached its interval, 0 otherwise.

        Returns:
        - alpha_t: Updated confidence level.
        """

        # Update alpha_t using the standard ACI rule
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - breach_indicator)
        self.alpha_t = max(0.01, min(self.alpha_t, 0.99))
        return self.alpha_t


class DtACI(BaseACI):
    def __init__(
        self, alpha=0.1, gamma_values=None, initial_alphas=None, sigma=0.1, eta=1.0
    ):
        """
        Dynamically Tuned Adaptive Conformal Inference (DtACI).
        Implementation follows Algorithm 1 from Gradu et al. (2023).

        Parameters:
        - alpha: Target coverage level (1 - alpha is the desired coverage).
        - gamma_values: List of candidate step-size values {γᵢ}ᵏᵢ₌₁.
        - initial_alphas: List of starting points {αᵢ}ᵏᵢ₌₁.
        - sigma: Parameter for weight smoothing.
        - eta: Learning rate parameter.
        """
        super().__init__(alpha=alpha)

        # Set default values if not provided
        if gamma_values is None:
            gamma_values = [0.001, 0.01, 0.05, 0.1]
        if initial_alphas is None:
            initial_alphas = [alpha] * len(gamma_values)

        self.k = len(gamma_values)
        self.gamma_values = gamma_values
        self.alpha_t_values = initial_alphas.copy()
        self.sigma = sigma
        self.eta = eta

        # Initialize weights
        self.weights = [1.0] * self.k

        # The selected alpha_t for the current step
        self.chosen_idx = None
        self.alpha_t = self.sample_alpha_t()

    def sample_alpha_t(self):
        """Sample alpha_t based on the current weights."""
        # Calculate probabilities
        total_weight = sum(self.weights)
        probs = [w / total_weight for w in self.weights]

        # Sample an index based on probabilities
        self.chosen_idx = random.choices(range(self.k), weights=probs, k=1)[0]

        # Set the current alpha_t
        self.alpha_t = self.alpha_t_values[self.chosen_idx]

        return self.alpha_t

    def update(self, breach_indicators):
        """
        Update using the DtACI algorithm with individual breach indicators for each expert.

        Parameters:
        - breach_indicators: List of indicators (1 if breached, 0 otherwise) for each expert

        Returns:
        - alpha_t: The new alpha_t value for the next step.
        """
        if len(breach_indicators) != self.k:
            raise ValueError(
                f"Expected {self.k} breach indicators, got {len(breach_indicators)}"
            )

        # Use breach indicators directly as errors (err_i_t in the algorithm)
        errors = breach_indicators

        # Update weights with exponential weighting
        # w̄ᵗⁱ ← wᵗⁱ exp(-η ℓ(βₜ, αᵗⁱ))
        # Here the loss ℓ is just the breach indicator
        weights_bar = [
            w * np.exp(-self.eta * err) for w, err in zip(self.weights, errors)
        ]

        # Calculate total weight W_t
        total_weight_bar = sum(weights_bar)

        # Update weights for the next round with smoothing
        # wᵗ⁺¹ⁱ ← (1-σ)w̄ᵗⁱ + W_t σ/k
        self.weights = [
            (1 - self.sigma) * w_bar + total_weight_bar * self.sigma / self.k
            for w_bar in weights_bar
        ]

        # Update each alpha_t value for the experts
        # αᵗ⁺¹ⁱ = αᵗⁱ + γᵢ(α - errᵗⁱ)
        for i in range(self.k):
            self.alpha_t_values[i] += self.gamma_values[i] * (self.alpha - errors[i])
            # Ensure all alpha values stay within reasonable bounds
            self.alpha_t_values[i] = max(0.01, min(0.99, self.alpha_t_values[i]))

        # Sample the new alpha_t for the next step
        return self.sample_alpha_t()
