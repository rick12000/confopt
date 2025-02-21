import numpy as np


class BaseACI:
    def __init__(self, alpha=0.1, gamma=0.01):
        """
        Base class for Adaptive Conformal Inference (ACI).

        Parameters:
        - alpha: Target coverage level (1 - alpha is the desired coverage).
        - gamma: Step-size parameter for updating alpha_t.
        """
        self.alpha = alpha
        self.gamma = gamma
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
    def __init__(self, alpha=0.1, gamma=0.01):
        """
        Standard Adaptive Conformal Inference (ACI).

        Parameters:
        - alpha: Target coverage level (1 - alpha is the desired coverage).
        - gamma: Step-size parameter for updating alpha_t.
        """
        super().__init__(alpha, gamma)

    def update(self, breach_indicator):
        """
        Update the confidence level alpha_t using the standard ACI update rule.

        Parameters:
        - breach_indicator: 1 if the previous prediction breached its interval, 0 otherwise.

        Returns:
        - alpha_t: Updated confidence level.
        """
        # Update alpha_t using the standard ACI rule
        self.alpha_t += self.gamma * (self.alpha - breach_indicator)
        self.alpha_t = max(0.01, min(self.alpha_t, 0.99))
        return self.alpha_t


class DtACI(BaseACI):
    def __init__(self, alpha=0.1, gamma_candidates=None, eta=0.1, sigma=0.01):
        """
        Dynamically-Tuned Adaptive Conformal Intervals (DtACI).

        Parameters:
        - alpha (float): Target coverage level (1 - alpha is the desired coverage). Must be between 0 and 1.
        - gamma_candidates (list of float): List of candidate step sizes for the experts. Defaults to a predefined list.
        - eta (float): Learning rate for expert weights. Controls the magnitude of weight adjustments. Must be positive.
        - sigma (float): Exploration rate for expert weights. Small sigma encourages more reliance on the best experts. Must be in [0, 1].
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1.")
        if gamma_candidates is None:
            gamma_candidates = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]
        if any(g <= 0 for g in gamma_candidates):
            raise ValueError("All gamma candidates must be positive.")
        if eta <= 0:
            raise ValueError("eta (learning rate) must be positive.")
        if not (0 <= sigma <= 1):
            raise ValueError("sigma (exploration rate) must be in [0, 1].")

        super().__init__(alpha, gamma=None)  # gamma is not used in DtACI
        self.gamma_candidates = gamma_candidates
        self.eta = eta
        self.sigma = sigma

        # Initialize experts
        self.num_experts = len(self.gamma_candidates)
        self.alpha_t = (
            np.ones(self.num_experts) * alpha
        )  # Initial quantile estimates for each expert
        self.weights = (
            np.ones(self.num_experts) / self.num_experts
        )  # Uniform initial weights

    def update(self, breach_indicator):
        """
        Update the confidence level alpha_t using the DtACI update rule.

        Parameters:
        - breach_indicator (int): 1 if the previous prediction breached its interval, 0 otherwise.

        Returns:
        - float: Updated confidence level, calculated as a weighted average of the experts' estimates.
        """
        if breach_indicator not in [0, 1]:
            raise ValueError("breach_indicator must be either 0 or 1.")

        # Update each expert's alpha estimate based on the breach indicator
        for i in range(self.num_experts):
            self.alpha_t[i] += self.gamma_candidates[i] * (
                self.alpha - breach_indicator
            )

        # Update expert weights using the exponential weighting scheme
        losses = np.abs(
            self.alpha - breach_indicator
        )  # Pinball loss simplifies to breach indicator here
        self.weights *= np.exp(-self.eta * losses)

        # Normalize weights to prevent underflow or overflow
        self.weights = (1 - self.sigma) * self.weights / np.sum(
            self.weights
        ) + self.sigma / self.num_experts

        # Compute the final alpha_t as a weighted average of experts' alpha estimates
        final_alpha_t = np.dot(self.weights, self.alpha_t)

        # Ensure final_alpha_t stays within valid bounds [0, 1]
        final_alpha_t = np.clip(final_alpha_t, 0.01, 0.99)

        return final_alpha_t
