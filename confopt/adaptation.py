import numpy as np


def pinball_loss(beta, theta, alpha):
    """
    Calculate the pinball loss where:
    - beta: The percentile/rank of the observation (not binary breach)
    - theta: The predicted quantile level
    - alpha: The target coverage level
    """
    return alpha * (beta - theta) - np.minimum(0, beta - theta)


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
        self.alpha_t = alpha

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
    def __init__(self, alpha=0.1, gamma_values=None, deterministic=False):
        """
        Dynamically Tuned Adaptive Conformal Inference (DtACI).
        Implementation follows Algorithm 1 from Gradu et al. (2023).

        Parameters:
        - alpha: Target coverage level (1 - alpha is the desired coverage).
        - gamma_values: List of candidate step-size values {γᵢ}ᵏᵢ₌₁.
        - deterministic: If True, always select expert with highest weight.
        """
        super().__init__(alpha=alpha)

        # Set default values if not provided
        if gamma_values is None:
            gamma_values = [0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128]

        self.k = len(gamma_values)
        self.gamma_values = np.asarray(gamma_values)
        self.alpha_t_values = np.array([alpha] * len(gamma_values))
        self.deterministic = deterministic

        # Use properties for sigma and eta if not provided
        self.interval = 500
        self.sigma = 1 / (2 * self.interval)
        self.eta = (
            (np.sqrt(3 / self.interval))
            * np.sqrt(np.log(self.interval * self.k) + 2)
            / ((1 - alpha) ** 2 * alpha**3)
        )

        # Initialize log weights (using log space for numerical stability)
        self.log_weights = np.ones(self.k) / self.k  # Equal weights at start

        # The selected alpha_t for the current step
        self.chosen_idx = None
        self.alpha_t = alpha

    def update(self, beta_t):
        """
        Update using the DtACI algorithm with beta_t value and breach indicators.

        Parameters:
        - beta_t: The percentile/rank of the latest observation in the validation set
        - breaches: Binary breach indicators (1 if breached, 0 otherwise) for each expert

        Returns:
        - alpha_t: The new alpha_t value for the next step.
        """
        # Calculate pinball losses using beta_t
        losses = pinball_loss(beta=beta_t, theta=self.alpha_t_values, alpha=self.alpha)

        # Update log weights using pinball loss
        log_weights_bar = self.log_weights * np.exp(-self.eta * losses)
        sum_log_weights_bar = np.sum(log_weights_bar)

        # Apply smoothing
        self.log_weights = (1 - self.sigma) * log_weights_bar + (
            sum_log_weights_bar * self.sigma / self.k
        )

        # Normalize log weights
        self.log_weights = self.log_weights / np.sum(self.log_weights)

        errors = self.alpha_t_values > beta_t
        # Update alpha values for each expert using breach information
        self.alpha_t_values = np.clip(
            self.alpha_t_values + self.gamma_values * (self.alpha - errors), 0.01, 0.99
        )

        # Choose expert - either deterministically or probabilistically
        if self.deterministic:
            # Choose expert with highest weight
            self.chosen_idx = None
            self.alpha_t = (self.log_weights * self.alpha_t_values).sum()
        else:
            # Probabilistic selection based on weights
            self.chosen_idx = np.random.choice(
                range(self.k), size=1, p=self.log_weights
            )[0]
            self.alpha_t = self.alpha_t_values[self.chosen_idx]
        return self.alpha_t
