import numpy as np


def pinball_loss(beta, theta, alpha):
    return alpha * (beta - theta) - np.minimum(0, beta - theta)


class DtACI:
    def __init__(self, alpha=0.1, gamma_values=None):
        self.alpha = alpha
        self.alpha_t = alpha

        if gamma_values is None:
            gamma_values = [0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128]

        self.k = len(gamma_values)
        self.gamma_values = np.asarray(gamma_values)
        self.alpha_t_values = np.array([alpha] * len(gamma_values))

        self.interval = 500
        self.sigma = 1 / (2 * self.interval)
        self.eta = (
            (np.sqrt(3 / self.interval))
            * np.sqrt(np.log(self.interval * self.k) + 2)
            / ((1 - alpha) ** 2 * alpha**3)
        )

        self.weights = np.ones(self.k) / self.k

        # TODO: TEMP FOR PAPER
        self.error_history = []
        self.previous_chosen_idx = None

    def update(self, beta: float) -> float:
        losses = pinball_loss(beta=beta, theta=self.alpha_t_values, alpha=self.alpha)

        weights_bar = self.weights * np.exp(-self.eta * losses)
        sum_weights_bar = np.sum(weights_bar)

        self.weights = (1 - self.sigma) * weights_bar + (
            sum_weights_bar * self.sigma / self.k
        )
        self.weights = self.weights / np.sum(self.weights)

        errors = self.alpha_t_values > beta

        # TODO: TEMP FOR PAPER
        if self.previous_chosen_idx is not None:
            self.error_history.append(errors[self.previous_chosen_idx])

        self.alpha_t_values = np.clip(
            self.alpha_t_values + self.gamma_values * (self.alpha - errors), 0.01, 0.99
        )

        chosen_idx = np.random.choice(range(self.k), size=1, p=self.weights)[0]
        self.alpha_t = self.alpha_t_values[chosen_idx]

        # TODO: TEMP FOR PAPER
        self.previous_chosen_idx = chosen_idx

        return self.alpha_t
