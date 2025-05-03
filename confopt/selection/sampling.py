from typing import Optional, List, Literal
import numpy as np
from confopt.selection.adaptation import DtACI
import warnings


class PessimisticLowerBoundSampler:
    def __init__(
        self,
        interval_width: float = 0.8,
        adapter: Optional[Literal["DtACI"]] = None,
    ):
        self.interval_width = interval_width

        self.alpha = 1 - interval_width
        self.adapter = self._initialize_adapter(adapter)

    def _initialize_adapter(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[DtACI]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return DtACI(alpha=self.alpha, gamma_values=[0.05, 0.01, 0.1])
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return [self.alpha]

    def update_interval_width(self, beta: float) -> None:
        if self.adapter is not None:
            self.alpha = self.adapter.update(beta=beta)
        else:
            warnings.warn(
                "'update_interval_width()' method was called, but no adapter was initialized."
            )


class LowerBoundSampler(PessimisticLowerBoundSampler):
    def __init__(
        self,
        interval_width: float = 0.8,
        adapter: Optional[Literal["DtACI"]] = None,
        beta_decay: Optional[
            Literal[
                "inverse_square_root_decay",
                "logarithmic_decay",
            ]
        ] = "logarithmic_decay",
        c: float = 1,
        beta_max: float = 10,
    ):
        super().__init__(interval_width, adapter)
        self.beta_decay = beta_decay
        self.c = c
        self.t = 1
        self.beta = 1
        self.beta_max = beta_max
        self.mu_max = float("-inf")

    def update_exploration_step(self):
        self.t += 1
        if self.beta_decay == "inverse_square_root_decay":
            self.beta = np.sqrt(self.c / self.t)
        elif self.beta_decay == "logarithmic_decay":
            self.beta = np.sqrt((self.c * np.log(self.t)) / self.t)
        elif self.beta_decay is None:
            self.beta = 1
        else:
            raise ValueError(
                "beta_decay must be 'inverse_square_root_decay', 'logarithmic_decay', or None."
            )


class ThompsonSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI"]] = None,
        enable_optimistic_sampling: bool = False,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of Thompson quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.enable_optimistic_sampling = enable_optimistic_sampling

        self.alphas = self._initialize_alphas()
        self.adapters = self._initialize_adapters(adapter)

    def _initialize_alphas(self) -> list[float]:
        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, self.n_quantiles + 1)
        ]
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            alphas.append(1 - (upper - lower))
        return alphas

    def _initialize_adapters(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[List[DtACI]]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return [
                DtACI(alpha=alpha, gamma_values=[0.05, 0.01, 0.1])
                for alpha in self.alphas
            ]
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        if self.adapters:
            for i, (adapter, beta) in enumerate(zip(self.adapters, betas)):
                updated_alpha = adapter.update(beta=beta)
                self.alphas[i] = updated_alpha


class ExpectedImprovementSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI"]] = None,
        current_best_value: float = float("inf"),
        num_ei_samples: int = 20,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.current_best_value = current_best_value
        self.num_ei_samples = num_ei_samples

        self.alphas = self._initialize_alphas()
        self.adapters = self._initialize_adapters(adapter)

    def update_best_value(self, value: float):
        """Update the current best value found in optimization."""
        self.current_best_value = min(self.current_best_value, value)

    def _initialize_alphas(self) -> list[float]:
        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, self.n_quantiles + 1)
        ]
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            alphas.append(1 - (upper - lower))
        return alphas

    def _initialize_adapters(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[List[DtACI]]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return [
                DtACI(alpha=alpha, gamma_values=[0.05, 0.01, 0.1])
                for alpha in self.alphas
            ]
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        if self.adapters:
            for i, (adapter, beta) in enumerate(zip(self.adapters, betas)):
                updated_alpha = adapter.update(beta=beta)
                self.alphas[i] = updated_alpha


class InformationGainSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI"]] = None,
        n_paths: int = 100,
        n_X_candidates: int = 10,
        n_y_candidates_per_x: int = 3,
        sampling_strategy: str = "uniform",
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.n_paths = n_paths
        self.n_X_candidates = n_X_candidates
        self.n_y_candidates_per_x = n_y_candidates_per_x
        self.sampling_strategy = sampling_strategy

        self.alphas = self._initialize_alphas()
        self.adapters = self._initialize_adapters(adapter)

    def _initialize_alphas(self) -> list[float]:
        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, self.n_quantiles + 1)
        ]
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            alphas.append(1 - (upper - lower))
        return alphas

    def _initialize_adapters(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[List[DtACI]]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return [
                DtACI(alpha=alpha, gamma_values=[0.05, 0.01, 0.1])
                for alpha in self.alphas
            ]
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        if self.adapters:
            for i, (adapter, beta) in enumerate(zip(self.adapters, betas)):
                updated_alpha = adapter.update(beta=beta)
                self.alphas[i] = updated_alpha


class MaxValueEntropySearchSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI"]] = None,
        n_min_samples: int = 100,  # Number of samples to estimate minimum value distribution
        n_y_samples: int = 20,  # Number of y samples to evaluate per candidate point
        alpha: float = 0.1,  # Parameter for entropy estimation
        sampling_strategy: str = "uniform",  # Strategy for selecting initial candidate points if needed
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.n_min_samples = n_min_samples
        self.n_y_samples = n_y_samples
        self.alpha = alpha
        self.sampling_strategy = sampling_strategy

        self.alphas = self._initialize_alphas()
        self.adapters = self._initialize_adapters(adapter)

    def _initialize_alphas(self) -> list[float]:
        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, self.n_quantiles + 1)
        ]
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            alphas.append(1 - (upper - lower))
        return alphas

    def _initialize_adapters(
        self, adapter: Optional[Literal["DtACI"]] = None
    ) -> Optional[List[DtACI]]:
        if adapter is None:
            return None
        elif adapter == "DtACI":
            return [
                DtACI(alpha=alpha, gamma_values=[0.05, 0.01, 0.1])
                for alpha in self.alphas
            ]
        else:
            raise ValueError("adapter must be None or 'DtACI'")

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def update_interval_width(self, betas: List[float]):
        if self.adapters:
            for i, (adapter, beta) in enumerate(zip(self.adapters, betas)):
                updated_alpha = adapter.update(beta=beta)
                self.alphas[i] = updated_alpha
