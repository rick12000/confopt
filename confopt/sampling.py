from typing import Optional, List, Literal, Union
import numpy as np
from confopt.adaptation import ACI, DtACI
from confopt.data_classes import QuantileInterval


class PessimisticLowerBoundSampler:
    def __init__(
        self,
        interval_width: float = 0.8,
        adapter_framework: Optional[Literal["ACI", "DtACI"]] = None,
    ):
        self.interval_width = interval_width

        self.alpha = 1 - interval_width
        self.adapter = self._initialize_adapter(adapter_framework)
        self.quantiles = self._calculate_quantiles()

    def _initialize_adapter(
        self, framework: Optional[Literal["ACI", "DtACI"]] = None
    ) -> Optional[Union[ACI, DtACI]]:
        if framework == "ACI":
            adapter = ACI(alpha=self.alpha)
        elif framework == "DtACI":
            adapter = DtACI(alpha=self.alpha)
            self.expert_alphas = adapter.alpha_t_values
        elif framework is None:
            adapter = None
        else:
            raise ValueError(f"Unknown adapter framework: {framework}")
        return adapter

    def fetch_alpha(self) -> float:
        return self.alpha

    def fetch_expert_alphas(self) -> List[float]:
        if hasattr(self, "expert_alphas"):
            return self.expert_alphas
        return [self.alpha]

    def _calculate_quantiles(self) -> QuantileInterval:
        return QuantileInterval(
            lower_quantile=self.alpha / 2, upper_quantile=1 - (self.alpha / 2)
        )

    def fetch_quantile_interval(self) -> QuantileInterval:
        return self.quantiles

    def update_interval_width(self, breaches: list[int]) -> None:
        if isinstance(self.adapter, ACI):
            if len(breaches) != 1:
                raise ValueError("ACI adapter requires a single breach indicator.")
            self.alpha = self.adapter.update(breach_indicator=breaches[0])
        elif isinstance(self.adapter, DtACI):
            self.alpha = self.adapter.update(breach_indicators=breaches)
        self.quantiles = self._calculate_quantiles()


class LowerBoundSampler(PessimisticLowerBoundSampler):
    def __init__(
        self,
        beta_decay: Literal[
            "inverse_square_root_decay", "logarithmic_decay"
        ] = "logarithmic_decay",
        c: float = 1,
        interval_width: float = 0.8,
        adapter_framework: Optional[Literal["ACI", "DtACI"]] = None,
        upper_quantile_cap: Optional[float] = None,
    ):
        self.beta_decay = beta_decay
        self.c = c
        self.t = 1
        self.beta = 1
        self.upper_quantile_cap = upper_quantile_cap

        # Call at this position, there are initialization methods
        # in the base class:
        super().__init__(interval_width, adapter_framework)

    def _calculate_quantiles(self) -> QuantileInterval:
        if self.upper_quantile_cap:
            interval = QuantileInterval(
                lower_quantile=self.alpha / 2, upper_quantile=self.upper_quantile_cap
            )
        else:
            interval = QuantileInterval(
                lower_quantile=self.alpha / 2, upper_quantile=1 - (self.alpha / 2)
            )
        return interval

    def update_exploration_step(self):
        self.t += 1
        if self.beta_decay == "inverse_square_root_decay":
            self.beta = np.sqrt(self.c / self.t)
        elif self.beta_decay == "logarithmic_decay":
            self.beta = np.sqrt((self.c * np.log(self.t)) / self.t)


class ThompsonSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter_framework: Optional[Literal["ACI"]] = None,
        enable_optimistic_sampling: bool = False,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of Thompson quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.enable_optimistic_sampling = enable_optimistic_sampling

        self.quantiles, self.alphas = self._initialize_quantiles_and_alphas()
        self.adapters = self._initialize_adapters(adapter_framework)

    def _initialize_quantiles_and_alphas(
        self,
    ) -> tuple[list[QuantileInterval], list[float]]:
        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, self.n_quantiles + 1)
        ]
        quantiles = []
        alphas = []
        half_length = len(starting_quantiles) // 2

        for i in range(half_length):
            lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
            quantiles.append(
                QuantileInterval(lower_quantile=lower, upper_quantile=upper)
            )
            alphas.append(1 - (upper - lower))
        return quantiles, alphas

    def _initialize_adapters(
        self, framework: Optional[Literal["ACI"]] = None
    ) -> Optional[List[Union[ACI]]]:
        if framework == "ACI":
            adapter_class = ACI
            adapters = [adapter_class(alpha=alpha) for alpha in self.alphas]
        elif framework is None:
            adapters = None
        else:
            raise ValueError(f"Unknown adapter framework: {framework}")

        return adapters

    def fetch_alphas(self) -> List[float]:
        return self.alphas

    def fetch_intervals(self) -> List[QuantileInterval]:
        return self.quantiles

    def update_interval_width(self, breaches: List[int]):
        for i, (adapter, breach) in enumerate(zip(self.adapters, breaches)):
            updated_alpha = adapter.update(breach_indicator=breach)
            self.alphas[i] = updated_alpha
            self.quantiles[i] = QuantileInterval(
                lower_quantile=updated_alpha / 2, upper_quantile=1 - (updated_alpha / 2)
            )
