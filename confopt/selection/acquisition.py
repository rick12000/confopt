import logging
from typing import Optional, Union, Literal
import numpy as np
from abc import ABC, abstractmethod


from confopt.selection.conformalization import (
    LocallyWeightedConformalEstimator,
    QuantileConformalEstimator,
)
from confopt.selection.sampling import (
    LowerBoundSampler,
    ThompsonSampler,
    PessimisticLowerBoundSampler,
    ExpectedImprovementSampler,
    InformationGainSampler,
    MaxValueEntropySearchSampler,
)
from confopt.selection.estimation import initialize_estimator

logger = logging.getLogger(__name__)

DEFAULT_IG_SAMPLER_RANDOM_STATE = 1234

# Point estimator architecture literals for LocallyWeightedConformalSearcher
PointEstimatorArchitecture = Literal["gbm", "lgbm", "rf", "knn", "kr", "pens"]

# Quantile estimator architecture literals for QuantileConformalSearcher
QuantileEstimatorArchitecture = Literal[
    "qrf",
    "qgbm",
    "qlgbm",
    "qknn",
    "ql",
    "qgp",
    "qens1",
    "qens2",
    "qens3",
    "qens4",
    "qens5",
]


class BaseConformalSearcher(ABC):
    def __init__(
        self,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
            InformationGainSampler,
            MaxValueEntropySearchSampler,
        ],
    ):
        self.sampler = sampler
        self.conformal_estimator = None
        self.X_train = None
        self.y_train = None
        self.last_beta = None

    def predict(self, X: np.array):
        if isinstance(self.sampler, LowerBoundSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)
        elif isinstance(self.sampler, ExpectedImprovementSampler):
            return self._predict_with_expected_improvement(X)
        elif isinstance(self.sampler, InformationGainSampler):
            return self._predict_with_information_gain(X)
        elif isinstance(self.sampler, MaxValueEntropySearchSampler):
            return self._predict_with_max_value_entropy_search(X)
        else:
            raise ValueError(f"Unsupported sampler type: {type(self.sampler)}")

    @abstractmethod
    def _predict_with_ucb(self, X: np.array):
        pass

    @abstractmethod
    def _predict_with_thompson(self, X: np.array):
        pass

    @abstractmethod
    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        pass

    @abstractmethod
    def _predict_with_expected_improvement(self, X: np.array):
        pass

    @abstractmethod
    def _predict_with_information_gain(self, X: np.array):
        pass

    @abstractmethod
    def _predict_with_max_value_entropy_search(self, X: np.array):
        pass

    @abstractmethod
    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        pass

    def calculate_breach(self, X: np.array, y_true: float) -> int:
        """
        Calculate whether y_true breaches the predicted interval.
        Only works for LowerBoundSampler and PessimisticLowerBoundSampler.

        Args:
            X: Input configuration (1D array)
            y_true: True performance value

        Returns:
            int: 1 if y_true is outside the interval (breach), 0 if inside (no breach)
        """
        if isinstance(self.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)):

            predictions_per_interval = self.conformal_estimator.predict_intervals(
                X.reshape(1, -1)
            )

            # Grab first predictions per interval object, since these samplers have only one alpha/interval
            # Then grab first index of upper and lower bound, since we're predicting for only one X configuration
            interval = predictions_per_interval[0]
            lower_bound = interval.lower_bounds[0]
            upper_bound = interval.upper_bounds[0]

            breach_status = int(y_true < lower_bound or y_true > upper_bound)

        else:
            raise ValueError(
                "Breach calculation only supported for LowerBoundSampler and PessimisticLowerBoundSampler"
            )

        return breach_status

    def update(self, X: np.array, y_true: float) -> None:
        if self.X_train is not None:
            self.X_train = np.vstack([self.X_train, X])
            self.y_train = np.append(self.y_train, y_true)
        else:
            self.X_train = X.reshape(1, -1)
            self.y_train = np.array([y_true])
        if isinstance(self.sampler, ExpectedImprovementSampler):
            self.sampler.update_best_value(y_true)
        if isinstance(self.sampler, LowerBoundSampler):
            self.sampler.update_exploration_step()
        if self.conformal_estimator.nonconformity_scores is not None:
            uses_adaptation = hasattr(self.sampler, "adapter") or hasattr(
                self.sampler, "adapters"
            )
            if uses_adaptation:
                betas = self._calculate_betas(X, y_true)
                if isinstance(
                    self.sampler,
                    (
                        ThompsonSampler,
                        ExpectedImprovementSampler,
                        InformationGainSampler,
                        MaxValueEntropySearchSampler,
                    ),
                ):
                    self.sampler.update_interval_width(betas=betas)
                elif isinstance(
                    self.sampler, (PessimisticLowerBoundSampler, LowerBoundSampler)
                ):
                    if len(betas) == 1:
                        self.last_beta = betas[0]
                        self.sampler.update_interval_width(beta=betas[0])
                    else:
                        raise ValueError(
                            "Multiple betas returned for single beta sampler."
                        )
                self.conformal_estimator.update_alphas(self.sampler.fetch_alphas())


class LocallyWeightedConformalSearcher(BaseConformalSearcher):
    def __init__(
        self,
        point_estimator_architecture: PointEstimatorArchitecture,
        variance_estimator_architecture: PointEstimatorArchitecture,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
            InformationGainSampler,
            MaxValueEntropySearchSampler,
        ],
    ):
        super().__init__(sampler)
        self.point_estimator_architecture = point_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture
        self.conformal_estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=self.point_estimator_architecture,
            variance_estimator_architecture=self.variance_estimator_architecture,
            alphas=self.sampler.fetch_alphas(),
        )

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        if isinstance(self.sampler, InformationGainSampler) and random_state is None:
            random_state = DEFAULT_IG_SAMPLER_RANDOM_STATE
        self.conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )
        self.primary_estimator_error = self.conformal_estimator.primary_estimator_error

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.predictions_per_interval[0].lower_bounds

    def _predict_with_ucb(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_estimates = np.array(
            self.conformal_estimator.pe_estimator.predict(X)
        ).reshape(-1, 1)
        interval = self.predictions_per_interval[0]
        width = (interval.upper_bounds - interval.lower_bounds).reshape(-1, 1) / 2
        return self.sampler.calculate_ucb_predictions(
            predictions_per_interval=self.predictions_per_interval,
            point_estimates=point_estimates,
            interval_width=width,
        )

    def _predict_with_thompson(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_predictions = None
        if self.sampler.enable_optimistic_sampling:
            point_predictions = self.conformal_estimator.pe_estimator.predict(X)
        return self.sampler.calculate_thompson_predictions(
            predictions_per_interval=self.predictions_per_interval,
            point_predictions=point_predictions,
        )

    def _predict_with_expected_improvement(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_expected_improvement(
            predictions_per_interval=self.predictions_per_interval
        )

    def _predict_with_information_gain(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_information_gain(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_space=X,
            conformal_estimator=self.conformal_estimator,
            predictions_per_interval=self.predictions_per_interval,
            n_jobs=1,
        )

    def _predict_with_max_value_entropy_search(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_max_value_entropy_search(
            predictions_per_interval=self.predictions_per_interval,
            n_jobs=1,
        )

    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        return self.conformal_estimator.calculate_betas(X, y_true)


class QuantileConformalSearcher(BaseConformalSearcher):
    def __init__(
        self,
        quantile_estimator_architecture: QuantileEstimatorArchitecture,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
            InformationGainSampler,
            MaxValueEntropySearchSampler,
        ],
        n_pre_conformal_trials: int = 20,
    ):
        super().__init__(sampler)
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.conformal_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=self.quantile_estimator_architecture,
            alphas=self.sampler.fetch_alphas(),
            n_pre_conformal_trials=self.n_pre_conformal_trials,
        )

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        random_state = random_state
        if isinstance(self.sampler, InformationGainSampler) and random_state is None:
            random_state = DEFAULT_IG_SAMPLER_RANDOM_STATE
        if isinstance(self.sampler, (PessimisticLowerBoundSampler, LowerBoundSampler)):
            upper_quantile_cap = 0.5
        elif isinstance(
            self.sampler,
            (ThompsonSampler, InformationGainSampler, MaxValueEntropySearchSampler),
        ):
            upper_quantile_cap = None
            if (
                hasattr(self.sampler, "enable_optimistic_sampling")
                and self.sampler.enable_optimistic_sampling
            ):
                self.point_estimator = initialize_estimator(
                    estimator_architecture="gbm",
                    random_state=random_state,
                )
                self.point_estimator.fit(
                    X=np.vstack((X_train, X_val)),
                    y=np.concatenate((y_train, y_val)),
                )
        elif isinstance(self.sampler, ExpectedImprovementSampler):
            upper_quantile_cap = None
        else:
            raise ValueError(f"Unsupported sampler type: {type(self.sampler)}")
        self.conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
            upper_quantile_cap=upper_quantile_cap,
        )
        self.primary_estimator_error = self.conformal_estimator.primary_estimator_error

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.predictions_per_interval[0].lower_bounds

    def _predict_with_ucb(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        interval = self.predictions_per_interval[0]
        width = interval.upper_bounds - interval.lower_bounds
        return self.sampler.calculate_ucb_predictions(
            predictions_per_interval=self.predictions_per_interval,
            point_estimates=interval.upper_bounds,
            interval_width=width,
        )

    def _predict_with_thompson(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_predictions = None
        if self.sampler.enable_optimistic_sampling:
            point_predictor = getattr(self, "point_estimator", None)
            if point_predictor:
                point_predictions = point_predictor.predict(X)
        return self.sampler.calculate_thompson_predictions(
            predictions_per_interval=self.predictions_per_interval,
            point_predictions=point_predictions,
        )

    def _predict_with_expected_improvement(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_expected_improvement(
            predictions_per_interval=self.predictions_per_interval
        )

    def _predict_with_information_gain(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_information_gain(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_space=X,
            conformal_estimator=self.conformal_estimator,
            predictions_per_interval=self.predictions_per_interval,
            n_jobs=1,
        )

    def _predict_with_max_value_entropy_search(self, X: np.array):
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_max_value_entropy_search(
            predictions_per_interval=self.predictions_per_interval,
            n_jobs=1,
        )

    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        return self.conformal_estimator.calculate_betas(X, y_true)
