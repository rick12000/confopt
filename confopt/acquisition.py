import logging
from typing import Optional, List, Union, Literal
from pydantic import BaseModel

import numpy as np
from sklearn.metrics import mean_pinball_loss
from confopt.tracking import RuntimeTracker
from confopt.adaptation import ACI, DtACI
from confopt.conformalization import LocallyWeightedConformalEstimator
from confopt.estimation import (
    initialize_point_estimator,
    initialize_quantile_estimator,
    tune,
    SEARCH_MODEL_DEFAULT_CONFIGURATIONS,
)

logger = logging.getLogger(__name__)


class QuantileInterval(BaseModel):
    lower_quantile: float
    upper_quantile: float


class UCBSampler:
    def __init__(
        self,
        beta_decay: str = "logarithmic_decay",
        beta: float = 1,
        c: float = 1,
        interval_width: float = 0.2,
        adapter_framework: Optional[str] = None,
    ):
        self.beta_decay = beta_decay
        self.beta = beta
        self.c = c
        self.interval_width = interval_width
        self.alpha = 1 - interval_width
        self.t = 1

        # Initialize adapter if specified
        self.adapter = self._initialize_adapter(adapter_framework)

        self.quantiles = self._calculate_quantiles()

    def _initialize_adapter(self, framework: Optional[str]):
        if framework == "ACI":
            adapter = ACI(alpha=self.alpha)
        elif framework == "DtACI":
            adapter = DtACI(alpha=self.alpha)
            self.expert_alphas = adapter.alpha_t_values
        else:
            adapter = None
        return adapter

    def _calculate_quantiles(self) -> QuantileInterval:
        return QuantileInterval(
            lower_quantile=self.alpha / 2, upper_quantile=1 - (self.alpha / 2)
        )

    def fetch_alpha(self) -> float:
        return self.alpha

    def fetch_expert_alphas(self) -> List[float]:
        return self.expert_alphas

    def fetch_interval(self) -> QuantileInterval:
        return self.quantiles

    def update_exploration_step(self):
        if self.beta_decay == "logarithmic_decay":
            self.beta = self.c * np.log(self.t) / self.t
        elif self.beta_decay == "logarithmic_growth":
            self.beta = 2 * np.log(self.t + 1)
        self.t += 1

    def update_interval_width(self, breaches: list[int]):
        if isinstance(self.adapter, ACI):
            if len(breaches) != 1:
                raise ValueError("ACI adapter requires a single breach indicator.")
            self.alpha = self.adapter.update(breach_indicator=breaches[0])
            self.quantiles = self._calculate_quantiles()
        elif isinstance(self.adapter, DtACI):
            self.alpha = self.adapter.update(breach_indicators=breaches)
            self.quantiles = self._calculate_quantiles()


class ThompsonSampler:
    def __init__(
        self,
        n_quantiles: int = 4,
        adapter_framework: Optional[str] = None,
        enable_optimistic_sampling: bool = False,
    ):
        if n_quantiles % 2 != 0:
            raise ValueError("Number of Thompson quantiles must be even.")

        self.n_quantiles = n_quantiles
        self.enable_optimistic_sampling = enable_optimistic_sampling

        starting_quantiles = [
            round(i / (self.n_quantiles + 1), 2) for i in range(1, n_quantiles + 1)
        ]
        self.quantiles, self.alphas = self._initialize_quantiles_and_alphas(
            starting_quantiles
        )
        self.adapters = self._initialize_adapters(adapter_framework)

    def _initialize_quantiles_and_alphas(self, starting_quantiles: List[float]):
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

    def _initialize_adapters(self, framework: Optional[str]):
        if not framework:
            return []

        adapter_class = ACI if framework == "ACI" else None
        if not adapter_class:
            raise ValueError(f"Unknown adapter framework: {framework}")

        return [adapter_class(alpha=alpha) for alpha in self.alphas]

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


class LocallyWeightedConformalSearcher:
    """
    Locally weighted conformal regression with sampling.

    Uses a locally weighted conformal estimator and applies sampling strategies
    to form point and variability predictions for y.
    """

    def __init__(
        self,
        point_estimator_architecture: str,
        variance_estimator_architecture: str,
        sampler: Union[UCBSampler, ThompsonSampler],
    ):
        self.conformal_estimator = LocallyWeightedConformalEstimator(
            point_estimator_architecture=point_estimator_architecture,
            variance_estimator_architecture=variance_estimator_architecture,
        )
        self.sampler = sampler
        self.training_time = None
        self.predictions_per_interval = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit the conformal estimator.
        """
        self.conformal_estimator.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )
        self.training_time = self.conformal_estimator.training_time
        self.primary_estimator_error = self.conformal_estimator.primary_estimator_error

    def predict(self, X: np.array):
        """
        Predict using the conformal estimator and apply the sampler.
        """
        if isinstance(self.sampler, UCBSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)

    def _predict_with_ucb(self, X: np.array):
        """
        Predict using UCB sampling strategy.
        """
        if isinstance(self.sampler.adapter, DtACI):
            self.predictions_per_interval = []
            for alpha in self.sampler.fetch_expert_alphas():
                lower_bound, upper_bound = self.conformal_estimator.predict_interval(
                    X=X, alpha=alpha, beta=self.sampler.beta
                )
                self.predictions_per_interval.append(
                    np.hstack([lower_bound, upper_bound])
                )
                # Use the current best alpha as the bound
                if self.sampler.fetch_alpha() == alpha:
                    result_lower_bound = lower_bound
        else:
            alpha = self.sampler.fetch_alpha()
            lower_bound, upper_bound = self.conformal_estimator.predict_interval(
                X=X, alpha=alpha, beta=self.sampler.beta
            )
            self.predictions_per_interval = [np.hstack([lower_bound, upper_bound])]
            result_lower_bound = lower_bound

        self.sampler.update_exploration_step()
        return result_lower_bound

    def _predict_with_thompson(self, y_pred: np.array, var_pred: np.array):
        self.predictions_per_interval = []
        for alpha in self.sampler.fetch_alphas():
            score_quantile = np.quantile(self.nonconformity_scores, 1 - alpha)
            scaled_score = score_quantile * var_pred
            self.predictions_per_interval.append(
                np.hstack([y_pred - scaled_score, y_pred + scaled_score])
            )

        predictions_per_quantile = np.hstack(self.predictions_per_interval)
        lower_bound = []
        for i in range(predictions_per_quantile.shape[0]):
            # Use numpy's choice for reproducibility
            ts_idx = np.random.choice(range(self.sampler.n_quantiles))
            if self.sampler.enable_optimistic_sampling:
                lower_bound.append(
                    min(predictions_per_quantile[i, ts_idx], y_pred[i, 0])
                )
            else:
                lower_bound.append(predictions_per_quantile[i, ts_idx])
        lower_bound = np.array(lower_bound)

        return lower_bound

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        breaches = []
        for predictions in self.predictions_per_interval:
            sampled_predictions = predictions[sampled_idx, :]
            lower_quantile, upper_quantile = (
                sampled_predictions[0],
                sampled_predictions[1],
            )
            if lower_quantile <= sampled_performance <= upper_quantile:
                breach = 0
            else:
                breach = 1
            breaches.append(breach)
        self.sampler.update_interval_width(breaches=breaches)


class SingleFitQuantileConformalSearcher:
    def __init__(
        self,
        quantile_estimator_architecture: Literal["qknn", "qrf"],
        sampler: Union[UCBSampler, ThompsonSampler],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.training_time = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit quantile estimator with option to tune.

        Quantile estimators are fitted based on a specified confidence
        level and return two quantile estimates for the symmetrical
        lower and upper bounds around that level.

        Parameters
        ----------
        X_train :
            Explanatory variables used to train the quantile estimator.
        y_train :
            Target variable used to train the quantile estimator.
        X_val :
            Explanatory variables used to calibrate conformal intervals.
        y_val :
            Target variable used to calibrate conformal intervals.
        confidence_level :
            Confidence level determining quantiles to be predicted
            by the quantile estimator. Quantiles are obtained symmetrically
            around the confidence level (eg. 0.5 confidence level would
            result in a quantile estimator for the 25th and 75th percentiles
            of the target variable).
        tuning_iterations :
            Number of tuning searches to perform (eg. 5 means
            the model will randomly select 5 hyperparameter
            configurations for the quantile estimator to evaluate).
            To skip tuning during fitting, set this to 0.
        random_state :
            Random generation seed.

        Returns
        -------
        estimator :
            Fitted estimator object.
        """
        training_time_tracker = RuntimeTracker()
        training_time_tracker.pause_runtime()
        if isinstance(self.sampler, UCBSampler):
            quantile_intervals = [self.sampler.fetch_interval()]
        elif isinstance(self.sampler, ThompsonSampler):
            quantile_intervals = self.sampler.fetch_intervals()
            if self.sampler.enable_optimistic_sampling:
                pass

        training_time_tracker.resume_runtime()
        if tuning_iterations > 1 and len(X_train) > 10:
            flattened_quantiles = []
            for interval in quantile_intervals:
                flattened_quantiles.append(interval.lower_quantile)
                flattened_quantiles.append(interval.upper_quantile)
            initialization_params = tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                quantiles=flattened_quantiles,
                random_state=random_state,
            )
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                self.quantile_estimator_architecture
            ].copy()

        # TODO HERE
        self.quantile_estimator = initialize_point_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )

        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            self.quantile_estimator.fit(X_train, y_train)

            if isinstance(self.sampler, UCBSampler):
                self.nonconformity_scores_per_interval = []
                for interval in quantile_intervals:
                    val_prediction = self.quantile_estimator.predict(
                        X=X_val,
                        quantiles=[interval.lower_quantile, interval.upper_quantile],
                    )
                    lower_conformal_deviations = list(val_prediction[:, 0] - y_val)
                    upper_conformal_deviations = list(y_val - val_prediction[:, 1])
                    nonconformity_scores = []
                    for lower_deviation, upper_deviation in zip(
                        lower_conformal_deviations, upper_conformal_deviations
                    ):
                        nonconformity_scores.append(
                            max(lower_deviation, upper_deviation)
                        )
                    self.nonconformity_scores_per_interval.append(
                        np.array(nonconformity_scores)
                    )

            elif isinstance(self.sampler, ThompsonSampler):
                self.nonconformity_scores_per_interval = []
                for interval in quantile_intervals:
                    val_prediction = self.quantile_estimator.predict(
                        X=X_val,
                        quantiles=[interval.lower_quantile, interval.upper_quantile],
                    )
                    lower_conformal_deviations = list(val_prediction[:, 0] - y_val)
                    upper_conformal_deviations = list(y_val - val_prediction[:, 1])
                    nonconformity_scores = []
                    for lower_deviation, upper_deviation in zip(
                        lower_conformal_deviations, upper_conformal_deviations
                    ):
                        nonconformity_scores.append(
                            max(lower_deviation, upper_deviation)
                        )
                    self.nonconformity_scores_per_interval.append(
                        np.array(nonconformity_scores)
                    )

            self.conformalize_predictions = True

        else:
            self.quantile_estimator.fit(
                X=np.vstack((X_train, X_val)), y=np.concatenate((y_train, y_val))
            )
            self.conformalize_predictions = False

        self.training_time = training_time_tracker.return_runtime()

        # TODO: TEMP
        scores = []
        for quantile_interval in quantile_intervals:
            predictions = self.quantile_estimator.predict(
                X=X_val,
                quantiles=[
                    quantile_interval.lower_quantile,
                    quantile_interval.upper_quantile,
                ],
            )
            lo_y_pred = predictions[:, 0]
            hi_y_pred = predictions[:, 1]
            lo_score = mean_pinball_loss(
                y_val, lo_y_pred, alpha=quantile_interval.lower_quantile
            )
            hi_score = mean_pinball_loss(
                y_val, hi_y_pred, alpha=quantile_interval.upper_quantile
            )
            score = (lo_score + hi_score) / 2
            scores.append(score)
        self.primary_estimator_error = sum(scores) / len(scores)
        # TODO: END OF TEMP

    def predict(self, X: np.array):
        if isinstance(self.sampler, UCBSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)

    def _predict_with_ucb(self, X: np.array):
        if self.conformalize_predictions:
            interval = self.sampler.fetch_interval()
            score = np.quantile(
                self.nonconformity_scores_per_interval[0],
                interval.upper_quantile - interval.lower_quantile,
            )
        else:
            score = 0
        interval = self.sampler.fetch_interval()
        prediction = self.quantile_estimator.predict(
            X=X, quantiles=[interval.lower_quantile, interval.upper_quantile]
        )
        lower_interval_bound = np.array(prediction[:, 0]) - score
        upper_interval_bound = np.array(prediction[:, 1]) + score

        self.predictions_per_interval = [prediction]

        lower_bound = lower_interval_bound + self.sampler.beta * (
            upper_interval_bound - lower_interval_bound
        )

        self.sampler.update_exploration_step()

        return lower_bound

    def _predict_with_thompson(self, X):
        self.predictions_per_interval = []
        if self.conformalize_predictions:
            for nonconformity_scores, interval in zip(
                self.nonconformity_scores_per_interval, self.sampler.fetch_intervals()
            ):
                score = np.quantile(
                    nonconformity_scores,
                    interval.upper_quantile - interval.lower_quantile,
                )
                scores = [-score, score]
                predictions = self.quantile_estimator.predict(
                    X=X, quantiles=[interval.lower_quantile, interval.upper_quantile]
                )
                adjusted_predictions = predictions + np.array(scores).reshape(-1, 1).T
                self.predictions_per_interval.append(adjusted_predictions)
        else:
            for interval in self.sampler.fetch_intervals():
                predictions = self.quantile_estimator.predict(
                    X=X, quantiles=[interval.lower_quantile, interval.upper_quantile]
                )
                self.predictions_per_interval.append(predictions)

        if self.sampler.enable_optimistic_sampling:
            median_predictions = np.array(
                self.quantile_estimator.predict(X=X, quantiles=[0.5])[:, 0]
            ).reshape(-1, 1)

        predictions_per_quantile = np.hstack(self.predictions_per_interval)
        lower_bound = []
        for i in range(predictions_per_quantile.shape[0]):
            # Use numpy's random choice instead of random.choice
            ts_idx = np.random.choice(range(self.sampler.n_quantiles))
            if self.sampler.enable_optimistic_sampling:
                lower_bound.append(
                    min(
                        predictions_per_quantile[i, ts_idx],
                        median_predictions[i, 0],
                    )
                )
            else:
                lower_bound.append(predictions_per_quantile[i, ts_idx])
        lower_bound = np.array(lower_bound)

        return lower_bound

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        breaches = []
        for predictions in self.predictions_per_interval:
            sampled_predictions = predictions[sampled_idx, :]
            lower_quantile, upper_quantile = (
                sampled_predictions[0],
                sampled_predictions[1],
            )
            if lower_quantile <= sampled_performance <= upper_quantile:
                breach = 0
            else:
                breach = 1
            breaches.append(breach)
        self.sampler.update_interval_width(breaches=breaches)


# TODO


class MultiFitQuantileConformalSearcher:
    """
    Quantile conformal regression.

    Fits quantile estimators on X and y data and applies non-conformity
    adjustments to validate quantile estimates.

    The class contains tuning, fitting and prediction methods.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        sampler: Union[UCBSampler, ThompsonSampler],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.sampler = sampler
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.training_time = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """
        Fit quantile estimator with option to tune.

        Quantile estimators are fitted based on a specified confidence
        level and return two quantile estimates for the symmetrical
        lower and upper bounds around that level.

        Parameters
        ----------
        X_train :
            Explanatory variables used to train the quantile estimator.
        y_train :
            Target variable used to train the quantile estimator.
        X_val :
            Explanatory variables used to calibrate conformal intervals.
        y_val :
            Target variable used to calibrate conformal intervals.
        confidence_level :
            Confidence level determining quantiles to be predicted
            by the quantile estimator. Quantiles are obtained symmetrically
            around the confidence level (eg. 0.5 confidence level would
            result in a quantile estimator for the 25th and 75th percentiles
            of the target variable).
        tuning_iterations :
            Number of tuning searches to perform (eg. 5 means
            the model will randomly select 5 hyperparameter
            configurations for the quantile estimator to evaluate).
            To skip tuning during fitting, set this to 0.
        random_state :
            Random generation seed.

        Returns
        -------
        estimator :
            Fitted estimator object.
        """
        training_time_tracker = RuntimeTracker()
        training_time_tracker.pause_runtime()
        if isinstance(self.sampler, UCBSampler):
            quantile_intervals = [self.sampler.fetch_interval()]
        elif isinstance(self.sampler, ThompsonSampler):
            quantile_intervals = self.sampler.fetch_intervals()
            if self.sampler.enable_optimistic_sampling:
                training_time_tracker.resume_runtime()
                median_estimator_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                    self.quantile_estimator_architecture
                ].copy()
                self.median_estimator = initialize_quantile_estimator(
                    estimator_architecture=self.quantile_estimator_architecture,
                    initialization_params=median_estimator_params,
                    pinball_loss_alpha=[0.5],
                    random_state=random_state,
                )
                self.median_estimator.fit(
                    np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))
                )
                training_time_tracker.pause_runtime()

        training_time_tracker.resume_runtime()
        if tuning_iterations > 1 and len(X_train) > 10:
            params_per_interval = []
            for interval in quantile_intervals:
                initialization_params = tune(
                    X=X_train,
                    y=y_train,
                    estimator_architecture=self.quantile_estimator_architecture,
                    n_searches=tuning_iterations,
                    quantiles=[interval.lower_quantile, interval.upper_quantile],
                    random_state=random_state,
                )
                params_per_interval.append(initialization_params)
        else:
            initialization_params = SEARCH_MODEL_DEFAULT_CONFIGURATIONS[
                self.quantile_estimator_architecture
            ].copy()
            params_per_interval = [initialization_params] * len(quantile_intervals)

        self.estimators_per_interval = []
        for interval in quantile_intervals:
            quantile_estimator = initialize_quantile_estimator(
                estimator_architecture=self.quantile_estimator_architecture,
                initialization_params=initialization_params,
                pinball_loss_alpha=[interval.lower_quantile, interval.upper_quantile],
                random_state=random_state,
            )
            self.estimators_per_interval.append(quantile_estimator)

        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            for estimator in self.estimators_per_interval:
                estimator.fit(X_train, y_train)

            if isinstance(self.sampler, UCBSampler):
                self.nonconformity_scores_per_interval = []
                val_prediction = self.estimators_per_interval[0].predict(X_val)
                lower_conformal_deviations = list(val_prediction[:, 0] - y_val)
                upper_conformal_deviations = list(y_val - val_prediction[:, -1])
                nonconformity_scores = []
                for lower_deviation, upper_deviation in zip(
                    lower_conformal_deviations, upper_conformal_deviations
                ):
                    nonconformity_scores.append(max(lower_deviation, upper_deviation))
                self.nonconformity_scores_per_interval.append(
                    np.array(nonconformity_scores)
                )

            elif isinstance(self.sampler, ThompsonSampler):
                self.nonconformity_scores_per_interval = []
                for estimator in self.estimators_per_interval:
                    val_prediction = estimator.predict(X_val)
                    lower_conformal_deviations = list(val_prediction[:, 0] - y_val)
                    upper_conformal_deviations = list(y_val - val_prediction[:, 1])
                    nonconformity_scores = []
                    for lower_deviation, upper_deviation in zip(
                        lower_conformal_deviations, upper_conformal_deviations
                    ):
                        nonconformity_scores.append(
                            max(lower_deviation, upper_deviation)
                        )
                    self.nonconformity_scores_per_interval.append(
                        np.array(nonconformity_scores)
                    )

            self.conformalize_predictions = True

        else:
            for estimator in self.estimators_per_interval:
                estimator.fit(
                    np.vstack((X_train, X_val)), np.concatenate((y_train, y_val))
                )

            self.conformalize_predictions = False

        self.training_time = training_time_tracker.return_runtime()

        # TODO: TEMP
        scores = []
        for quantile_interval, estimator in zip(
            quantile_intervals, self.estimators_per_interval
        ):
            predictions = estimator.predict(X_val)
            lo_y_pred = predictions[:, 0]
            hi_y_pred = predictions[:, 1]
            lo_score = mean_pinball_loss(
                y_val, lo_y_pred, alpha=quantile_interval.lower_quantile
            )
            hi_score = mean_pinball_loss(
                y_val, hi_y_pred, alpha=quantile_interval.upper_quantile
            )
            score = (lo_score + hi_score) / 2
            scores.append(score)
        self.primary_estimator_error = sum(scores) / len(scores)
        # TODO: END OF TEMP

    def predict(self, X: np.array):
        if isinstance(self.sampler, UCBSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)

    def _predict_with_ucb(self, X: np.array):
        if self.conformalize_predictions:
            interval = self.sampler.fetch_interval()
            score = np.quantile(
                self.nonconformity_scores_per_interval[0],
                interval.upper_quantile - interval.lower_quantile,
            )
        else:
            score = 0
        prediction = self.estimators_per_interval[0].predict(X)
        lower_interval_bound = np.array(prediction[:, 0]) - score
        upper_interval_bound = np.array(prediction[:, 1]) + score

        self.predictions_per_interval = [prediction]

        lower_bound = lower_interval_bound + self.sampler.beta * (
            upper_interval_bound - lower_interval_bound
        )

        self.sampler.update_exploration_step()

        return lower_bound

    def _predict_with_thompson(self, X):
        self.predictions_per_interval = []
        if self.conformalize_predictions:
            for nonconformity_scores, estimator in zip(
                self.nonconformity_scores_per_interval, self.estimators_per_interval
            ):
                score = np.quantile(
                    nonconformity_scores,
                    estimator.quantiles[1] - estimator.quantiles[0],
                )
                scores = [-score, score]
                predictions = estimator.predict(X)
                adjusted_predictions = predictions + np.array(scores).reshape(-1, 1).T
                self.predictions_per_interval.append(adjusted_predictions)
        else:
            for estimator in self.estimators_per_interval:
                predictions = estimator.predict(X)
                self.predictions_per_interval.append(predictions)

        if self.sampler.enable_optimistic_sampling:
            median_predictions = np.array(
                self.median_estimator.predict(X)[:, 0]
            ).reshape(-1, 1)

        predictions_per_quantile = np.hstack(self.predictions_per_interval)
        lower_bound = []
        for i in range(predictions_per_quantile.shape[0]):
            # Use numpy's choice instead of random.choice
            ts_idx = np.random.choice(range(self.sampler.n_quantiles))
            if self.sampler.enable_optimistic_sampling:
                lower_bound.append(
                    min(
                        predictions_per_quantile[i, ts_idx],
                        median_predictions[i, 0],
                    )
                )
            else:
                lower_bound.append(predictions_per_quantile[i, ts_idx])
        lower_bound = np.array(lower_bound)

        return lower_bound

    def update_interval_width(self, sampled_idx: int, sampled_performance: float):
        breaches = []
        for predictions in self.predictions_per_interval:
            sampled_predictions = predictions[sampled_idx, :]
            lower_quantile, upper_quantile = (
                sampled_predictions[0],
                sampled_predictions[1],
            )
            if lower_quantile <= sampled_performance <= upper_quantile:
                breach = 0
            else:
                breach = 1
            breaches.append(breach)
        self.sampler.update_interval_width(breaches=breaches)
