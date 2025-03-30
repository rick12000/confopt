import logging
import numpy as np
from typing import Optional, Tuple, List
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from confopt.wrapping import ConformalBounds
from confopt.utils.preprocessing import train_val_split
from confopt.selection.estimation import (
    initialize_estimator,
    PointTuner,
    QuantileTuner,
)

logger = logging.getLogger(__name__)


class LocallyWeightedConformalEstimator:
    def __init__(
        self,
        point_estimator_architecture: str,
        variance_estimator_architecture: str,
        alphas: List[float],
    ):
        self.point_estimator_architecture = point_estimator_architecture
        self.variance_estimator_architecture = variance_estimator_architecture
        self.alphas = alphas
        self.pe_estimator = None
        self.ve_estimator = None
        self.nonconformity_scores = None
        self.primary_estimator_error = None

    def _tune_fit_component_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator_architecture: str,
        tuning_iterations: int,
        min_obs_for_tuning: int = 15,
        random_state: Optional[int] = None,
    ):
        if tuning_iterations > 1 and len(X) > min_obs_for_tuning:
            tuner = PointTuner(random_state=random_state)
            initialization_params = tuner.tune(
                X=X,
                y=y,
                estimator_architecture=estimator_architecture,
                n_searches=tuning_iterations,
            )
        else:
            initialization_params = None

        estimator = initialize_estimator(
            estimator_architecture=estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )
        estimator.fit(X, y)

        return estimator

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 15,
        random_state: Optional[int] = None,
    ):
        (X_pe, y_pe, X_ve, y_ve,) = train_val_split(
            X_train,
            y_train,
            train_split=0.75,
            normalize=False,
            random_state=random_state,
        )

        self.pe_estimator = self._tune_fit_component_estimator(
            X=X_pe,
            y=y_pe,
            estimator_architecture=self.point_estimator_architecture,
            tuning_iterations=tuning_iterations,
            min_obs_for_tuning=min_obs_for_tuning,
            random_state=random_state,
        )
        abs_pe_residuals = abs(y_ve - self.pe_estimator.predict(X_ve))

        self.ve_estimator = self._tune_fit_component_estimator(
            X=X_ve,
            y=abs_pe_residuals,
            estimator_architecture=self.variance_estimator_architecture,
            tuning_iterations=tuning_iterations,
            min_obs_for_tuning=min_obs_for_tuning,
            random_state=random_state,
        )
        var_pred = self.ve_estimator.predict(X_val)
        var_pred = np.array([0.001 if x <= 0 else x for x in var_pred])

        self.nonconformity_scores = (
            abs(y_val - self.pe_estimator.predict(X_val)) / var_pred
        )

        # TODO: Temporary, for paper calculations:
        self.primary_estimator_error = mean_squared_error(
            self.pe_estimator.predict(X=X_val), y_val
        )

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        if self.pe_estimator is None or self.ve_estimator is None:
            raise ValueError("Estimators must be fitted before prediction")

        y_pred = np.array(self.pe_estimator.predict(X)).reshape(-1, 1)
        var_pred = self.ve_estimator.predict(X)
        var_pred = np.array([max(x, 0) for x in var_pred]).reshape(-1, 1)

        intervals = []
        for alpha in self.alphas:
            non_conformity_score_quantile = np.quantile(
                self.nonconformity_scores, 1 - alpha
            )
            scaled_score = non_conformity_score_quantile * var_pred

            lower_bounds = y_pred - scaled_score
            upper_bounds = y_pred + scaled_score
            intervals.append(
                ConformalBounds(lower_bounds=lower_bounds, upper_bounds=upper_bounds)
            )

        return intervals

    def calculate_betas(self, X: np.array, y_true: float) -> float:
        if self.pe_estimator is None or self.ve_estimator is None:
            raise ValueError("Estimators must be fitted before calculating beta")

        X = X.reshape(1, -1)
        y_pred = self.pe_estimator.predict(X)[0]
        var_pred = max(0.001, self.ve_estimator.predict(X)[0])

        nonconformity = abs(y_true - y_pred) / var_pred

        beta = np.mean(self.nonconformity_scores >= nonconformity)
        betas = [beta] * len(self.alphas)

        return betas


def alpha_to_quantiles(
    alpha: float, upper_quantile_cap: Optional[float] = None
) -> Tuple[float, float]:
    lower_quantile = alpha / 2
    upper_quantile = (
        upper_quantile_cap if upper_quantile_cap is not None else 1 - lower_quantile
    )
    return lower_quantile, upper_quantile


class QuantileConformalEstimator:
    def __init__(
        self,
        quantile_estimator_architecture: str,
        alphas: List[float],
        n_pre_conformal_trials: int = 20,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.alphas = alphas
        self.n_pre_conformal_trials = n_pre_conformal_trials

        self.quantile_estimator = None
        self.nonconformity_scores = None
        self.all_quantiles = None
        self.conformalize_predictions = False
        self.primary_estimator_error = None

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 15,
        upper_quantile_cap: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.upper_quantile_cap = upper_quantile_cap

        all_quantiles = []
        for alpha in self.alphas:
            lower_quantile, upper_quantile = alpha_to_quantiles(
                alpha, upper_quantile_cap
            )
            all_quantiles.append(lower_quantile)
            all_quantiles.append(upper_quantile)
        all_quantiles = sorted(all_quantiles)

        self.quantile_indices = {q: i for i, q in enumerate(all_quantiles)}

        if tuning_iterations > 1 and len(X_train) > min_obs_for_tuning:
            tuner = QuantileTuner(random_state=random_state, quantiles=all_quantiles)
            initialization_params = tuner.tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
            )
        else:
            initialization_params = None

        self.quantile_estimator = initialize_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )

        if len(X_train) + len(X_val) > self.n_pre_conformal_trials:
            self.nonconformity_scores = [np.array([]) for _ in self.alphas]
            self.quantile_estimator.fit(X_train, y_train, quantiles=all_quantiles)

            for i, alpha in enumerate(self.alphas):
                lower_quantile, upper_quantile = alpha_to_quantiles(
                    alpha, upper_quantile_cap
                )

                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                val_prediction = self.quantile_estimator.predict(X_val)

                lower_conformal_deviations = val_prediction[:, lower_idx] - y_val
                upper_conformal_deviations = y_val - val_prediction[:, upper_idx]

                self.nonconformity_scores[i] = np.maximum(
                    lower_conformal_deviations, upper_conformal_deviations
                )
            self.conformalize_predictions = True
        else:
            self.quantile_estimator.fit(
                X=np.vstack((X_train, X_val)),
                y=np.concatenate((y_train, y_val)),
                quantiles=all_quantiles,
            )
            self.conformalize_predictions = False

        # TODO: Temporary, for paper calculations:
        scores = []
        for alpha in self.alphas:
            lower_quantile, upper_quantile = alpha_to_quantiles(
                alpha, upper_quantile_cap
            )
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            predictions = self.quantile_estimator.predict(X_val)

            lo_y_pred = predictions[:, lower_idx]
            hi_y_pred = predictions[:, upper_idx]

            lo_score = mean_pinball_loss(y_val, lo_y_pred, alpha=lower_quantile)
            hi_score = mean_pinball_loss(y_val, hi_y_pred, alpha=upper_quantile)
            scores.extend([lo_score, hi_score])

        self.primary_estimator_error = np.mean(scores)

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before prediction")

        intervals = []
        prediction = self.quantile_estimator.predict(X)

        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(
                alpha, self.upper_quantile_cap
            )

            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            if self.conformalize_predictions:
                score = np.quantile(
                    self.nonconformity_scores[i],
                    1 - alpha,
                )
                lower_interval_bound = np.array(prediction[:, lower_idx]) - score
                upper_interval_bound = np.array(prediction[:, upper_idx]) + score
            else:
                lower_interval_bound = np.array(prediction[:, lower_idx])
                upper_interval_bound = np.array(prediction[:, upper_idx])

            intervals.append(
                ConformalBounds(
                    lower_bounds=lower_interval_bound, upper_bounds=upper_interval_bound
                )
            )

        return intervals

    def calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        if self.quantile_estimator is None:
            raise ValueError("Estimator must be fitted before calculating beta")

        X = X.reshape(1, -1)

        betas = []
        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(
                alpha, self.upper_quantile_cap
            )
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            prediction = self.quantile_estimator.predict(X)
            lower_bound = prediction[0, lower_idx]
            upper_bound = prediction[0, upper_idx]

            lower_deviation = lower_bound - y_true
            upper_deviation = y_true - upper_bound
            nonconformity = max(lower_deviation, upper_deviation)

            beta = np.mean(self.nonconformity_scores[i] >= nonconformity)

            betas.append(beta)

        return betas
