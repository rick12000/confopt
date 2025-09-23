"""Conformal acquisition functions for Bayesian optimization.

This module implements acquisition functions that combine conformal prediction with
Bayesian optimization strategies. It provides uncertainty-aware point selection
for hyperparameter optimization through two main approaches: locally weighted
conformal prediction and quantile-based conformal prediction.

The module bridges conformal prediction estimators with acquisition strategies,
enabling adaptive optimization that adjusts exploration based on prediction
uncertainty and coverage feedback. All acquisition functions provide finite-sample
coverage guarantees while optimizing for different exploration-exploitation trade-offs.

Key Components:
    - BaseConformalSearcher: Abstract interface for conformal acquisition functions
    - QuantileConformalSearcher: Quantile-based conformal acquisition

Integration Context:
    Serves as the primary interface between the conformal prediction framework
    and optimization algorithms, supporting various acquisition strategies while
    maintaining theoretical coverage guarantees throughout the optimization process.
"""

import logging
from typing import Optional, Union, Literal, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


from confopt.selection.conformalization import (
    QuantileConformalEstimator,
)
from confopt.selection.sampling.bound_samplers import (
    LowerBoundSampler,
    PessimisticLowerBoundSampler,
)
from confopt.selection.sampling.thompson_samplers import ThompsonSampler
from confopt.selection.sampling.expected_improvement_samplers import (
    ExpectedImprovementSampler,
)

from confopt.selection.estimation import initialize_estimator
from confopt.selection.estimator_configuration import (
    QUANTILE_TO_POINT_ESTIMATOR_MAPPING,
)

logger = logging.getLogger(__name__)

DEFAULT_IG_SAMPLER_RANDOM_STATE = 1234


class BaseConformalSearcher(ABC):
    """Abstract base class for conformal prediction-based acquisition functions.

    Defines the common interface for acquisition functions that combine conformal
    prediction with various sampling strategies for Bayesian optimization. Provides
    unified handling of different acquisition strategies while maintaining coverage
    guarantees through conformal prediction.

    The class implements a strategy pattern where different samplers define the
    acquisition behavior, while the searcher manages the conformal prediction
    component and adaptive alpha updating based on coverage feedback.

    Args:
        sampler: Acquisition strategy implementation that defines point selection
            behavior. Must implement the appropriate calculation methods for the
            chosen acquisition function.

    Attributes:
        sampler: The acquisition strategy instance.
        conformal_estimator: Fitted conformal prediction estimator (set by subclasses).
        X_train: Current training features, updated through optimization process.
        y_train: Current training targets, updated through optimization process.
        X_val: Validation features for conformal calibration.
        y_val: Validation targets for conformal calibration.
        last_beta: Most recent coverage feedback for single-alpha samplers.
        predictions_per_interval: Cached interval predictions from last predict() call.
        point_estimator: Fitted point estimator for optimistic Thompson sampling.

    Design Pattern:
        Implements Template Method pattern with strategy injection, where the
        acquisition strategy is delegated to the sampler while coverage tracking
        and adaptive behavior are handled by the base searcher framework.
    """

    def __init__(
        self,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
        ],
    ):
        self.sampler = sampler
        self.conformal_estimator: Optional[QuantileConformalEstimator] = None
        self.X_train = None
        self.y_train = None
        self.last_beta = None
        self.predictions_per_interval = None

    def predict(self, X: np.array):
        """Generate acquisition function values for candidate points.

        Routes prediction requests to the appropriate sampler-specific method
        based on the configured acquisition strategy. Handles the interface
        between the generic acquisition API and strategy-specific implementations.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Acquisition function values, shape (n_candidates,). Higher values
            indicate more promising candidates for evaluation.

        Raises:
            ValueError: If sampler type is not supported or conformal estimator
                is not fitted.

        Implementation Notes:
            Caches interval predictions in self.predictions_per_interval for
            potential reuse by update() method. The specific acquisition behavior
            depends on the sampler strategy:
            - LowerBoundSampler: Upper confidence bound with exploration decay
            - ThompsonSampler: Posterior sampling with optional optimistic bias
            - PessimisticLowerBoundSampler: Conservative lower bound selection
            - ExpectedImprovementSampler: Expected improvement over current best


        """
        if isinstance(self.sampler, LowerBoundSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)
        elif isinstance(self.sampler, ExpectedImprovementSampler):
            return self._predict_with_expected_improvement(X)

        else:
            raise ValueError(f"Unsupported sampler type: {type(self.sampler)}")

    @abstractmethod
    def _predict_with_ucb(self, X: np.array):
        """Generate upper confidence bound acquisition values.

        Subclasses must implement UCB acquisition using their
        specific conformal prediction approach.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            UCB acquisition values, shape (n_candidates,).
        """

    @abstractmethod
    def _predict_with_thompson(self, X: np.array):
        """Generate Thompson sampling acquisition values.

        Subclasses must implement Thompson sampling using their
        specific conformal prediction approach.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Thompson sampling acquisition values, shape (n_candidates,).
        """

    @abstractmethod
    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        """Generate pessimistic lower bound acquisition values.

        Subclasses must implement pessimistic lower bound acquisition
        using their specific conformal prediction approach.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Lower bound acquisition values, shape (n_candidates,).
        """

    @abstractmethod
    def _predict_with_expected_improvement(self, X: np.array):
        """Generate expected improvement acquisition values.

        Subclasses must implement expected improvement acquisition
        using their specific conformal prediction approach.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Expected improvement acquisition values, shape (n_candidates,).
        """

    @abstractmethod
    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        """Calculate coverage feedback (beta values) for adaptive alpha updating.

        Subclasses must implement beta calculation using their
        specific conformal prediction approach.

        Args:
            X: Configuration where observation was made, shape (n_features,).
            y_true: Observed performance value at the configuration.

        Returns:
            List of beta values, one per alpha level, representing coverage feedback.
        """

    def get_interval(self, X: np.array) -> Tuple[float, float]:
        """Get prediction interval bounds for a given configuration.

        Returns the lower and upper bounds of the prediction interval for
        interval-based samplers. This method is specifically designed for
        samplers that provide single coverage levels.

        Args:
            X: Input configuration, shape (n_features,).

        Returns:
            Tuple of (lower_bound, upper_bound) for the prediction interval.

        Raises:
            ValueError: If conformal estimator is not fitted or if sampler type
                does not support interval retrieval.

        Coverage Information:
            Only works for LowerBoundSampler and PessimisticLowerBoundSampler as
            these samplers use single intervals. Multi-alpha samplers require
            more complex interval handling through the adaptive alpha mechanism.
        """
        if isinstance(self.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)):
            if self.conformal_estimator is None:
                raise ValueError(
                    "Conformal estimator not initialized. Call fit() before getting interval."
                )

            predictions_per_interval = self.conformal_estimator.predict_intervals(
                X.reshape(1, -1)
            )

            # Grab first predictions per interval object, since these samplers have only one alpha/interval
            # Then grab first index of upper and lower bound, since we're predicting for only one X configuration
            interval = predictions_per_interval[0]
            lower_bound = interval.lower_bounds[0]
            upper_bound = interval.upper_bounds[0]

            return lower_bound, upper_bound

        else:
            raise ValueError(
                "Interval retrieval only supported for LowerBoundSampler and PessimisticLowerBoundSampler"
            )

    def update(self, X: np.array, y_true: float) -> None:
        """Update searcher state with new observation and adapt coverage levels.

        Incorporates new data point into the optimization process and updates
        adaptive components based on observed coverage performance. Handles
        sampler-specific updates and alpha adaptation for coverage control.

        Args:
            X: Newly evaluated configuration, shape (n_features,).
            y_true: Observed performance for the configuration.

        Adaptive Mechanisms:
            - ExpectedImprovementSampler: Updates best observed value
            - LowerBoundSampler: Updates exploration schedule and beta decay
            - Adaptive samplers: Updates interval widths based on coverage feedback
            - Conformal estimator: Updates alpha levels if adaptation is enabled

        Coverage Adaptation Process:
            1. Calculate coverage feedback (betas) for the new observation
            2. Update sampler interval widths based on coverage performance
            3. Propagate updated alphas to conformal estimator
            4. Maintain coverage targets through adaptive alpha adjustment

        Implementation Notes:
            The update process varies by sampler type:
            - Single-alpha samplers receive scalar beta values
            - Multi-alpha samplers receive beta vectors for each coverage level
            - Information-gain samplers may cache additional state for efficiency
        """
        if isinstance(self.sampler, ExpectedImprovementSampler):
            self.sampler.update_best_value(y_true)
        if isinstance(self.sampler, LowerBoundSampler):
            self.sampler.update_exploration_step()
        if self.conformal_estimator.fold_scores_per_alpha is not None:
            uses_adaptation = (
                hasattr(self.sampler, "adapter") and self.sampler.adapter is not None
            ) or (
                hasattr(self.sampler, "adapters") and self.sampler.adapters is not None
            )
            if uses_adaptation:
                betas = self._calculate_betas(X, y_true)
                if isinstance(
                    self.sampler,
                    (
                        ThompsonSampler,
                        ExpectedImprovementSampler,
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


QuantileEstimatorArchitecture = Literal[
    "qgbm", "qgp", "qrf", "qknn", "ql", "qleaf", "qens5"
]


class QuantileConformalSearcher(BaseConformalSearcher):
    """Conformal acquisition function using quantile-based prediction intervals.

    Implements acquisition functions based on quantile conformal prediction,
    directly estimating prediction quantiles and applying conformal adjustments
    when sufficient calibration data is available. Provides flexible acquisition
    strategies while maintaining coverage guarantees.

    This approach is particularly effective when the objective function exhibits
    asymmetric uncertainty or when specific quantile behaviors are of interest.
    Automatically switches between conformalized and non-conformalized modes
    based on data availability.

    Args:
        quantile_estimator_architecture: Architecture identifier for the quantile
            estimator. Must be registered in ESTIMATOR_REGISTRY and support
            simultaneous multi-quantile estimation.
        sampler: Acquisition strategy that defines point selection behavior.
        n_pre_conformal_trials: Minimum total samples required for conformal mode.
            Below this threshold, uses direct quantile predictions.

    Attributes:
        quantile_estimator_architecture: Quantile estimator configuration.
        n_pre_conformal_trials: Threshold for conformal vs non-conformal mode.
        conformal_estimator: Fitted QuantileConformalEstimator instance.
        point_estimator: Optional point estimator for optimistic Thompson sampling.


    Mathematical Foundation:
        Uses quantile conformal prediction where intervals have the form:

        Conformalized: [q̂_{α/2}(x) - C_α, q̂_{1-α/2}(x) + C_α]
        Non-conformalized: [q̂_{α/2}(x), q̂_{1-α/2}(x)]

        Where:
        - q̂_τ(x): τ-quantile estimate at location x
        - C_α: Conformal adjustment based on nonconformity scores
        - Mode selection based on n_pre_conformal_trials threshold

    Adaptive Behavior:
        Supports sampler-specific adaptation mechanisms including upper quantile
        capping for conservative samplers and point estimator integration for
        optimistic Thompson sampling when enabled.
    """

    def __init__(
        self,
        quantile_estimator_architecture: QuantileEstimatorArchitecture,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
        ],
        n_pre_conformal_trials: int = 32,
        n_calibration_folds: int = 3,
        calibration_split_strategy: Literal[
            "cv", "train_test_split", "adaptive"
        ] = "adaptive",
    ):
        super().__init__(sampler)
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.n_calibration_folds = n_calibration_folds
        self.calibration_split_strategy = calibration_split_strategy

        self.scaler = StandardScaler()
        self.conformal_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=self.quantile_estimator_architecture,
            alphas=self.sampler.fetch_alphas(),
            n_pre_conformal_trials=self.n_pre_conformal_trials,
            n_calibration_folds=self.n_calibration_folds,
            calibration_split_strategy=self.calibration_split_strategy,
        )

    def fit(
        self,
        X: np.array,
        y: np.array,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ):
        """Fit the quantile conformal estimator for acquisition.

        Trains the quantile estimator and sets up conformal calibration,
        with automatic mode selection based on data availability. Handles
        sampler-specific configurations and point estimator setup for
        optimistic Thompson sampling and median estimation for bound samplers.

        Args:
            X: Input features for estimator fitting, shape (n_samples, n_features).
            y: Target values for estimator fitting, shape (n_samples,).
            tuning_iterations: Number of hyperparameter tuning iterations (0 disables tuning).
            random_state: Random seed for reproducible results.

        Implementation Process:
            1. Store data for potential use by acquisition strategies
            2. Configure sampler-specific quantile estimation and point estimators
            3. Set default random state for Information Gain Sampler if not provided
            4. Fit QuantileConformalEstimator with internal data splitting
            5. Store estimator performance metrics for quality assessment

        Sampler-Specific Setup:
            - Bound samplers: Median (0.5 quantile) estimator for UCB point estimates
            - Optimistic Thompson: Additional point estimator training
            - Information-based: Full quantile range support
        """
        # Store data for potential use by samplers (though splitting is now internal)
        self.X_train = X  # For backwards compatibility
        self.y_train = y
        random_state = random_state

        # Create median/mean estimator for bound samplers (UCB point estimates) and Optimistic Thompson sampling
        if isinstance(
            self.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)
        ) or (
            isinstance(self.sampler, ThompsonSampler)
            and (
                hasattr(self.sampler, "enable_optimistic_sampling")
                and self.sampler.enable_optimistic_sampling
            )
        ):
            # Fit scaler on training data and transform X for point estimator training
            X_normalized = self.scaler.fit_transform(X)

            if (
                self.quantile_estimator_architecture
                in QUANTILE_TO_POINT_ESTIMATOR_MAPPING
            ):
                point_estimator_architecture = QUANTILE_TO_POINT_ESTIMATOR_MAPPING[
                    self.quantile_estimator_architecture
                ]
                self.point_estimator = initialize_estimator(
                    estimator_architecture=point_estimator_architecture,
                    random_state=random_state,
                )
                self.point_estimator.fit(X=X_normalized, y=y)
            # TODO: Temporary fallback to median as point estimator for architectures that
            # don't yet have a point counterpart in the code:
            else:
                self.point_estimator = initialize_estimator(
                    estimator_architecture=self.quantile_estimator_architecture,
                    random_state=random_state,
                )
                self.point_estimator.fit(
                    X=X_normalized,
                    y=y,
                    quantiles=[0.5],  # Only estimate the median
                )

                # NOTE: Scrappy wrapper to align predict calls between quantile and point estimators
                # TODO: Remove in future
                class PointWrapper:
                    def __init__(self, estimator: QuantileConformalEstimator):
                        self.estimator = estimator

                    def predict(self, X):
                        return self.estimator.predict(X)[:, 0]

                self.point_estimator = PointWrapper(self.point_estimator)

        self.conformal_estimator.fit(
            X=X,
            y=y,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )

    def _predict_with_pessimistic_lower_bound(self, X: np.array):
        """Generate pessimistic lower bound acquisition values.

        Returns the lower bounds of quantile-based prediction intervals,
        implementing conservative exploration using direct quantile predictions
        or conformally adjusted intervals depending on data availability.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Lower bounds of prediction intervals, shape (n_candidates,).

        Quantile-Based Strategy:
            Uses estimated quantiles directly for conservative point selection,
            with automatic conformal adjustment when sufficient calibration
            data is available.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.predictions_per_interval[0].lower_bounds

    def _predict_with_ucb(self, X: np.array):
        """Generate upper confidence bound acquisition values.

        Implements UCB acquisition using quantile-based intervals with
        median estimator predictions as point estimates and symmetric variance assumption.
        Adapts automatically to conformalized or non-conformalized mode.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            UCB acquisition values, shape (n_candidates,).

        Mathematical Formulation:
            UCB(x) = point_estimate(x) - β × (interval_width(x) / 2)
            Where point_estimate comes from dedicated point estimator and
            interval bounds come from quantile estimation with symmetric variance assumption.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        interval = self.predictions_per_interval[0]

        # Use dedicated point estimator for point estimates (index 0 since we only fit quantile 0.5)
        X_normalized = self.scaler.transform(X)
        point_estimates = self.point_estimator.predict(X_normalized)

        # Use half the interval width for symmetric variance assumption
        half_width = np.abs(interval.upper_bounds - interval.lower_bounds) / 2
        return self.sampler.calculate_ucb_predictions(
            point_estimates=point_estimates,
            half_width=half_width,
        )

    def _predict_with_thompson(self, X: np.array):
        """Generate Thompson sampling acquisition values.

        Implements Thompson sampling using quantile-based prediction intervals,
        with optional point estimator integration for optimistic bias.
        Automatically adapts to available conformal calibration.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Thompson sampling acquisition values, shape (n_candidates,).

        Sampling Strategy:
            Draws random samples from quantile-based intervals, with optional
            optimistic constraints from separately fitted point estimator
            when enable_optimistic_sampling is True.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_predictions = None
        if self.sampler.enable_optimistic_sampling:
            X_normalized = self.scaler.transform(X)
            point_predictions = self.point_estimator.predict(X_normalized)
        return self.sampler.calculate_thompson_predictions(
            predictions_per_interval=self.predictions_per_interval,
            point_predictions=point_predictions,
        )

    def _predict_with_expected_improvement(self, X: np.array):
        """Generate expected improvement acquisition values.

        Calculates expected improvement using quantile-based prediction
        intervals, automatically accounting for conformalized or
        non-conformalized interval construction.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Expected improvement acquisition values, shape (n_candidates,).

        Quantile-Based EI:
            Integrates improvement probabilities over quantile-estimated
            intervals, naturally handling asymmetric uncertainty patterns
            in the objective function.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_expected_improvement(
            predictions_per_interval=self.predictions_per_interval
        )

    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        """Calculate coverage feedback (beta values) for adaptive alpha updating.

        Computes alpha-specific coverage feedback using quantile-based
        nonconformity scores. Provides separate beta values for each
        alpha level to enable granular coverage control.

        Args:
            X: Configuration where observation was made, shape (n_features,).
            y_true: Observed performance value at the configuration.

        Returns:
            List of beta values, one per alpha level, representing coverage feedback.

        Quantile-Based Beta Calculation:
            For each alpha level, computes nonconformity as the maximum
            deviation from the corresponding quantile interval, then
            calculates the proportion of calibration scores at or below
            this nonconformity for adaptive alpha adjustment.
        """
        return self.conformal_estimator.calculate_betas(X, y_true)
