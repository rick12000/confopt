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
    - LocallyWeightedConformalSearcher: Variance-adapted conformal acquisition
    - QuantileConformalSearcher: Quantile-based conformal acquisition

Integration Context:
    Serves as the primary interface between the conformal prediction framework
    and optimization algorithms, supporting various acquisition strategies while
    maintaining theoretical coverage guarantees throughout the optimization process.
"""

import logging
from typing import Optional, Union, Literal
import numpy as np
from abc import ABC, abstractmethod


from confopt.selection.conformalization import (
    LocallyWeightedConformalEstimator,
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
from confopt.selection.sampling.entropy_samplers import (
    EntropySearchSampler,
    MaxValueEntropySearchSampler,
)
from confopt.selection.estimation import initialize_estimator

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
            EntropySearchSampler,
            MaxValueEntropySearchSampler,
        ],
    ):
        self.sampler = sampler
        self.conformal_estimator: Optional[
            Union[LocallyWeightedConformalEstimator, QuantileConformalEstimator]
        ] = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
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
            - InformationGainSampler: Information-theoretic point selection
            - MaxValueEntropySearchSampler: Maximum value entropy search
        """
        if isinstance(self.sampler, LowerBoundSampler):
            return self._predict_with_ucb(X)
        elif isinstance(self.sampler, ThompsonSampler):
            return self._predict_with_thompson(X)
        elif isinstance(self.sampler, PessimisticLowerBoundSampler):
            return self._predict_with_pessimistic_lower_bound(X)
        elif isinstance(self.sampler, ExpectedImprovementSampler):
            return self._predict_with_expected_improvement(X)
        elif isinstance(self.sampler, EntropySearchSampler):
            return self._predict_with_information_gain(X)
        elif isinstance(self.sampler, MaxValueEntropySearchSampler):
            return self._predict_with_max_value_entropy_search(X)
        else:
            raise ValueError(f"Unsupported sampler type: {type(self.sampler)}")

    @abstractmethod
    def _predict_with_ucb(self, X: np.array):
        """Generate upper confidence bound acquisition values.

        Subclasses must implement UCB acquisition strategy using their
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
    def _predict_with_information_gain(self, X: np.array):
        """Generate information gain acquisition values.

        Subclasses must implement information gain acquisition
        using their specific conformal prediction approach.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Information gain acquisition values, shape (n_candidates,).
        """

    @abstractmethod
    def _predict_with_max_value_entropy_search(self, X: np.array):
        """Generate max-value entropy search acquisition values.

        Subclasses must implement max-value entropy search acquisition
        using their specific conformal prediction approach.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            MES acquisition values, shape (n_candidates,).
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

    def calculate_breach(self, X: np.array, y_true: float) -> int:
        """Calculate whether y_true breaches the predicted interval.

        Determines if the observed value falls outside the prediction interval,
        providing feedback for coverage assessment. This method is specifically
        designed for interval-based samplers that provide single coverage levels.

        Args:
            X: Input configuration, shape (n_features,).
            y_true: Observed performance value for the configuration.

        Returns:
            1 if y_true is outside the interval (breach), 0 if inside (coverage).

        Raises:
            ValueError: If conformal estimator is not fitted or if sampler type
                does not support breach calculation.

        Coverage Feedback:
            Only works for LowerBoundSampler and PessimisticLowerBoundSampler as
            these samplers use single intervals. Multi-alpha samplers require
            more complex coverage tracking through the adaptive alpha mechanism.

        Mathematical Definition:
            breach = 1 if y_true < lower_bound OR y_true > upper_bound
            breach = 0 if lower_bound ≤ y_true ≤ upper_bound
        """
        if isinstance(self.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)):
            if self.conformal_estimator is None:
                raise ValueError(
                    "Conformal estimator not initialized. Call fit() before calculating breach."
                )

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
                        EntropySearchSampler,
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


PointEstimatorArchitecture = Literal["gbm", "lgbm", "rf", "knn", "kr", "pens"]


class LocallyWeightedConformalSearcher(BaseConformalSearcher):
    """Conformal acquisition function using locally weighted variance adaptation.

    Implements acquisition functions based on locally weighted conformal prediction,
    where prediction intervals adapt to local variance patterns in the objective
    function. Combines point estimation with variance estimation to create
    heteroscedastic prediction intervals that guide optimization effectively.

    This approach excels when the objective function exhibits varying uncertainty
    across the parameter space, as it can narrow intervals in low-noise regions
    while expanding them in high-uncertainty areas.

    Args:
        point_estimator_architecture: Architecture identifier for the point estimator
            that models the conditional mean. Must be registered in ESTIMATOR_REGISTRY.
        variance_estimator_architecture: Architecture identifier for the variance
            estimator that models prediction uncertainty. Must be registered in
            ESTIMATOR_REGISTRY.
        sampler: Acquisition strategy that defines point selection behavior.

    Attributes:
        point_estimator_architecture: Point estimator configuration.
        variance_estimator_architecture: Variance estimator configuration.
        conformal_estimator: Fitted LocallyWeightedConformalEstimator instance.
        primary_estimator_error: Point estimator validation error for quality assessment.

    Mathematical Foundation:
        Uses locally weighted conformal prediction where intervals have the form:
        [μ̂(x) - q₁₋ₐ(R) × σ̂(x), μ̂(x) + q₁₋ₐ(R) × σ̂(x)]

        Where:
        - μ̂(x): Point estimate at location x
        - σ̂(x): Variance estimate at location x
        - R: Nonconformity scores |yᵢ - μ̂(xᵢ)| / σ̂(xᵢ)
        - q₁₋ₐ(R): (1-α)-quantile of nonconformity scores

    Coverage Adaptation:
        Supports adaptive alpha adjustment through sampler feedback mechanisms,
        allowing dynamic coverage control based on optimization progress and
        coverage performance monitoring.
    """

    def __init__(
        self,
        point_estimator_architecture: PointEstimatorArchitecture,
        variance_estimator_architecture: PointEstimatorArchitecture,
        sampler: Union[
            LowerBoundSampler,
            ThompsonSampler,
            PessimisticLowerBoundSampler,
            ExpectedImprovementSampler,
            EntropySearchSampler,
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
        """Fit the locally weighted conformal estimator for acquisition.

        Trains both point and variance estimators using the provided data,
        following the locally weighted conformal prediction methodology.
        Sets up the acquisition function for subsequent optimization.

        Args:
            X_train: Training features for estimator fitting, shape (n_train, n_features).
            y_train: Training targets for estimator fitting, shape (n_train,).
            X_val: Validation features for conformal calibration, shape (n_val, n_features).
            y_val: Validation targets for conformal calibration, shape (n_val,).
            tuning_iterations: Number of hyperparameter tuning iterations (0 disables tuning).
            random_state: Random seed for reproducible results, required for InformationGainSampler.

        Implementation Process:
            1. Store training and validation data for access by acquisition strategies
            2. Set default random state for Information Gain Sampler if not provided
            3. Fit LocallyWeightedConformalEstimator with data splitting for proper calibration
            4. Store point estimator validation error for performance monitoring

        Data Usage:
            - X_train, y_train: Split internally for point and variance estimation
            - X_val, y_val: Used for conformal calibration and nonconformity score computation
            - Ensures proper separation required for conformal prediction guarantees
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        if isinstance(self.sampler, EntropySearchSampler) and random_state is None:
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
        """Generate pessimistic lower bound acquisition values.

        Returns the lower bounds of prediction intervals as acquisition values,
        implementing a conservative exploration strategy that prioritizes
        configurations with potentially good worst-case performance.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Lower bounds of prediction intervals, shape (n_candidates,).

        Acquisition Strategy:
            Selects points based on interval lower bounds, encouraging exploration
            of regions where even pessimistic estimates suggest good performance.
            Naturally balances exploration and exploitation through interval width.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.predictions_per_interval[0].lower_bounds

    def _predict_with_ucb(self, X: np.array):
        """Generate upper confidence bound acquisition values.

        Implements upper confidence bound (UCB) acquisition using point estimates
        adjusted by exploration terms based on prediction uncertainty. Combines
        locally weighted variance estimates with adaptive exploration control.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            UCB acquisition values, shape (n_candidates,).

        Mathematical Formulation:
            UCB(x) = μ̂(x) - β × σ̂(x)/2
            Where β is the exploration parameter that decays over time.

        Implementation Details:
            Uses point estimator predictions as mean estimates and interval
            half-widths as uncertainty measures. The beta parameter controls
            exploration-exploitation trade-off and adapts over optimization steps.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_estimates = np.array(
            self.conformal_estimator.pe_estimator.predict(X)
        ).reshape(-1, 1)
        interval = self.predictions_per_interval[0]
        width = (interval.upper_bounds - interval.lower_bounds).reshape(-1, 1) / 2
        return self.sampler.calculate_ucb_predictions(
            point_estimates=point_estimates,
            interval_width=width,
        )

    def _predict_with_thompson(self, X: np.array):
        """Generate Thompson sampling acquisition values.

        Implements Thompson sampling by drawing random samples from prediction
        intervals, optionally incorporating point predictions for optimistic bias.
        Provides natural exploration through posterior sampling.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Thompson sampling acquisition values, shape (n_candidates,).

        Sampling Strategy:
            Randomly samples from prediction intervals to represent epistemic
            uncertainty. When optimistic sampling is enabled, samples are
            constrained by point predictions to bias toward exploitation.

        Implementation Details:
            Uses locally weighted intervals for sampling, with optional
            point prediction constraints for optimistic Thompson sampling.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        point_predictions = None
        if self.sampler.enable_optimistic_sampling:
            point_predictions = self.conformal_estimator.pe_estimator.predict(X)
        return self.sampler.calculate_thompson_predictions(
            predictions_per_interval=self.predictions_per_interval,
            point_predictions=point_predictions,
        )

    def _predict_with_expected_improvement(self, X: np.array):
        """Generate expected improvement acquisition values.

        Calculates expected improvement over the current best observed value
        using locally weighted prediction intervals. Balances exploitation
        of promising regions with exploration of uncertain areas.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Expected improvement acquisition values, shape (n_candidates,).

        Acquisition Strategy:
            Computes expected improvement by integrating improvement probabilities
            over locally weighted prediction intervals, naturally accounting for
            heteroscedastic uncertainty in the objective function.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_expected_improvement(
            predictions_per_interval=self.predictions_per_interval
        )

    def _predict_with_information_gain(self, X: np.array):
        """Generate information gain acquisition values.

        Calculates information-theoretic acquisition values that prioritize
        points expected to provide maximal information about the objective
        function. Uses locally weighted prediction intervals for uncertainty
        quantification in information gain calculations.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Information gain acquisition values, shape (n_candidates,).

        Information-Theoretic Approach:
            Selects points that maximize expected reduction in prediction
            uncertainty, using locally adapted intervals to capture
            heteroscedastic uncertainty patterns in information calculations.

        Implementation Notes:
            Requires access to training and validation data for proper
            information gain computation. Uses single-threaded execution
            for consistent results across different environments.
        """
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
        """Generate max-value entropy search acquisition values.

        Implements max-value entropy search (MES) acquisition that focuses
        on reducing uncertainty about the global optimum location. Uses
        locally weighted intervals for uncertainty representation.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            MES acquisition values, shape (n_candidates,).

        Max-Value Strategy:
            Prioritizes points that provide maximal information about the
            location of the global optimum, using locally adaptive uncertainty
            estimates to guide the search toward promising regions.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_information_gain(
            predictions_per_interval=self.predictions_per_interval,
            n_jobs=1,
        )

    def _calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        """Calculate coverage feedback (beta values) for adaptive alpha updating.

        Computes the proportion of calibration points with nonconformity scores
        greater than or equal to the observed nonconformity for the new point.
        Provides coverage feedback for adaptive interval width adjustment.

        Args:
            X: Configuration where observation was made, shape (n_features,).
            y_true: Observed performance value at the configuration.

        Returns:
            List of beta values, one per alpha level, representing coverage feedback.

        Beta Calculation:
            For each alpha level, beta represents the empirical coverage rate
            based on the new observation's nonconformity relative to calibration
            scores. Used for adaptive alpha adjustment in coverage control.
        """
        return self.conformal_estimator.calculate_betas(X, y_true)


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
        primary_estimator_error: Mean pinball loss across quantiles for quality assessment.

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
            EntropySearchSampler,
            MaxValueEntropySearchSampler,
        ],
        n_pre_conformal_trials: int = 32,
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
        """Fit the quantile conformal estimator for acquisition.

        Trains the quantile estimator and sets up conformal calibration,
        with automatic mode selection based on data availability. Handles
        sampler-specific configurations and point estimator setup for
        optimistic Thompson sampling.

        Args:
            X_train: Training features for estimator fitting, shape (n_train, n_features).
            y_train: Training targets for estimator fitting, shape (n_train,).
            X_val: Validation features for conformal calibration, shape (n_val, n_features).
            y_val: Validation targets for conformal calibration, shape (n_val,).
            tuning_iterations: Number of hyperparameter tuning iterations (0 disables tuning).
            random_state: Random seed for reproducible results, required for InformationGainSampler.

        Implementation Process:
            1. Store training and validation data for access by acquisition strategies
            2. Configure sampler-specific quantile estimation (upper caps, point estimators)
            3. Set default random state for Information Gain Sampler if not provided
            4. Fit QuantileConformalEstimator with appropriate quantile configuration
            5. Store estimator performance metrics for quality assessment

        Sampler-Specific Setup:
            - Conservative samplers: Upper quantile capping at 0.5
            - Optimistic Thompson: Additional point estimator training
            - Information-based: Full quantile range support
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        random_state = random_state
        if isinstance(self.sampler, EntropySearchSampler) and random_state is None:
            random_state = DEFAULT_IG_SAMPLER_RANDOM_STATE
        if isinstance(self.sampler, (PessimisticLowerBoundSampler, LowerBoundSampler)):
            upper_quantile_cap = 0.5
        elif isinstance(
            self.sampler,
            (ThompsonSampler),
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
        elif isinstance(
            self.sampler,
            (
                ExpectedImprovementSampler,
                EntropySearchSampler,
                MaxValueEntropySearchSampler,
            ),
        ):
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
        upper bounds as point estimates and interval widths for exploration.
        Adapts automatically to conformalized or non-conformalized mode.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            UCB acquisition values, shape (n_candidates,).

        Mathematical Formulation:
            UCB(x) = upper_bound(x) - β × interval_width(x)
            Where interval bounds come from quantile estimation with
            optional conformal adjustment.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        interval = self.predictions_per_interval[0]
        width = interval.upper_bounds - interval.lower_bounds
        return self.sampler.calculate_ucb_predictions(
            point_estimates=interval.upper_bounds,
            interval_width=width,
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
            point_predictor = getattr(self, "point_estimator", None)
            if point_predictor:
                point_predictions = point_predictor.predict(X)
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

    def _predict_with_information_gain(self, X: np.array):
        """Generate information gain acquisition values.

        Calculates information-theoretic acquisition values using quantile-based
        uncertainty quantification. Leverages full quantile range for
        comprehensive uncertainty characterization in information calculations.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            Information gain acquisition values, shape (n_candidates,).

        Quantile-Based Information:
            Uses quantile estimates to represent prediction uncertainty
            in information gain calculations, providing rich uncertainty
            characterization for information-theoretic point selection.
        """
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
        """Generate max-value entropy search acquisition values.

        Implements max-value entropy search using quantile-based uncertainty
        estimates. Focuses on reducing uncertainty about global optimum
        location using asymmetric quantile-based intervals.

        Args:
            X: Candidate points for evaluation, shape (n_candidates, n_features).

        Returns:
            MES acquisition values, shape (n_candidates,).

        Quantile-Based MES:
            Leverages quantile-based uncertainty representation for
            max-value entropy search, naturally handling skewed or
            asymmetric uncertainty patterns in optimum location inference.
        """
        self.predictions_per_interval = self.conformal_estimator.predict_intervals(X)
        return self.sampler.calculate_information_gain(
            predictions_per_interval=self.predictions_per_interval,
            n_jobs=1,
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
            calculates the proportion of calibration scores exceeding
            this nonconformity for adaptive alpha adjustment.
        """
        return self.conformal_estimator.calculate_betas(X, y_true)
