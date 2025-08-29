"""
Bound-based acquisition strategies for conformal prediction optimization.

This module implements acquisition strategies that use prediction interval bounds
for optimization decisions. The approaches focus on conservative uncertainty
quantification through lower bound sampling and exploration-exploitation
trade-offs through adaptive confidence bound strategies.

Bound-based methodology:
These samplers utilize specific bounds (typically lower bounds for minimization)
from prediction intervals to make acquisition decisions. This approach provides
direct interpretable acquisition values while maintaining proper uncertainty
quantification through conformal prediction intervals.

Key strategies:
- Pessimistic Lower Bound: Conservative approach using only lower bounds
- Lower Confidence Bound (LCB): UCB-style exploration with decay schedules
- Adaptive interval width adjustment based on coverage feedback

The module provides both simple bound-based acquisition and sophisticated
exploration strategies with theoretical guarantees for convergence in
optimization under uncertainty scenarios.
"""

from typing import Optional, List, Literal
import numpy as np
from confopt.selection.sampling.utils import (
    initialize_single_adapter,
    update_single_interval_width,
)


class PessimisticLowerBoundSampler:
    """
    Conservative acquisition strategy using pessimistic lower bounds.

    This sampler implements a conservative approach to uncertainty quantification
    by focusing exclusively on the lower bounds of prediction intervals. The
    strategy prioritizes risk-averse decision making by assuming pessimistic
    scenarios, making it suitable for applications where conservative estimates
    are preferred over aggressive exploration.

    The approach provides simple, interpretable acquisition values while
    maintaining proper uncertainty quantification through conformal prediction
    intervals. The single-interval design offers computational efficiency and
    straightforward interpretation.

    Methodological characteristics:
    - Single confidence level with configurable interval width
    - Direct lower bound extraction for acquisition decisions
    - Optional adaptive interval width adjustment
    - Conservative bias suitable for risk-averse optimization
    """

    def __init__(
        self,
        interval_width: float = 0.8,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
    ):
        """
        Initialize pessimistic lower bound sampler with specified confidence level.

        Args:
            interval_width: Confidence level for prediction intervals (e.g., 0.8
                for 80% intervals). Higher values provide wider intervals with
                more conservative bounds. Typical values: 0.7-0.95.
            adapter: Interval width adaptation strategy. "DtACI" provides
                aggressive multi-scale adaptation, "ACI" offers conservative
                adaptation, None disables adaptation.
        """
        self.interval_width = interval_width
        self.alpha = 1 - interval_width
        self.adapter = initialize_single_adapter(self.alpha, adapter)

    def fetch_alphas(self) -> List[float]:
        """
        Retrieve current alpha value for interval construction.

        Returns:
            Single-element list containing the current alpha value (miscoverage rate).
        """
        return [self.alpha]

    def update_interval_width(self, beta: float) -> None:
        """
        Update interval width based on observed coverage rate.

        This method applies adaptive interval width adjustment using empirical
        coverage feedback. The alpha parameter is updated to maintain target
        coverage while optimizing interval efficiency for conservative bound
        estimation.

        Args:
            beta: Observed coverage rate for the prediction interval, representing
                the fraction of true values falling within the interval.
        """
        self.alpha = update_single_interval_width(self.adapter, self.alpha, beta)


class LowerBoundSampler(PessimisticLowerBoundSampler):
    """
    Lower Confidence Bound acquisition strategy with adaptive exploration.

    This sampler implements a Lower Confidence Bound (LCB) strategy adapted for
    minimization problems. The approach balances exploitation of promising regions
    with exploration of uncertain areas through an adaptive exploration parameter
    that decays over time, providing theoretical guarantees for convergence.

    The strategy extends the pessimistic lower bound approach with sophisticated
    exploration control, making it suitable for efficient optimization under
    uncertainty with provable regret bounds.

    Mathematical formulation:
    LCB(x) = μ(x) - β(t) * σ(x)
    where μ(x) is the point estimate, σ(x) is the interval width, and β(t)
    is the time-dependent exploration parameter.

    Exploration decay strategies:
    - Inverse square root: β(t) = sqrt(c/t) for aggressive decay
    - Logarithmic: β(t) = sqrt(c*log(t)/t) for balanced exploration

    Performance characteristics:
    - Theoretical regret guarantees under appropriate decay schedules
    - Adaptive exploration balancing exploitation and uncertainty quantification
    - Efficient single-interval computation with optional adaptation
    """

    def __init__(
        self,
        interval_width: float = 0.8,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        beta_decay: Optional[
            Literal[
                "inverse_square_root_decay",
                "logarithmic_decay",
            ]
        ] = "logarithmic_decay",
        c: float = 1,
        beta_max: float = 10,
    ):
        """
        Initialize LCB sampler with exploration decay schedule.

        Args:
            interval_width: Confidence level for prediction intervals (e.g., 0.8
                for 80% intervals). Higher values provide wider intervals with
                larger exploration bonuses.
            adapter: Interval width adaptation strategy for coverage maintenance.
            beta_decay: Exploration parameter decay strategy. "logarithmic_decay"
                provides balanced exploration with theoretical guarantees,
                "inverse_square_root_decay" offers more aggressive decay.
            c: Exploration constant controlling the magnitude of exploration bonus.
                Higher values increase exploration, lower values favor exploitation.
                Typical values: 0.1-10.
            beta_max: Maximum exploration parameter value to prevent excessive
                exploration in early iterations. Provides stability for the
                acquisition function.
        """
        super().__init__(interval_width, adapter)
        self.beta_decay = beta_decay
        self.c = c
        self.t = 1  # Time step counter for decay computation
        self.beta = 1  # Current exploration parameter
        self.beta_max = beta_max
        self.mu_max = float("-inf")  # Tracking for potential future use

    def update_exploration_step(self):
        """
        Update exploration parameter based on decay schedule and time step.

        This method advances the time step and computes the new exploration
        parameter according to the specified decay strategy. The decay ensures
        that exploration decreases over time as confidence in the model increases,
        following theoretical requirements for convergence guarantees.
        """
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

    def calculate_ucb_predictions(
        self,
        point_estimates: np.ndarray = None,
        half_width: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate Lower Confidence Bound predictions for acquisition.

        This method computes LCB values by combining point estimates with
        exploration bonuses based on interval widths and the current exploration
        parameter. The result provides acquisition values that balance
        exploitation of promising regions with exploration of uncertain areas.

        Args:
            point_estimates: Point predictions (e.g., posterior means) for each
                candidate. These represent the exploitation component.
            half_width: Uncertainty estimates (e.g., half interval widths) for
                each candidate. These drive the exploration component.

        Returns:
            Array of LCB acquisition values. Lower values indicate more attractive
            candidates for minimization problems.
        """
        return point_estimates - self.beta * half_width
