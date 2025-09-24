import logging
import random
from typing import Optional, Dict, Tuple, get_type_hints, Literal, List
from confopt.wrapping import ParameterRange

import numpy as np
from tqdm import tqdm
from datetime import datetime
import inspect
from confopt.utils.tracking import (
    Trial,
    Study,
    RuntimeTracker,
    StaticConfigurationManager,
    DynamicConfigurationManager,
    ProgressBarManager,
)
from confopt.utils.optimization import FixedSearcherOptimizer, DecayingSearcherOptimizer
from confopt.selection.acquisition import (
    QuantileConformalSearcher,
    BaseConformalSearcher,
)
from confopt.selection.sampling.bound_samplers import (
    LowerBoundSampler,
    PessimisticLowerBoundSampler,
)
from confopt.selection.sampling.thompson_samplers import ThompsonSampler

logger = logging.getLogger(__name__)


def stop_search(
    n_remaining_configurations: int,
    current_iter: int,
    current_runtime: float,
    max_runtime: Optional[float] = None,
    max_searches: Optional[int] = None,
) -> bool:
    """Determine whether to terminate the hyperparameter search process.

    Evaluates multiple stopping criteria to determine if the optimization should halt.
    The function implements a logical OR of termination conditions: exhausted search space,
    runtime budget exceeded, or iteration limit reached.

    Args:
        n_remaining_configurations: Number of configurations still available for evaluation
        current_iter: Current iteration count in the search process
        current_runtime: Elapsed time since search initiation in seconds
        max_runtime: Maximum allowed runtime in seconds, None for no limit
        max_searches: Maximum allowed iterations, None for no limit

    Returns:
        True if any stopping criterion is met, False otherwise
    """
    if n_remaining_configurations == 0:
        return True

    if max_runtime is not None:
        if current_runtime >= max_runtime:
            return True

    if max_searches is not None:
        if current_iter >= max_searches:
            return True

    return False


class ConformalTuner:
    """Conformal prediction-based hyperparameter optimization framework.

    Implements a sophisticated hyperparameter optimization system that combines random search
    initialization with conformal prediction-guided exploration. The tuner uses uncertainty
    quantification to make statistically principled decisions about which configurations
    to evaluate next, providing both efficiency improvements and theoretical guarantees.

    The optimization process follows a two-phase strategy:
    1. Random search phase: Explores the search space randomly to establish baseline performance
    2. Conformal search phase: Uses conformal prediction models to guide configuration selection

    The framework supports adaptive retraining of prediction models, dynamic configuration
    sampling, and multi-armed bandit optimization for automatically tuning searcher parameters.
    Statistical validity is maintained through proper conformal prediction procedures that
    provide distribution-free coverage guarantees.

    Args:
        objective_function: Function to optimize, must accept 'configuration' dict parameter
        search_space: Dictionary mapping parameter names to ParameterRange objects
        minimize: Whether to minimize (True) or maximize (False) the objective function
        n_candidates: Number of candidate configurations to sample from the search space at
            each iteration of conformal search
        warm_starts: Pre-evaluated (configuration, performance) pairs to seed the search
        dynamic_sampling: Whether to dynamically resample configuration candidates at each
            iteration of conformal search
        random_state: Random seed for reproducible results. Default: None.

    Attributes:
        study: Container for storing trial results and optimization history
        config_manager: Handles configuration sampling and tracking
        search_timer: Tracks total optimization runtime
    """

    def __init__(
        self,
        objective_function: callable,
        search_space: Dict[str, ParameterRange],
        minimize: bool = True,
        n_candidates: int = 5000,
        warm_starts: Optional[List[Tuple[Dict, float]]] = None,
        dynamic_sampling: bool = True,
    ) -> None:
        self.objective_function = objective_function
        self.check_objective_function()

        self.search_space = search_space
        self.minimize = minimize
        self.metric_sign = 1 if minimize else -1
        self.warm_starts = warm_starts
        self.n_candidates = n_candidates
        self.dynamic_sampling = dynamic_sampling
        self.config_manager = None

    def check_objective_function(self) -> None:
        """Validate objective function signature and type annotations.

        Ensures the objective function conforms to the required interface:
        single parameter named 'configuration' of type Dict, returning numeric value.
        This validation prevents runtime errors and ensures compatibility with
        the optimization framework.

        Raises:
            ValueError: If function signature doesn't match requirements
            TypeError: If type annotations are incorrect
        """
        signature = inspect.signature(self.objective_function)
        args = list(signature.parameters.values())

        if len(args) != 1:
            raise ValueError("Objective function must take exactly one argument.")

        first_arg = args[0]
        if first_arg.name != "configuration":
            raise ValueError(
                "The objective function must take exactly one argument named 'configuration'."
            )

        type_hints = get_type_hints(self.objective_function)
        if "configuration" in type_hints and type_hints["configuration"] is not Dict:
            raise TypeError(
                "The 'configuration' argument of the objective must be of type Dict."
            )
        if "return" in type_hints and type_hints["return"] not in [
            int,
            float,
            np.number,
        ]:
            raise TypeError(
                "The return type of the objective function must be numeric (int, float, or np.number)."
            )

    def process_warm_starts(self) -> None:
        """Initialize optimization with pre-evaluated configurations.

        Processes warm start configurations by marking them as searched and creating
        corresponding trial records. This allows the optimization to begin with
        prior knowledge, potentially accelerating convergence by skipping known
        poor configurations and leveraging good starting points.

        The warm start configurations are treated as iteration 0 data and assigned
        the 'warm_start' acquisition source for tracking purposes.
        """
        for idx, (config, performance) in enumerate(self.warm_starts):
            self.config_manager.mark_as_searched(config, performance)
            trial = Trial(
                iteration=idx,
                timestamp=datetime.now(),
                configuration=config.copy(),
                tabularized_configuration=self.config_manager.listify_configs([config])[
                    0
                ],
                performance=performance,
                acquisition_source="warm_start",
            )
            self.study.append_trial(trial)

    def initialize_tuning_resources(self) -> None:
        """Initialize core optimization components and data structures.

        Sets up the study container for trial tracking, configuration manager for
        handling search space sampling, and processes any warm start configurations.
        The configuration manager uses the optimized incremental approach for
        maximum performance.
        """
        self.study = Study(
            metric_optimization="minimize" if self.minimize else "maximize"
        )

        # Instantiate appropriate configuration manager based on dynamic_sampling setting
        if self.dynamic_sampling:
            self.config_manager = DynamicConfigurationManager(
                search_space=self.search_space,
                n_candidate_configurations=self.n_candidates,
            )
        else:
            self.config_manager = StaticConfigurationManager(
                search_space=self.search_space,
                n_candidate_configurations=self.n_candidates,
            )

        if self.warm_starts:
            self.process_warm_starts()

    def _evaluate_configuration(self, configuration: Dict) -> Tuple[float, float]:
        """Evaluate a configuration and measure execution time.

        Executes the objective function with the given configuration while tracking
        runtime. This method provides the core evaluation mechanism used throughout
        both random and conformal search phases.

        Args:
            configuration: Parameter configuration dictionary to evaluate

        Returns:
            Tuple of (performance_value, evaluation_runtime)
        """
        runtime_tracker = RuntimeTracker()
        performance = self.objective_function(configuration=configuration)
        runtime = runtime_tracker.return_runtime()
        return performance, runtime

    def random_search(
        self,
        max_random_iter: int,
        max_runtime: Optional[int] = None,
        max_searches: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Execute random search phase to initialize optimization with baseline data.

        Performs uniform random sampling of configurations to establish initial
        performance landscape understanding. This phase is crucial for subsequent
        conformal prediction model training, as it provides the foundational
        dataset for uncertainty quantification.

        Args:
            max_random_iter: Maximum number of random configurations to evaluate
            max_runtime: Optional runtime budget in seconds
            max_searches: Optional total iteration limit
            verbose: Whether to display progress information
        """

        available_configs = self.config_manager.get_searchable_configurations()
        adj_n_searches = min(max_random_iter, len(available_configs))
        if adj_n_searches == 0:
            logger.warning("No configurations available for random search")

        search_idxs = np.random.choice(
            len(available_configs), size=adj_n_searches, replace=False
        )
        sampled_configs = [available_configs[idx] for idx in search_idxs]

        progress_iter = (
            tqdm(sampled_configs, desc="Random search: ")
            if verbose
            else sampled_configs
        )

        for config in progress_iter:
            validation_performance, training_time = self._evaluate_configuration(config)

            if np.isnan(validation_performance):
                logger.debug(
                    "Obtained non-numerical performance, skipping configuration."
                )
                self.config_manager.add_to_banned_configurations(config)
                continue

            self.config_manager.mark_as_searched(config, validation_performance)

            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=config.copy(),
                tabularized_configuration=self.config_manager.listify_configs([config])[
                    0
                ],
                performance=validation_performance,
                acquisition_source="rs",
                target_model_runtime=training_time,
            )
            self.study.append_trial(trial)

            searchable_count = self.config_manager.get_searchable_configurations_count()
            current_runtime = self.search_timer.return_runtime()

            stop = stop_search(
                n_remaining_configurations=searchable_count,
                current_runtime=current_runtime,
                max_runtime=max_runtime,
                current_iter=len(self.study.trials),
                max_searches=max_searches,
            )
            if stop:
                break

    def setup_conformal_search_resources(
        self,
        verbose: bool,
        max_runtime: Optional[int],
        max_searches: Optional[int],
    ) -> Tuple[ProgressBarManager, float]:
        """Initialize progress tracking and iteration limits for conformal search.

        Sets up the progress bar manager for displaying search progress and calculates
        the maximum number of conformal search iterations based on total limits and
        already completed trials from previous phases.

        Args:
            verbose: Whether to display progress information
            max_runtime: Optional maximum runtime in seconds
            max_searches: Optional maximum total iterations

        Returns:
            Tuple of (progress_manager, conformal_max_searches)
        """
        progress_manager = ProgressBarManager(verbose=verbose)
        progress_manager.create_progress_bar(
            max_runtime=max_runtime,
            max_searches=max_searches,
            current_trials=len(self.study.trials),
            description="Conformal search",
        )

        conformal_max_searches = (
            max_searches - len(self.study.trials)
            if max_searches is not None
            else float("inf")
        )

        return progress_manager, conformal_max_searches

    def initialize_searcher_optimizer(
        self,
        optimizer_framework: Optional[str],
    ):
        """Initialize searcher parameter tuner.

        Args:
            optimizer_framework: Tuning strategy ('decaying', 'fixed', None)

        Returns:
            Configured optimizer instance
        """
        if optimizer_framework == "fixed":
            optimizer = FixedSearcherOptimizer(
                n_tuning_episodes=10,
                tuning_interval=20,
            )
        elif optimizer_framework == "decaying":
            optimizer = DecayingSearcherOptimizer(
                n_tuning_episodes=10,
                initial_tuning_interval=10,
                decay_rate=0.1,
                decay_type="linear",
                max_tuning_interval=40,
            )
        elif optimizer_framework is None:
            optimizer = FixedSearcherOptimizer(
                n_tuning_episodes=0,
                tuning_interval=1,
            )
        else:
            raise ValueError(
                "optimizer_framework must be either 'fixed', 'decaying', or None."
            )
        return optimizer

    def retrain_searcher(
        self,
        searcher: BaseConformalSearcher,
        X: np.array,
        y: np.array,
        tuning_count: int,
    ) -> float:
        """Train conformal prediction searcher on accumulated data.

        Fits the conformal prediction model using the provided data,
        tracking training time and model performance for adaptive parameter
        optimization. The tuning_count parameter controls internal hyperparameter
        optimization within the searcher.

        Args:
            searcher: Conformal searcher instance to train
            X: Feature matrix (sign-adjusted)
            y: Target values (sign-adjusted)
            tuning_count: Number of internal tuning iterations

        Returns:
            Training runtime in seconds
        """
        runtime_tracker = RuntimeTracker()
        searcher.fit(
            X=X,
            y=y,
            tuning_iterations=tuning_count,
        )

        training_runtime = runtime_tracker.return_runtime()
        return training_runtime

    def select_next_configuration(
        self,
        searcher: BaseConformalSearcher,
        searchable_configs: List,
        transformed_configs: np.array,
    ) -> Dict:
        """Select the most promising configuration using conformal predictions.

        Uses the conformal searcher to predict lower bounds for all available
        configurations and selects the one with the minimum predicted lower bound.
        This implements a pessimistic acquisition strategy that favors configurations
        with high confidence of good performance.

        Args:
            searcher: Trained conformal searcher for predictions
            searchable_configs: List of available configuration dictionaries
            transformed_configs: Scaled feature matrix for configurations

        Returns:
            Selected configuration dictionary
        """
        bounds = searcher.predict(X=transformed_configs)
        next_idx = np.argmin(bounds)
        next_config = searchable_configs[next_idx]
        return next_config

    def get_interval_if_applicable(
        self,
        searcher: BaseConformalSearcher,
        transformed_config: np.array,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get prediction interval bounds if supported by searcher.

        Returns the lower and upper bounds of the prediction interval for
        configurations using lower bound samplers. This provides the raw
        interval information for storage and analysis.

        Args:
            searcher: Conformal searcher instance
            transformed_config: Scaled configuration features

        Returns:
            Tuple of (lower_bound, upper_bound) if applicable, (None, None) otherwise
        """
        if isinstance(
            searcher.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)
        ):
            lower_bound, upper_bound = searcher.get_interval(X=transformed_config)
            return lower_bound, upper_bound
        else:
            return None, None

    def update_optimizer_parameters(
        self,
        optimizer,
        search_iter: int,
    ) -> Tuple[int, int]:
        """Update multi-armed bandit optimizer and select new parameter values.

        Updates the parameter optimizer with the current search iteration and
        selects new parameter values for subsequent iterations.

        Args:
            optimizer: Multi-armed bandit optimizer instance
            search_iter: Current search iteration number

        Returns:
            Tuple of (new_tuning_count, new_searcher_retuning_frequency)
        """
        optimizer.update(
            search_iter=search_iter,
        )

        new_tuning_count, new_searcher_retuning_frequency = optimizer.select_arm()
        return new_tuning_count, new_searcher_retuning_frequency

    def conformal_search(
        self,
        searcher: BaseConformalSearcher,
        verbose: bool,
        max_searches: Optional[int],
        max_runtime: Optional[int],
        optimizer_framework: Optional[str] = None,
    ) -> None:
        """Execute conformal prediction-guided hyperparameter search.

        Implements the main conformal search loop that iteratively trains conformal
        prediction models, selects promising configurations based on uncertainty
        quantification, and updates the models with new observations.

        Args:
            searcher: Conformal prediction searcher for configuration selection
            verbose: Whether to display search progress
            max_searches: Maximum total iterations including previous phases
            max_runtime: Maximum total runtime budget in seconds
            optimizer_framework: Parameter tuning strategy
        """
        (
            progress_manager,
            conformal_max_searches,
        ) = self.setup_conformal_search_resources(verbose, max_runtime, max_searches)
        optimizer = self.initialize_searcher_optimizer(
            optimizer_framework=optimizer_framework,
        )

        tuning_count = 0
        searcher_retuning_frequency = 1
        training_runtime = 0

        for search_iter in range(conformal_max_searches):
            progress_manager.update_progress(
                current_runtime=(
                    self.search_timer.return_runtime() if max_runtime else None
                ),
                iteration_count=1 if max_searches else 0,
            )

            X = self.config_manager.tabularize_configs(
                self.config_manager.searched_configs
            )
            y = np.array(self.config_manager.searched_performances) * self.metric_sign

            searchable_configs = self.config_manager.get_searchable_configurations()
            X_searchable = self.config_manager.tabularize_configs(searchable_configs)

            if search_iter == 0 or search_iter % 1 == 0:
                training_runtime = self.retrain_searcher(searcher, X, y, tuning_count)

                (
                    tuning_count,
                    searcher_retuning_frequency,
                ) = self.update_optimizer_parameters(
                    optimizer,
                    search_iter,
                )

            # Select next configuration
            next_config = self.select_next_configuration(
                searcher, searchable_configs, X_searchable
            )

            # Evaluate configuration
            performance, _ = self._evaluate_configuration(next_config)
            if np.isnan(performance):
                self.config_manager.add_to_banned_configurations(next_config)
                continue

            # Get interval bounds
            transformed_config = self.config_manager.tabularize_configs([next_config])

            lower_bound, upper_bound = self.get_interval_if_applicable(
                searcher, transformed_config
            )

            # Convert bounds back to original units and handle interval orientation
            if lower_bound is not None and upper_bound is not None:
                converted_lower = lower_bound * self.metric_sign
                converted_upper = upper_bound * self.metric_sign
                # For maximization (metric_sign = -1), swap bounds to maintain proper ordering
                if not self.minimize:
                    signed_lower_bound = converted_upper  # What was upper becomes lower
                    signed_upper_bound = converted_lower  # What was lower becomes upper
                else:
                    signed_lower_bound = converted_lower
                    signed_upper_bound = converted_upper
            else:
                signed_lower_bound = None
                signed_upper_bound = None

            signed_performance = self.metric_sign * performance
            searcher.update(X=transformed_config.flatten(), y_true=signed_performance)

            self.config_manager.mark_as_searched(next_config, performance)
            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=next_config.copy(),
                tabularized_configuration=self.config_manager.listify_configs(
                    [next_config]
                )[0],
                performance=performance,
                acquisition_source=str(searcher),
                searcher_runtime=training_runtime,
                lower_bound=signed_lower_bound,
                upper_bound=signed_upper_bound,
            )
            self.study.append_trial(trial)

            searchable_count = self.config_manager.get_searchable_configurations_count()
            should_stop = stop_search(
                n_remaining_configurations=searchable_count,
                current_runtime=self.search_timer.return_runtime(),
                max_runtime=max_runtime,
                current_iter=len(self.study.trials),
                max_searches=max_searches,
            )

            if should_stop:
                break

        progress_manager.close_progress_bar()

    def tune(
        self,
        max_searches: Optional[int] = 100,
        max_runtime: Optional[int] = None,
        searcher: Optional[QuantileConformalSearcher] = None,
        n_random_searches: int = 15,
        optimizer_framework: Optional[Literal["decaying", "fixed"]] = None,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Execute hyperparameter optimization using conformal prediction surrogate models.

        Performs intelligent hyperparameter search by randomly sampling an initial number
        of hyperparameter configurations, then activating surrogate based search according
        to the specified searcher.

        Args:
            max_searches: Maximum total configurations to search (random + conformal searches).
                Default: 100.
            max_runtime: Maximum search time in seconds. Search will terminate after this time,
                regardless of iterations. Default: None (no time limit).
            searcher: Conformal searcher object responsible for the selection of candidate
                hyperparameter configurations. When none is provided, the searcher defaults
                to a QGBM surrogate with a Thompson Sampler.
                Should you want to use a custom searcher, see confopt.selection.acquisition for
                searcher instantiation and confopt.selection.acquisition.samplers to set the
                searcher's sampler.
                Default: None.
            n_random_searches: Number of random configurations to evaluate before conformal search.
                Provides initial training data for the surrogate model. Default: 15.
            optimizer_framework: Controls how and when the surrogate model tunes its own parameters
                (this is different from tuning your target model). Options are 'decaying' for
                adaptive tuning with increasing intervals over time, 'fixed' for
                deterministic tuning at fixed intervals, or None for no tuning. Surrogate tuning
                adds computational cost and is recommended only if your target model takes more
                than 5 minutes to train. Default: None.
            random_state: Random seed for reproducible results. Default: None.
            verbose: Whether to enable progress display. Default: True.

        Example:
            Basic usage::

                import numpy as np
                from confopt.tuning import ConformalTuner
                from confopt.wrapping import FloatRange

                def objective(configuration):
                    x1 = configuration['x1']
                    x2 = configuration['x2']
                    A = 10
                    n = 2
                    return A * n + (x1**2 - A * np.cos(2 * np.pi * x1)) + (x2**2 - A * np.cos(2 * np.pi * x2))

                search_space = {
                    'x1': FloatRange(min_value=-5.12, max_value=5.12),
                    'x2': FloatRange(min_value=-5.12, max_value=5.12)
                }

                tuner = ConformalTuner(
                    objective_function=objective,
                    search_space=search_space,
                    minimize=True
                )

                tuner.tune(n_random_searches=10, max_searches=50)

                best_config = tuner.get_best_params()
                best_score = tuner.get_best_value()
        """

        if random_state is not None:
            random.seed(a=random_state)
            np.random.seed(seed=random_state)

        if searcher is None:
            searcher = QuantileConformalSearcher(
                quantile_estimator_architecture="qgbm",
                sampler=ThompsonSampler(
                    n_quantiles=4,
                    adapter="DtACI",
                    enable_optimistic_sampling=False,
                ),
                calibration_split_strategy="adaptive",
                n_calibration_folds=5,
                n_pre_conformal_trials=32,
            )

        self.initialize_tuning_resources()
        self.search_timer = RuntimeTracker()

        n_warm_starts = len(self.warm_starts) if self.warm_starts else 0
        remaining_random_searches = max(0, n_random_searches - n_warm_starts)
        if remaining_random_searches > 0:
            self.random_search(
                max_random_iter=remaining_random_searches,
                max_runtime=max_runtime,
                max_searches=max_searches,
                verbose=verbose,
            )

        self.conformal_search(
            searcher=searcher,
            verbose=verbose,
            max_searches=max_searches,
            max_runtime=max_runtime,
            optimizer_framework=optimizer_framework,
        )

    def get_best_params(self) -> Dict:
        """Retrieve the best configuration found during optimization.

        Returns the parameter configuration that achieved the optimal objective
        function value, according to the specified optimization direction.

        Returns:
            Dictionary containing the optimal parameter configuration
        """
        return self.study.get_best_configuration()

    def get_best_value(self) -> float:
        """Retrieve the best objective function value achieved during optimization.

        Returns the optimal performance value found across all evaluated
        configurations, according to the specified optimization direction.

        Returns:
            Best objective function value achieved
        """
        return self.study.get_best_performance()
