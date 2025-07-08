import logging
import random
from typing import Optional, Dict, Tuple, get_type_hints, Literal, Union, List
from confopt.wrapping import ParameterRange

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
import inspect
from confopt.utils.preprocessing import train_val_split, remove_iqr_outliers
from confopt.utils.tracking import (
    Trial,
    Study,
    RuntimeTracker,
    DynamicConfigurationManager,
    StaticConfigurationManager,
    ProgressBarManager,
)
from confopt.utils.optimization import BayesianSearcherOptimizer, FixedSearcherOptimizer
from confopt.selection.acquisition import (
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,
    LowerBoundSampler,
    PessimisticLowerBoundSampler,
    BaseConformalSearcher,
)

logger = logging.getLogger(__name__)


def stop_search(
    n_remaining_configurations: int,
    current_iter: int,
    current_runtime: float,
    max_runtime: Optional[float] = None,
    max_iter: Optional[int] = None,
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
        max_iter: Maximum allowed iterations, None for no limit

    Returns:
        True if any stopping criterion is met, False otherwise
    """
    if n_remaining_configurations == 0:
        return True

    if max_runtime is not None:
        if current_runtime >= max_runtime:
            return True

    if max_iter is not None:
        if current_iter >= max_iter:
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
        metric_optimization: Whether to 'maximize' or 'minimize' the objective function
        n_candidate_configurations: Size of discrete configuration pool for selection
        warm_start_configurations: Pre-evaluated (configuration, performance) pairs
        dynamic_sampling: Whether to dynamically resample configuration candidates

    Attributes:
        study: Container for storing trial results and optimization history
        config_manager: Handles configuration sampling and tracking
        search_timer: Tracks total optimization runtime
        error_history: Sequence of conformal model prediction errors
    """

    def __init__(
        self,
        objective_function: callable,
        search_space: Dict[str, ParameterRange],
        metric_optimization: Literal["maximize", "minimize"],
        n_candidate_configurations: int = 10000,
        warm_start_configurations: Optional[List[Tuple[Dict, float]]] = None,
        dynamic_sampling: bool = False,
    ) -> None:
        self.objective_function = objective_function
        self.check_objective_function()

        self.search_space = search_space
        self.metric_optimization = metric_optimization
        self.metric_sign = -1 if metric_optimization == "maximize" else 1
        self.warm_start_configurations = warm_start_configurations
        self.n_candidate_configurations = n_candidate_configurations
        self.dynamic_sampling = dynamic_sampling

    @staticmethod
    def _set_conformal_validation_split(X: np.array) -> float:
        """Determine appropriate validation split ratio for conformal model training.

        Implements adaptive validation split sizing based on available data volume.
        Uses larger validation splits for small datasets to ensure statistical validity
        of conformal predictions, while using standard splits for larger datasets.

        Args:
            X: Feature matrix of evaluated configurations

        Returns:
            Validation split ratio between 0 and 1
        """
        return 4 / len(X) if len(X) <= 30 else 0.20

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
        for idx, (config, performance) in enumerate(self.warm_start_configurations):
            self.config_manager.mark_as_searched(config, performance)
            trial = Trial(
                iteration=idx,
                timestamp=datetime.now(),
                configuration=config.copy(),
                performance=performance,
                acquisition_source="warm_start",
            )
            self.study.append_trial(trial)

    def initialize_tuning_resources(self) -> None:
        """Initialize core optimization components and data structures.

        Sets up the study container for trial tracking, configuration manager for
        handling search space sampling, and processes any warm start configurations.
        The configuration manager type (static vs dynamic) determines whether
        the candidate pool is fixed or adaptively resampled during optimization.
        """
        self.study = Study(metric_optimization=self.metric_optimization)

        if self.dynamic_sampling:
            self.config_manager = DynamicConfigurationManager(
                search_space=self.search_space,
                n_candidate_configurations=self.n_candidate_configurations,
            )
        else:
            self.config_manager = StaticConfigurationManager(
                search_space=self.search_space,
                n_candidate_configurations=self.n_candidate_configurations,
            )

        if self.warm_start_configurations:
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
        max_iter: Optional[int] = None,
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
            max_iter: Optional total iteration limit
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
                performance=validation_performance,
                acquisition_source="rs",
                target_model_runtime=training_time,
            )
            self.study.append_trial(trial)

            searchable_count = len(self.config_manager.get_searchable_configurations())
            current_runtime = self.search_timer.return_runtime()

            stop = stop_search(
                n_remaining_configurations=searchable_count,
                current_runtime=current_runtime,
                max_runtime=max_runtime,
                current_iter=len(self.study.trials),
                max_iter=max_iter,
            )
            if stop:
                break

    def setup_conformal_search_resources(
        self,
        verbose: bool,
        max_runtime: Optional[int],
        max_iter: Optional[int],
    ) -> Tuple[ProgressBarManager, float]:
        """Initialize progress tracking and iteration limits for conformal search.

        Sets up the progress bar manager for displaying search progress and calculates
        the maximum number of conformal search iterations based on total limits and
        already completed trials from previous phases.

        Args:
            verbose: Whether to display progress information
            max_runtime: Optional maximum runtime in seconds
            max_iter: Optional maximum total iterations

        Returns:
            Tuple of (progress_manager, conformal_max_iter)
        """
        progress_manager = ProgressBarManager(verbose=verbose)
        progress_manager.create_progress_bar(
            max_runtime=max_runtime,
            max_iter=max_iter,
            current_trials=len(self.study.trials),
            description="Conformal search",
        )

        conformal_max_iter = (
            max_iter - len(self.study.trials) if max_iter is not None else float("inf")
        )

        return progress_manager, conformal_max_iter

    def initialize_searcher_optimizer(
        self,
        searcher_tuning_framework: Optional[str],
        conformal_retraining_frequency: int,
    ):
        """Initialize multi-armed bandit optimizer for searcher parameter tuning.

        Creates an optimizer instance for automatically tuning searcher parameters
        such as retraining frequency and internal tuning iterations. The optimizer
        uses reward-cost trade-offs to balance prediction improvement against
        computational overhead.

        Args:
            searcher_tuning_framework: Tuning strategy ('reward_cost', 'fixed', None)
            conformal_retraining_frequency: Base retraining frequency for validation

        Returns:
            Configured optimizer instance
        """
        if searcher_tuning_framework == "reward_cost":
            optimizer = BayesianSearcherOptimizer(
                max_tuning_count=20,
                max_tuning_interval=15,
                conformal_retraining_frequency=conformal_retraining_frequency,
                min_observations=5,
                exploration_weight=0.1,
                random_state=42,
            )
        elif searcher_tuning_framework == "fixed":
            optimizer = FixedSearcherOptimizer(
                n_tuning_episodes=10,
                tuning_interval=3 * conformal_retraining_frequency,
                conformal_retraining_frequency=conformal_retraining_frequency,
            )
        elif searcher_tuning_framework is None:
            optimizer = FixedSearcherOptimizer(
                n_tuning_episodes=0,
                tuning_interval=conformal_retraining_frequency,
                conformal_retraining_frequency=conformal_retraining_frequency,
            )
        else:
            raise ValueError(
                "searcher_tuning_framework must be either 'reward_cost', 'fixed', or None."
            )
        return optimizer

    def prepare_searcher_data(
        self,
        validation_split: float,
        filter_outliers: bool = False,
        outlier_scope: str = "top_and_bottom",
        random_state: Optional[int] = None,
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        """Prepare training and validation data for conformal model fitting.

        Processes the accumulated search history into properly formatted training
        and validation sets for conformal prediction model training. Includes
        optional outlier filtering and applies metric sign transformation for
        consistent optimization direction handling.

        Args:
            validation_split: Fraction of data reserved for validation
            filter_outliers: Whether to remove statistical outliers
            outlier_scope: Outlier removal scope ('top_and_bottom', 'top', 'bottom')
            random_state: Random seed for reproducible data splits

        Returns:
            Tuple of (X_train, y_train, X_val, y_val) arrays
        """
        searched_configs = self.config_manager.tabularize_configs(
            self.config_manager.searched_configs
        )
        searched_performances = np.array(self.config_manager.searched_performances)

        X = searched_configs.copy()
        y = searched_performances.copy()
        logger.debug(f"Minimum performance in searcher data: {y.min()}")
        logger.debug(f"Maximum performance in searcher data: {y.max()}")

        if filter_outliers:
            X, y = remove_iqr_outliers(X=X, y=y, scope=outlier_scope)

        X_train, y_train, X_val, y_val = train_val_split(
            X=X,
            y=y,
            train_split=(1 - validation_split),
            normalize=False,
            ordinal=False,
            random_state=random_state,
        )

        y_train = y_train * self.metric_sign
        y_val = y_val * self.metric_sign

        return X_train, y_train, X_val, y_val

    def fit_transform_searcher_data(
        self, X_train: np.array, X_val: np.array
    ) -> Tuple[StandardScaler, np.array, np.array]:
        """Fit feature scaler and transform training and validation data.

        Applies standard scaling (zero mean, unit variance) to feature matrices
        to ensure consistent scaling for conformal prediction models. The scaler
        is fitted only on training data to prevent data leakage.

        Args:
            X_train: Training feature matrix
            X_val: Validation feature matrix

        Returns:
            Tuple of (fitted_scaler, X_train_scaled, X_val_scaled)
        """
        scaler = StandardScaler()
        scaler.fit(X=X_train)
        X_train_scaled = scaler.transform(X=X_train)
        X_val_scaled = scaler.transform(X=X_val)
        return scaler, X_train_scaled, X_val_scaled

    def retrain_searcher(
        self,
        searcher: BaseConformalSearcher,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        tuning_count: int,
    ) -> Tuple[float, float]:
        """Train conformal prediction searcher on accumulated data.

        Fits the conformal prediction model using current training and validation
        data, tracking training time and model performance for adaptive parameter
        optimization. The tuning_count parameter controls internal hyperparameter
        optimization within the searcher.

        Args:
            searcher: Conformal searcher instance to train
            X_train: Training feature matrix
            y_train: Training target values (sign-adjusted)
            X_val: Validation feature matrix
            y_val: Validation target values (sign-adjusted)
            tuning_count: Number of internal tuning iterations

        Returns:
            Tuple of (training_runtime, estimator_error)
        """
        runtime_tracker = RuntimeTracker()
        searcher.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tuning_iterations=tuning_count,
        )

        training_runtime = runtime_tracker.return_runtime()
        estimator_error = searcher.primary_estimator_error
        self.error_history.append(estimator_error)

        return training_runtime, estimator_error

    def select_next_configuration(
        self,
        searcher: BaseConformalSearcher,
        searchable_configs: List,
        transformed_configs: np.array,
    ) -> Tuple[Dict, int]:
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

    def calculate_breach_if_applicable(
        self,
        searcher: BaseConformalSearcher,
        transformed_config: np.array,
        performance: float,
    ) -> Optional[float]:
        """Calculate prediction interval breach if supported by searcher.

        Computes how much the observed performance violates the predicted confidence
        interval for configurations using lower bound samplers. This metric helps
        assess conformal model calibration and prediction quality.

        Args:
            searcher: Conformal searcher instance
            transformed_config: Scaled configuration features
            performance: Observed performance value (sign-adjusted)

        Returns:
            Breach amount if applicable, None otherwise
        """
        if isinstance(
            searcher.sampler, (LowerBoundSampler, PessimisticLowerBoundSampler)
        ):
            breach = searcher.calculate_breach(X=transformed_config, y_true=performance)
        else:
            breach = None

        return breach

    def update_optimizer_parameters(
        self,
        optimizer,
        training_runtime: float,
        tuning_count: int,
        searcher_retuning_frequency: int,
        search_iter: int,
    ) -> Tuple[int, int]:
        """Update multi-armed bandit optimizer and select new parameter values.

        Provides feedback to the parameter optimizer about the effectiveness of
        current searcher settings, using prediction error improvement as reward
        and normalized training time as cost. Then selects new parameter values
        for subsequent iterations.

        Args:
            optimizer: Multi-armed bandit optimizer instance
            training_runtime: Time spent training the conformal model
            tuning_count: Current internal tuning iterations
            searcher_retuning_frequency: Current retraining frequency
            search_iter: Current search iteration number

        Returns:
            Tuple of (new_tuning_count, new_searcher_retuning_frequency)
        """
        has_multiple_errors = len(self.error_history) > 1
        if has_multiple_errors:
            error_improvement = max(0, self.error_history[-2] - self.error_history[-1])

            normalized_runtime = 0
            try:
                normalized_runtime = (
                    training_runtime / self.study.get_average_target_model_runtime()
                )
            except ZeroDivisionError:
                normalized_runtime = 0

            optimizer.update(
                arm=(tuning_count, searcher_retuning_frequency),
                reward=error_improvement,
                cost=normalized_runtime,
                search_iter=search_iter,
            )

        new_tuning_count, new_searcher_retuning_frequency = optimizer.select_arm()
        return new_tuning_count, new_searcher_retuning_frequency

    def conformal_search(
        self,
        searcher: BaseConformalSearcher,
        conformal_retraining_frequency: int,
        verbose: bool,
        max_iter: Optional[int],
        max_runtime: Optional[int],
        searcher_tuning_framework: Optional[str] = None,
    ) -> None:
        """Execute conformal prediction-guided hyperparameter search.

        Implements the main conformal search loop that iteratively trains conformal
        prediction models, selects promising configurations based on uncertainty
        quantification, and updates the models with new observations. The method
        supports adaptive parameter tuning through multi-armed bandit optimization.

        Args:
            searcher: Conformal prediction searcher for configuration selection
            conformal_retraining_frequency: Base frequency for model retraining
            verbose: Whether to display search progress
            max_iter: Maximum total iterations including previous phases
            max_runtime: Maximum total runtime budget in seconds
            searcher_tuning_framework: Parameter tuning strategy
        """
        progress_manager, conformal_max_iter = self.setup_conformal_search_resources(
            verbose, max_runtime, max_iter
        )
        optimizer = self.initialize_searcher_optimizer(
            searcher_tuning_framework=searcher_tuning_framework,
            conformal_retraining_frequency=conformal_retraining_frequency,
        )

        tuning_count = 0
        searcher_retuning_frequency = conformal_retraining_frequency
        self.error_history = []
        for search_iter in range(conformal_max_iter):
            progress_manager.update_progress(
                current_runtime=(
                    self.search_timer.return_runtime() if max_runtime else None
                ),
                iteration_count=1 if max_iter else 0,
            )

            tabularized_searched_configs = self.config_manager.tabularize_configs(
                self.config_manager.searched_configs
            )
            validation_split = self._set_conformal_validation_split(
                X=tabularized_searched_configs
            )
            X_train, y_train, X_val, y_val = self.prepare_searcher_data(
                validation_split
            )
            scaler, X_train_scaled, X_val_scaled = self.fit_transform_searcher_data(
                X_train, X_val
            )
            searchable_configs = self.config_manager.get_searchable_configurations()
            X_searchable = self.config_manager.tabularize_configs(searchable_configs)
            X_searchable_scaled = scaler.transform(X=X_searchable)

            if search_iter == 0 or search_iter % conformal_retraining_frequency == 0:
                training_runtime, estimator_error = self.retrain_searcher(
                    searcher, X_train_scaled, y_train, X_val_scaled, y_val, tuning_count
                )

                (
                    tuning_count,
                    searcher_retuning_frequency,
                ) = self.update_optimizer_parameters(
                    optimizer,
                    training_runtime,
                    tuning_count,
                    searcher_retuning_frequency,
                    search_iter,
                )
                if (
                    not searcher_retuning_frequency % conformal_retraining_frequency
                    == 0
                ):
                    raise ValueError(
                        "searcher_retuning_frequency must be a multiple of conformal_retraining_frequency."
                    )

            next_config = self.select_next_configuration(
                searcher, searchable_configs, X_searchable_scaled
            )
            performance, _ = self._evaluate_configuration(next_config)
            if np.isnan(performance):
                self.config_manager.add_to_banned_configurations(next_config)
                continue

            transformed_config = scaler.transform(
                self.config_manager.tabularize_configs([next_config])
            )
            signed_performance = self.metric_sign * performance
            searcher.update(X=transformed_config, y_true=signed_performance)

            breach = self.calculate_breach_if_applicable(
                searcher, transformed_config, signed_performance
            )

            self.config_manager.mark_as_searched(next_config, performance)
            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=next_config.copy(),
                performance=performance,
                acquisition_source=str(searcher),
                searcher_runtime=training_runtime,
                breached_interval=breach,
                primary_estimator_error=estimator_error,
            )
            self.study.append_trial(trial)

            searchable_count = len(self.config_manager.get_searchable_configurations())
            should_stop = stop_search(
                n_remaining_configurations=searchable_count,
                current_runtime=self.search_timer.return_runtime(),
                max_runtime=max_runtime,
                current_iter=len(self.study.trials),
                max_iter=max_iter,
            )
            if should_stop:
                break

        progress_manager.close_progress_bar()

    def tune(
        self,
        max_searches: Optional[int] = 100,
        max_runtime: Optional[int] = None,
        searcher: Optional[
            Union[LocallyWeightedConformalSearcher, QuantileConformalSearcher]
        ] = None,
        n_random_searches: int = 15,
        conformal_retraining_frequency: int = 1,
        optimizer: Optional[Literal["reward_cost", "fixed"]] = None,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Execute hyperparameter optimization using conformal prediction surrogate models.

        Performs intelligent hyperparameter search through two phases: random exploration
        for baseline data, then conformal prediction-guided optimization using uncertainty
        quantification to select promising configurations.

        Args:
            max_searches (Optional[int], default=100): Maximum total configurations to search (random + conformal searches).
            max_runtime (Optional[int], default=None): Maximum search time in seconds. Search will terminate after this time, regardless of iterations.
            searcher (Optional[object], default=None): Conformal acquisition function. Defaults to `QuantileConformalSearcher`
                with `LowerBoundSampler`. You should not need to change this, as the default searcher performs
                best across most tasks in offline benchmarks. Should you want to use a different searcher, you can pass any subclass of `BaseConformalSearcher`.
                See `confopt.selection.acquisition` for all available searchers and
                `confopt.selection.acquisition.samplers` to set the searcher's sampler.

                Example of a searcher initialization to pass to this argument:
                ```python
                searcher = QuantileConformalSearcher(
                    quantile_estimator_architecture='qrf',
                    sampler=LowerBoundSampler(interval_width=0.1)
                )
                ```
            n_random_searches (int, default=15): Number of random configurations to evaluate before conformal search.
                Provides initial training data for the surrogate model.
            conformal_retraining_frequency (int, default=1): How often the conformal surrogate model retrains
                (the model will retrain every `conformal_retraining_frequency`-th search iteration).
                Recommended values are `1`: if your target model takes >1 min to train. `2`-`5`: if your target model is very
                small, to reduce computational overhead.
            optimizer (Optional[str], default=None): Controls how and when the surrogate model tunes its own parameters
                (this is different from tuning your target model).
                Options are (1) `reward_cost`: Bayesian selection balancing prediction improvement vs cost.
                (2) `fixed`: Deterministic tuning at fixed intervals. (3) `None`: No tuning. Surrogate tuning
                adds computational cost and is recommended only if your target
                model takes more than 1–5 minutes to train.
            random_state (Optional[int], default=None): Random seed for reproducible results.
            verbose (bool, default=True): Whether to enable progress display.

        Example:
            ```python
            from confopt.tuning import ConformalTuner
            from confopt.wrapping import ParameterRange

            def objective(configuration):
                model = SomeModel(
                    learning_rate=configuration['lr'],
                    hidden_units=configuration['units']
                )
                return model.evaluate()  # validation accuracy

            search_space = {
                'lr': ParameterRange(0.001, 0.1, log_scale=True),
                'units': ParameterRange(32, 512, integer=True)
            }

            tuner = ConformalTuner(
                objective_function=objective,
                search_space=search_space,
                metric_optimization='maximize'
            )

            tuner.tune(n_random_searches=25, max_iter=100)

            best_config = tuner.get_best_params()
            best_score = tuner.get_best_value()
            ```
        """

        if random_state is not None:
            random.seed(a=random_state)
            np.random.seed(seed=random_state)

        if searcher is None:
            searcher = QuantileConformalSearcher(
                quantile_estimator_architecture="qrf",
                sampler=LowerBoundSampler(
                    interval_width=0.05,
                    adapter="DtACI",
                    beta_decay="logarithmic_decay",
                    c=1,
                ),
                n_pre_conformal_trials=20,
            )

        self.initialize_tuning_resources()
        self.search_timer = RuntimeTracker()

        n_warm_starts = (
            len(self.warm_start_configurations) if self.warm_start_configurations else 0
        )
        remaining_random_searches = max(0, n_random_searches - n_warm_starts)
        if remaining_random_searches > 0:
            self.random_search(
                max_random_iter=remaining_random_searches,
                max_runtime=max_runtime,
                max_iter=max_searches,
                verbose=verbose,
            )

        self.conformal_search(
            searcher=searcher,
            conformal_retraining_frequency=conformal_retraining_frequency,
            verbose=verbose,
            max_iter=max_searches,
            max_runtime=max_runtime,
            searcher_tuning_framework=optimizer,
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
