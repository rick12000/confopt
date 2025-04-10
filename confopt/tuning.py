import logging
import random
from typing import Optional, Dict, Tuple, get_type_hints, Literal, Union, List

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
import inspect
from confopt.utils.encoding import ConfigurationEncoder
from confopt.utils.preprocessing import train_val_split, remove_iqr_outliers
from confopt.utils.encoding import get_tuning_configurations
from confopt.utils.tracking import (
    Trial,
    Study,
    RuntimeTracker,
)
from confopt.utils.optimization import BayesianTuner, FixedSurrogateTuner
from confopt.selection.acquisition import (
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,
    LowerBoundSampler,
    BaseConformalSearcher,
)
from confopt.wrapping import ParameterRange

logger = logging.getLogger(__name__)


def process_and_split_estimation_data(
    searched_configurations: np.array,
    searched_performances: np.array,
    train_split: float,
    filter_outliers: bool = False,
    outlier_scope: str = "top_and_bottom",
    random_state: Optional[int] = None,
) -> Tuple[np.array, np.array, np.array, np.array]:
    X = searched_configurations.copy()
    y = searched_performances.copy()
    logger.debug(f"Minimum performance in searcher data: {y.min()}")
    logger.debug(f"Maximum performance in searcher data: {y.max()}")

    if filter_outliers:
        X, y = remove_iqr_outliers(X=X, y=y, scope=outlier_scope)

    X_train, y_train, X_val, y_val = train_val_split(
        X=X,
        y=y,
        train_split=train_split,
        normalize=False,
        ordinal=False,
        random_state=random_state,
    )

    return X_train, y_train, X_val, y_val


def check_early_stopping(
    searchable_indices,
    current_runtime=None,
    runtime_budget=None,
    current_iter=None,
    max_iter=None,
    n_random_searches=None,
):
    if len(searchable_indices) == 0:
        return True, "All configurations have been searched"

    if runtime_budget is not None and current_runtime is not None:
        if current_runtime > runtime_budget:
            return True, f"Runtime budget ({runtime_budget}) exceeded"

    if (
        max_iter is not None
        and current_iter is not None
        and n_random_searches is not None
    ):
        if n_random_searches + current_iter >= max_iter:
            return True, f"Maximum iterations ({max_iter}) reached"

    return False


class ConformalTuner:
    def __init__(
        self,
        objective_function: callable,
        search_space: Dict[str, ParameterRange],
        metric_optimization: Literal["maximize", "minimize"],
        n_candidate_configurations: int = 10000,
        warm_start_configurations: Optional[List[Tuple[Dict, float]]] = None,
    ):
        self.objective_function = objective_function
        self._check_objective_function()

        self.search_space = search_space
        self.metric_sign = -1 if metric_optimization == "maximize" else 1
        self.n_candidate_configurations = n_candidate_configurations
        self.warm_start_configurations = warm_start_configurations

    @staticmethod
    def _set_conformal_validation_split(X: np.array) -> float:
        if len(X) <= 30:
            validation_split = 4 / len(X)
        else:
            validation_split = 0.20
        return validation_split

    def _check_objective_function(self):
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

    def _initialize_tuning_resources(self):
        self.warm_start_configs = []
        self.warm_start_performances = []
        if self.warm_start_configurations:
            for config, perf in self.warm_start_configurations:
                self.warm_start_configs.append(config)
                self.warm_start_performances.append(perf)

        self.tuning_configurations = get_tuning_configurations(
            parameter_grid=self.search_space,
            n_configurations=self.n_candidate_configurations,
            random_state=None,
            warm_start_configs=self.warm_start_configs,
        )

        self.encoder = ConfigurationEncoder()
        self.encoder.fit(self.tuning_configurations)
        self.tabularized_configurations = self.encoder.transform(
            self.tuning_configurations
        ).to_numpy()

        self.searchable_indices = np.arange(len(self.tuning_configurations))
        self.searched_indices = np.array([], dtype=int)
        self.searched_performances = np.array([])

        self.study = Study()

        if self.warm_start_configurations:
            self._process_warm_start_configurations()

    def _process_warm_start_configurations(self):
        warm_start_trials = []
        warm_start_indices = []

        for i, (config, performance) in enumerate(
            zip(self.warm_start_configs, self.warm_start_performances)
        ):
            for idx, tuning_config in enumerate(self.tuning_configurations):
                if config == tuning_config:
                    warm_start_indices.append(idx)

                    warm_start_trials.append(
                        Trial(
                            iteration=i,
                            timestamp=datetime.now(),
                            configuration=config.copy(),
                            performance=performance,
                            acquisition_source="warm_start",
                        )
                    )
                    break
            else:
                raise ValueError(
                    f"Could not locate warm start configuration in tuning configurations: {config}"
                )

        warm_start_indices = np.array(object=warm_start_indices)
        warm_start_performances = np.array(
            object=self.warm_start_performances[: len(warm_start_indices)]
        )

        self.searched_indices = np.append(
            arr=self.searched_indices, values=warm_start_indices
        )
        self.searched_performances = np.append(
            arr=self.searched_performances, values=warm_start_performances
        )

        self.searchable_indices = np.setdiff1d(
            ar1=self.searchable_indices, ar2=warm_start_indices, assume_unique=True
        )

        self.study.batch_append_trials(trials=warm_start_trials)

        logger.debug(
            f"Added {len(warm_start_trials)} warm start configurations to search history"
        )

    def _evaluate_configuration(self, configuration):
        runtime_tracker = RuntimeTracker()
        performance = self.objective_function(configuration=configuration)
        runtime = runtime_tracker.return_runtime()
        return performance, runtime

    def _update_search_state(self, config_idx, performance):
        self.searched_indices = np.append(self.searched_indices, config_idx)
        self.searched_performances = np.append(self.searched_performances, performance)

        self.searchable_indices = np.setdiff1d(
            self.searchable_indices, [config_idx], assume_unique=True
        )

    def _random_search(
        self, n_searches: int, verbose: bool = True, max_runtime: Optional[int] = None
    ) -> list[Trial]:
        rs_trials = []
        adj_n_searches = min(n_searches, len(self.searchable_indices))
        randomly_sampled_indices = np.random.choice(
            a=self.searchable_indices, size=adj_n_searches, replace=False
        ).tolist()

        progress_iter = (
            tqdm(iterable=randomly_sampled_indices, desc="Random search: ")
            if verbose
            else randomly_sampled_indices
        )

        for configuration_idx in progress_iter:
            hyperparameter_configuration = self.tuning_configurations[configuration_idx]
            validation_performance, training_time = self._evaluate_configuration(
                hyperparameter_configuration
            )

            if np.isnan(validation_performance):
                logger.debug(
                    "Obtained non-numerical performance, forbidding configuration."
                )
                self.searchable_indices = np.setdiff1d(
                    ar1=self.searchable_indices,
                    ar2=[configuration_idx],
                    assume_unique=True,
                )
                continue

            self._update_search_state(
                config_idx=configuration_idx,
                performance=validation_performance,
            )

            # Create trial object separately
            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=hyperparameter_configuration.copy(),
                performance=validation_performance,
                acquisition_source="rs",
                target_model_runtime=training_time,
            )
            rs_trials.append(trial)

            logger.debug(
                f"Random search iter {len(rs_trials)} performance: {validation_performance}"
            )

            # Moved early stopping check to end of loop
            stop = check_early_stopping(
                searchable_indices=self.searchable_indices,
                current_runtime=(
                    self.search_timer.return_runtime() if max_runtime else None
                ),
                runtime_budget=max_runtime,
            )
            if stop:
                raise RuntimeError(
                    "confopt preliminary random search exceeded total runtime budget. "
                    "Retry with larger runtime budget or set iteration-capped budget instead."
                )

        return rs_trials

    def _select_next_configuration_idx(
        self, searcher, tabularized_searchable_configurations
    ):
        parameter_performance_bounds = searcher.predict(
            X=tabularized_searchable_configurations
        )
        config_idx = self.searchable_indices[np.argmin(parameter_performance_bounds)]
        return config_idx

    def _conformal_search(
        self,
        searcher: BaseConformalSearcher,
        n_random_searches,
        conformal_retraining_frequency,
        tabularized_searched_configurations,
        verbose,
        max_iter,
        runtime_budget,
        searcher_tuning_framework=None,
    ):
        # Setup progress bar directly in this method
        progress_bar = None
        if verbose:
            if runtime_budget is not None:
                progress_bar = tqdm(total=runtime_budget, desc="Conformal search: ")
            elif max_iter is not None:
                progress_bar = tqdm(
                    total=max_iter - n_random_searches, desc="Conformal search: "
                )

        scaler = StandardScaler()
        max_iterations = min(
            len(self.searchable_indices),
            len(self.tuning_configurations) - n_random_searches,
        )

        if searcher_tuning_framework == "reward_cost":
            tuning_optimizer = BayesianTuner(
                max_tuning_count=20,
                max_tuning_interval=15,
                conformal_retraining_frequency=conformal_retraining_frequency,
                min_observations=5,  # Updated to match the new default
                exploration_weight=0.1,
                random_state=42,
            )
        elif searcher_tuning_framework == "fixed":
            tuning_optimizer = FixedSurrogateTuner(
                n_tuning_episodes=10,
                tuning_interval=3 * conformal_retraining_frequency,
                conformal_retraining_frequency=conformal_retraining_frequency,
            )
        elif searcher_tuning_framework is None:
            tuning_optimizer = FixedSurrogateTuner(
                n_tuning_episodes=0,
                tuning_interval=conformal_retraining_frequency,
                conformal_retraining_frequency=conformal_retraining_frequency,
            )
        else:
            raise ValueError(
                "searcher_tuning_framework must be either 'reward_cost', 'fixed', or None."
            )

        search_model_retuning_frequency = 1
        search_model_tuning_count = 0
        searcher_error_history = []
        for search_iter in range(max_iterations):
            # Update progress bar
            if progress_bar:
                if runtime_budget is not None:
                    progress_bar.update(
                        int(self.search_timer.return_runtime()) - progress_bar.n
                    )
                elif max_iter is not None:
                    progress_bar.update(1)

            # Prepare data for conformal search
            tabularized_searchable_configurations = self.tabularized_configurations[
                self.searchable_indices
            ]

            # Directly implement _prepare_conformal_data logic here
            validation_split = self._set_conformal_validation_split(
                X=tabularized_searched_configurations
            )
            X_train, y_train, X_val, y_val = process_and_split_estimation_data(
                searched_configurations=tabularized_searched_configurations,
                searched_performances=self.searched_performances,
                train_split=(1 - validation_split),
                filter_outliers=False,
            )
            y_train = y_train * self.metric_sign
            y_val = y_val * self.metric_sign

            # Scale the data
            scaler.fit(X=X_train)
            X_train = scaler.transform(X=X_train)
            X_val = scaler.transform(X=X_val)
            tabularized_searchable_configurations = scaler.transform(
                X=tabularized_searchable_configurations
            )

            # Retrain the searcher if needed
            searcher_runtime = None
            if search_iter == 0 or search_iter % conformal_retraining_frequency == 0:
                if (
                    search_model_retuning_frequency % conformal_retraining_frequency
                    != 0
                ):
                    raise ValueError(
                        "search_model_retuning_frequency must be a multiple of conformal_retraining_frequency."
                    )

                runtime_tracker = RuntimeTracker()
                searcher.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    tuning_iterations=search_model_tuning_count,
                )
                searcher_runtime = runtime_tracker.return_runtime()
                searcher_error_history.append(searcher.primary_estimator_error)

                if len(searcher_error_history) > 1:
                    error_improvement = max(
                        0, searcher_error_history[-2] - searcher_error_history[-1]
                    )
                    try:
                        normalized_searcher_runtime = (
                            searcher_runtime
                            / self.study.get_average_target_model_runtime()
                        )
                    except ZeroDivisionError:
                        normalized_searcher_runtime = 0

                    # Pass the search iteration to update
                    tuning_optimizer.update(
                        arm=(
                            search_model_tuning_count,
                            search_model_retuning_frequency,
                        ),
                        reward=error_improvement,
                        cost=normalized_searcher_runtime,
                        search_iter=search_iter,  # Include search iteration
                    )

                (
                    search_model_tuning_count,
                    search_model_retuning_frequency,
                ) = tuning_optimizer.select_arm()

            # Get performance bounds and select next configuration to evaluate
            config_idx = self._select_next_configuration_idx(
                searcher=searcher,
                tabularized_searchable_configurations=tabularized_searchable_configurations,
            )
            minimal_parameter = self.tuning_configurations[config_idx].copy()

            # Evaluate the selected configuration
            validation_performance, _ = self._evaluate_configuration(minimal_parameter)
            logger.debug(
                f"Conformal search iter {search_iter} performance: {validation_performance}"
            )

            if np.isnan(validation_performance):
                self.searchable_indices = np.setdiff1d(
                    ar1=self.searchable_indices, ar2=[config_idx], assume_unique=True
                )
                continue

            # Use the new update method to update both stagnation and interval width
            transformed_X = scaler.transform(
                self.encoder.transform([minimal_parameter]).to_numpy(),
            )
            searcher.update(
                X=transformed_X, y_true=self.metric_sign * validation_performance
            )

            # TODO: TEMP FOR PAPER
            breach = None
            if (
                isinstance(searcher.sampler, LowerBoundSampler)
                and searcher.sampler.adapter is not None
                and len(searcher.sampler.adapter.error_history) > 0
            ):
                breach = searcher.sampler.adapter.error_history[-1]
            estimator_error = searcher.primary_estimator_error

            # Update search state and record trial
            self.searchable_indices = self.searchable_indices[
                self.searchable_indices != config_idx
            ]

            self._update_search_state(
                config_idx=config_idx,
                performance=validation_performance,
            )

            # Create trial object separately
            trial = Trial(
                iteration=len(self.study.trials),
                timestamp=datetime.now(),
                configuration=minimal_parameter.copy(),
                performance=validation_performance,
                acquisition_source=str(searcher),
                searcher_runtime=searcher_runtime,
                breached_interval=breach,
                primary_estimator_error=estimator_error,
            )
            self.study.append_trial(trial)

            # Update tabularized searched configurations
            tabularized_searched_configurations = np.vstack(
                tup=[
                    tabularized_searched_configurations,
                    self.tabularized_configurations[config_idx].reshape((1, -1)),
                ]
            )

            # Moved early stopping check to end of loop
            stop = check_early_stopping(
                searchable_indices=self.searchable_indices,
                current_runtime=self.search_timer.return_runtime(),
                runtime_budget=runtime_budget,
                current_iter=search_iter + 1,
                max_iter=max_iter,
                n_random_searches=n_random_searches,
            )
            if stop:
                break

        # Close progress bar if it exists
        if progress_bar:
            if runtime_budget is not None:
                progress_bar.update(n=runtime_budget - progress_bar.n)
            elif max_iter is not None:
                progress_bar.update(
                    n=max(
                        0,
                        max_iter
                        - n_random_searches
                        - len(self.study.trials)
                        + n_random_searches,
                    )
                )
            progress_bar.close()

    def tune(
        self,
        n_random_searches: int = 20,
        conformal_retraining_frequency: int = 1,
        searcher: Optional[
            Union[LocallyWeightedConformalSearcher, QuantileConformalSearcher]
        ] = None,
        searcher_tuning_framework: Optional[Literal["reward_cost", "fixed"]] = None,
        random_state: Optional[int] = None,
        max_iter: Optional[int] = None,
        runtime_budget: Optional[int] = None,
        verbose: bool = True,
    ):
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

        self._initialize_tuning_resources()
        self.search_timer = RuntimeTracker()

        if random_state is not None:
            random.seed(a=random_state)
            np.random.seed(seed=random_state)

        # Perform random search
        rs_trials = self._random_search(
            n_searches=n_random_searches,
            max_runtime=runtime_budget,
            verbose=verbose,
        )
        self.study.batch_append_trials(trials=rs_trials)

        # Setup for conformal search
        tabularized_searched_configurations = self.tabularized_configurations[
            self.searched_indices
        ]

        self._conformal_search(
            searcher=searcher,
            n_random_searches=n_random_searches,
            conformal_retraining_frequency=conformal_retraining_frequency,
            tabularized_searched_configurations=tabularized_searched_configurations,
            verbose=verbose,
            max_iter=max_iter,
            runtime_budget=runtime_budget,
            searcher_tuning_framework=searcher_tuning_framework,
        )

    def get_best_params(self) -> Dict:
        return self.study.get_best_configuration()

    def get_best_value(self) -> float:
        return self.study.get_best_performance()
