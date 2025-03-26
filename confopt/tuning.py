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
    derive_optimal_tuning_count,
)
from confopt.selection.acquisition import (
    LocallyWeightedConformalSearcher,
    QuantileConformalSearcher,
    LowerBoundSampler,
)
from confopt.data_classes import ParameterRange

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


class ConformalTuner:
    def __init__(
        self,
        objective_function: callable,
        search_space: Dict[str, ParameterRange],
        metric_optimization: Literal["direct", "inverse"],
        n_candidate_configurations: int = 10000,
        warm_start_configurations: Optional[List[Tuple[Dict, float]]] = None,
    ):
        self.objective_function = objective_function
        self._check_objective_function()

        self.search_space = search_space
        self.metric_optimization = metric_optimization
        self.n_candidate_configurations = n_candidate_configurations

        self.warm_start_configs = []
        self.warm_start_performances = []
        if warm_start_configurations:
            for config, perf in warm_start_configurations:
                self.warm_start_configs.append(config)
                self.warm_start_performances.append(perf)

        self.tuning_configurations = get_tuning_configurations(
            parameter_grid=self.search_space,
            n_configurations=self.n_candidate_configurations,
            random_state=1234,
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
        self.forbidden_indices = np.array([], dtype=int)

        self.study = Study()

        if warm_start_configurations:
            self._process_warm_start_configurations()

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

    def _random_search(
        self,
        n_searches: int,
        verbose: bool = True,
        max_runtime: Optional[int] = None,
    ) -> list[Trial]:
        rs_trials = []

        n_sample = min(n_searches, len(self.searchable_indices))
        random_indices = np.random.choice(
            self.searchable_indices, size=n_sample, replace=False
        )

        self.searchable_indices = np.setdiff1d(
            self.searchable_indices, random_indices, assume_unique=True
        )

        randomly_sampled_indices = random_indices.tolist()

        if verbose:
            iterator = tqdm(randomly_sampled_indices, desc="Random search: ")
        else:
            iterator = randomly_sampled_indices

        for config_idx, idx in enumerate(iterator):
            hyperparameter_configuration = self.tuning_configurations[idx]

            training_time_tracker = RuntimeTracker()
            validation_performance = self.objective_function(
                configuration=hyperparameter_configuration
            )
            training_time = training_time_tracker.return_runtime()

            if np.isnan(validation_performance):
                logger.debug(
                    "Obtained non-numerical performance, forbidding configuration."
                )
                self.forbidden_indices = np.append(self.forbidden_indices, idx)
                self.searchable_indices = np.setdiff1d(
                    self.searchable_indices, [idx], assume_unique=True
                )
                continue

            self.searched_indices = np.append(self.searched_indices, idx)
            self.searched_performances = np.append(
                self.searched_performances, validation_performance
            )

            rs_trials.append(
                Trial(
                    iteration=config_idx,
                    timestamp=datetime.now(),
                    configuration=hyperparameter_configuration.copy(),
                    performance=validation_performance,
                    target_model_runtime=training_time,
                    acquisition_source="rs",
                )
            )

            logger.debug(
                f"Random search iter {config_idx} performance: {validation_performance}"
            )

            if max_runtime is not None:
                if self.search_timer.return_runtime() > max_runtime:
                    raise RuntimeError(
                        "confopt preliminary random search exceeded total runtime budget. "
                        "Retry with larger runtime budget or set iteration-capped budget instead."
                    )

        return rs_trials

    @staticmethod
    def _set_conformal_validation_split(X: np.array) -> float:
        if len(X) <= 30:
            validation_split = 4 / len(X)
        else:
            validation_split = 0.20
        return validation_split

    def _process_warm_start_configurations(self):
        if not self.warm_start_configs:
            return

        warm_start_trials = []
        warm_start_indices = []

        def configs_equal(config1, config2):
            if set(config1.keys()) != set(config2.keys()):
                return False
            for key in config1:
                if config1[key] != config2[key]:
                    return False
            return True

        for i, (config, performance) in enumerate(
            zip(self.warm_start_configs, self.warm_start_performances)
        ):
            for idx, tuning_config in enumerate(self.tuning_configurations):
                if configs_equal(config, tuning_config):
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
                logger.warning(
                    f"Could not locate warm start configuration in tuning configurations: {config}"
                )

        warm_start_indices = np.array(warm_start_indices)
        warm_start_perfs = np.array(
            self.warm_start_performances[: len(warm_start_indices)]
        )

        self.searched_indices = np.append(self.searched_indices, warm_start_indices)
        self.searched_performances = np.append(
            self.searched_performances, warm_start_perfs
        )

        self.searchable_indices = np.setdiff1d(
            self.searchable_indices, warm_start_indices, assume_unique=True
        )

        self.study.batch_append_trials(trials=warm_start_trials)

        logger.debug(
            f"Added {len(warm_start_trials)} warm start configurations to search history"
        )

    def tune(
        self,
        searcher: Union[LocallyWeightedConformalSearcher, QuantileConformalSearcher],
        n_random_searches: int = 20,
        conformal_retraining_frequency: int = 1,
        searcher_tuning_framework: Optional[Literal["runtime", "ucb", "fixed"]] = None,
        verbose: bool = True,
        random_state: Optional[int] = None,
        max_iter: Optional[int] = None,
        runtime_budget: Optional[int] = None,
    ):
        self.search_timer = RuntimeTracker()

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        rs_trials = self._random_search(
            n_searches=n_random_searches,
            max_runtime=runtime_budget,
            verbose=verbose,
        )
        self.study.batch_append_trials(trials=rs_trials)

        search_model_tuning_count = 0
        scaler = StandardScaler()

        if verbose:
            if runtime_budget is not None:
                search_progress_bar = tqdm(
                    total=runtime_budget, desc="Conformal search: "
                )
            elif max_iter is not None:
                search_progress_bar = tqdm(
                    total=max_iter - n_random_searches, desc="Conformal search: "
                )

        tabularized_searched_configurations = self.tabularized_configurations[
            self.searched_indices
        ]

        max_iterations = min(
            len(self.searchable_indices),
            len(self.tuning_configurations) - n_random_searches,
        )
        for config_idx in range(max_iterations):
            if verbose:
                if runtime_budget is not None:
                    search_progress_bar.update(
                        int(self.search_timer.return_runtime()) - search_progress_bar.n
                    )
                elif max_iter is not None:
                    search_progress_bar.update(1)

            if len(self.searchable_indices) == 0:
                logger.info("All configurations have been searched. Stopping early.")
                break

            tabularized_searchable_configurations = self.tabularized_configurations[
                self.searchable_indices
            ]

            validation_split = ConformalTuner._set_conformal_validation_split(
                tabularized_searched_configurations
            )

            (
                X_train_conformal,
                y_train_conformal,
                X_val_conformal,
                y_val_conformal,
            ) = process_and_split_estimation_data(
                searched_configurations=tabularized_searched_configurations,
                searched_performances=self.searched_performances,
                train_split=(1 - validation_split),
                filter_outliers=False,
            )

            scaler.fit(X_train_conformal)
            X_train_conformal = scaler.transform(X_train_conformal)
            X_val_conformal = scaler.transform(X_val_conformal)
            tabularized_searchable_configurations = scaler.transform(
                tabularized_searchable_configurations
            )

            hit_retraining_interval = config_idx % conformal_retraining_frequency == 0
            if config_idx == 0 or hit_retraining_interval:
                runtime_tracker = RuntimeTracker()
                searcher.fit(
                    X_train=X_train_conformal,
                    y_train=y_train_conformal,
                    X_val=X_val_conformal,
                    y_val=y_val_conformal,
                    tuning_iterations=search_model_tuning_count,
                )
                searcher_runtime = runtime_tracker.return_runtime()

                if config_idx == 0:
                    first_searcher_runtime = searcher_runtime
            else:
                searcher_runtime = None

            if searcher_tuning_framework is not None:
                if searcher_tuning_framework == "runtime":
                    search_model_tuning_count = derive_optimal_tuning_count(
                        target_model_runtime=self.study.get_average_target_model_runtime(),
                        search_model_runtime=first_searcher_runtime,
                        search_model_retraining_freq=conformal_retraining_frequency,
                        search_to_baseline_runtime_ratio=0.3,
                    )
                elif searcher_tuning_framework == "fixed":
                    search_model_tuning_count = 10
                else:
                    raise ValueError("Invalid searcher tuning framework specified.")
            else:
                search_model_tuning_count = 0

            parameter_performance_bounds = searcher.predict(
                X=tabularized_searchable_configurations
            )

            minimal_searchable_idx = np.argmin(parameter_performance_bounds)
            minimal_starting_idx = self.searchable_indices[minimal_searchable_idx]
            minimal_parameter = self.tuning_configurations[minimal_starting_idx].copy()
            minimal_tabularized_configuration = tabularized_searchable_configurations[
                minimal_starting_idx
            ]

            validation_performance = self.objective_function(
                configuration=minimal_parameter
            )

            logger.debug(
                f"Conformal search iter {config_idx} performance: {validation_performance}"
            )

            if np.isnan(validation_performance):
                self.forbidden_indices = np.append(
                    self.forbidden_indices, minimal_starting_idx
                )
                self.searchable_indices = np.setdiff1d(
                    self.searchable_indices, minimal_starting_idx, assume_unique=True
                )
                continue

            if hasattr(searcher.sampler, "adapter") or hasattr(
                searcher.sampler, "adapters"
            ):
                searcher.update_interval_width(
                    sampled_idx=minimal_searchable_idx,
                    sampled_performance=validation_performance,
                    sampled_X=minimal_tabularized_configuration,
                )

            if isinstance(searcher.sampler, LowerBoundSampler):
                if (
                    searcher.predictions_per_interval[0].lower_bounds[
                        minimal_searchable_idx
                    ]
                    <= validation_performance
                    <= searcher.predictions_per_interval[0].upper_bounds[
                        minimal_searchable_idx
                    ]
                ):
                    breach = 0
                else:
                    breach = 1
            else:
                breach = None

            estimator_error = searcher.primary_estimator_error

            self.searchable_indices = self.searchable_indices[
                self.searchable_indices != minimal_starting_idx
            ]
            self.searched_indices = np.append(
                self.searched_indices, minimal_starting_idx
            )
            self.searched_performances = np.append(
                self.searched_performances, validation_performance
            )
            tabularized_searched_configurations = np.vstack(
                [
                    tabularized_searched_configurations,
                    self.tabularized_configurations[
                        minimal_starting_idx : minimal_starting_idx + 1
                    ],
                ]
            )

            self.study.append_trial(
                Trial(
                    iteration=config_idx,
                    timestamp=datetime.now(),
                    configuration=minimal_parameter.copy(),
                    performance=validation_performance,
                    acquisition_source=str(searcher),
                    searcher_runtime=searcher_runtime,
                    breached_interval=breach,
                    primary_estimator_error=estimator_error,
                )
            )

            if runtime_budget is not None:
                if self.search_timer.return_runtime() > runtime_budget:
                    if verbose:
                        if runtime_budget is not None:
                            search_progress_bar.update(
                                runtime_budget - search_progress_bar.n
                            )
                        elif max_iter is not None:
                            search_progress_bar.update(1)
                        search_progress_bar.close()
                    break
            elif max_iter is not None:
                if n_random_searches + config_idx + 1 >= max_iter:
                    if verbose:
                        if runtime_budget is not None:
                            search_progress_bar.update(
                                runtime_budget - search_progress_bar.n
                            )
                        elif max_iter is not None:
                            search_progress_bar.update(1)
                        search_progress_bar.close()
                    break

    def get_best_params(self) -> Dict:
        return self.study.get_best_configuration()

    def get_best_value(self) -> float:
        return self.study.get_best_performance()
