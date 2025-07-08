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
        return 4 / len(X) if len(X) <= 30 else 0.20

    def check_objective_function(self) -> None:
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
        n_random_searches: int = 20,
        conformal_retraining_frequency: int = 1,
        searcher: Optional[
            Union[LocallyWeightedConformalSearcher, QuantileConformalSearcher]
        ] = None,
        searcher_tuning_framework: Optional[Literal["reward_cost", "fixed"]] = None,
        random_state: Optional[int] = None,
        max_iter: Optional[int] = None,
        max_runtime: Optional[int] = None,
        verbose: bool = True,
    ):
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
                max_iter=max_iter,
                verbose=verbose,
            )

        self.conformal_search(
            searcher=searcher,
            conformal_retraining_frequency=conformal_retraining_frequency,
            verbose=verbose,
            max_iter=max_iter,
            max_runtime=max_runtime,
            searcher_tuning_framework=searcher_tuning_framework,
        )

    def get_best_params(self) -> Dict:
        return self.study.get_best_configuration()

    def get_best_value(self) -> float:
        return self.study.get_best_performance()
