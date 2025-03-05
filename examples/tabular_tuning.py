# %%

from sklearn.datasets import fetch_california_housing
from confopt.tuning import ObjectiveConformalSearcher
from confopt.acquisition import (
    LocallyWeightedConformalSearcher,
    # MultiFitQuantileConformalSearcher,
    # SingleFitQuantileConformalSearcher,
    UCBSampler,
    # ThompsonSampler,
)

import numpy as np
from hashlib import sha256
import random


# Set up toy data:
X, y = fetch_california_housing(return_X_y=True)
split_idx = int(len(X) * 0.5)
X_train, y_train = X[:split_idx, :], y[:split_idx]
X_val, y_val = X[split_idx:, :], y[split_idx:]

# Define parameter search space:
parameter_search_space = {
    "param1__range_float": [0, 100],
    "param2__range_float": [0, 100],
    "param3__range_float": [0, 100],
    "param4__range_float": [0, 100],
    "param5__range_float": [0, 100],
    "param6__range_float": [0, 100],
    "param7__range_float": [0, 100],
}

confopt_params = {}
for param_name, param_values in parameter_search_space.items():
    if "__range_int" in param_name:
        confopt_params[param_name.replace("__range_int", "")] = list(
            range(param_values[0], param_values[1] + 1)
        )
    elif "__range_float" in param_name:
        confopt_params[param_name.replace("__range_float", "")] = [
            random.uniform(param_values[0], param_values[1]) for _ in range(10000)
        ]
    else:
        confopt_params[param_name] = param_values


def noisy_rastrigin(x, A=20, noise_seed=42, noise=0):
    n = len(x)
    x_bytes = x.tobytes()
    combined_bytes = x_bytes + noise_seed.to_bytes(4, "big")
    hash_value = int.from_bytes(sha256(combined_bytes).digest()[:4], "big")
    rng = np.random.default_rng(hash_value)
    rastrigin_value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    noise = rng.normal(loc=0.0, scale=noise)
    return rastrigin_value + noise


class ObjectiveSurfaceGenerator:
    def __init__(self, generator: str):
        self.generator = generator

    def predict(self, params):
        x = np.array(list(params.values()), dtype=float)

        if self.generator == "rastrigin":
            y = noisy_rastrigin(x=x)

        return y


def confopt_artificial_objective_function(
    performance_generator: ObjectiveSurfaceGenerator,
):
    def objective_function(configuration):
        # TODO: check that values always unravels in right order, don't think it does for dicts
        return performance_generator.predict(params=configuration)

    return objective_function


objective_function_in_scope = confopt_artificial_objective_function(
    performance_generator=ObjectiveSurfaceGenerator(
        generator="rastrigin",
    )
)

best_values = []
primary_estimator_errors = []
breaches = []
for i in range(3):
    conformal_searcher = ObjectiveConformalSearcher(
        objective_function=objective_function_in_scope,
        search_space=confopt_params,
        metric_optimization="inverse",
    )

    sampler = UCBSampler(c=1, interval_width=0.8, adapter_framework="ACI")
    # sampler = ThompsonSampler(
    #     n_quantiles=4, adapter_framework=None, enable_optimistic_sampling=True
    # )
    searcher = LocallyWeightedConformalSearcher(
        point_estimator_architecture="kr",
        variance_estimator_architecture="kr",
        sampler=sampler,
    )
    # searcher = MultiFitQuantileConformalSearcher(
    #     quantile_estimator_architecture="qgbm",
    #     sampler=sampler,
    # )
    # searcher = SingleFitQuantileConformalSearcher(
    #     quantile_estimator_architecture="qknn",
    #     sampler=sampler,
    # )

    conformal_searcher.search(
        searcher=searcher,
        n_random_searches=20,
        max_iter=50,
        conformal_retraining_frequency=1,
        random_state=i * 2,
        searcher_tuning_framework="fixed",
    )
    best_value = conformal_searcher.get_best_value()
    best_values.append(best_value)
    breaches_list = []
    error_list = []
    for trial in conformal_searcher.study.trials:
        if trial.primary_estimator_error is not None:
            error_list.append(trial.primary_estimator_error)
        if trial.breached_interval is not None:
            breaches_list.append(trial.breached_interval)

    primary_estimator_errors.append(np.mean(np.array(error_list)))
    breaches.append(np.mean(np.array(breaches_list)))

print(f"Average best value: {np.mean(np.array(best_values))}")
print(f"Std of best values: {np.std(np.array(best_values))}")
print(f"Avg estimator error: {np.mean(np.array(primary_estimator_errors))}")
print(f"Avg breaches: {np.mean(np.array(breaches))}")

# print(trial)

print(f"Avg estimator error: {np.mean(np.array(primary_estimator_errors))}")

# %%
