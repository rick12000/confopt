<div align="center">
  <img src="assets/logo.png" alt="ConfOpt Logo" width="300"/>
</div>

<br>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-ACHO-cyan)](https://doi.org/10.48550/arXiv.2207.03017)

ConfOpt is an inferential hyperparameter optimization package designed to
speed up model hyperparameter tuning.

The package currently implements Adaptive Conformal Hyperparameter Optimization (ACHO), as detailed
in [the original paper](https://doi.org/10.48550/arXiv.2207.03017).

## Installation

You can install ConfOpt from [PyPI](https://pypi.org/project/confopt) using `pip`:

```bash
pip install confopt
```

## Getting Started

As an example, we'll tune a Random Forest model with data from a regression task.

Start by setting up your training and validation data:

```python
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
split_idx = int(len(X) * 0.5)
X_train, y_train = X[:split_idx, :], y[:split_idx]
X_val, y_val = X[split_idx:, :], y[split_idx:]
```

Then import the Random Forest model to tune and define a search space for
its parameters (must be a dictionary mapping the model's parameter names to
possible values of that parameter to search):

```python
from sklearn.ensemble import RandomForestRegressor

parameter_search_space = {
    "n_estimators": [10, 30, 50, 100, 150, 200, 300, 400],
    "min_samples_split": [0.005, 0.01, 0.1, 0.2, 0.3],
    "min_samples_leaf": [0.005, 0.01, 0.1, 0.2, 0.3],
    "max_features": [None, 0.8, 0.9, 1],
}
```

Now import the `ConformalTuner` class. You'll need to define an `objective_function`
that takes a parameter configuration, trains your model (e.g., `RandomForestRegressor`),
evaluates it on the validation set, and returns a score to be optimized.

Initialize `ConformalTuner` with this `objective_function`, the
`parameter_search_space`, and `metric_optimization` (either "minimize" or "maximize").

Hyperparameter tuning can be kicked off with the `tune` method, specifying
how long the tuning should run for (e.g., `runtime_budget` in seconds):

```python
from confopt.tuning import ConformalTuner
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define the objective function
# This function will be called by ConformalTuner with different hyperparameter configurations
def objective_function(config):
    # Initialize the model with the given configuration
    model = RandomForestRegressor(**config, random_state=42) # Using random_state for reproducibility

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    predictions = model.predict(X_val)

    # Calculate the score (e.g., Mean Squared Error for regression)
    score = mean_squared_error(y_val, predictions)

    return score

# Initialize the ConformalTuner
tuner = ConformalTuner(
    objective_function=objective_function,
    search_space=parameter_search_space,
    metric_optimization="minimize",  # We want to minimize MSE
)

# Start the tuning process
tuner.tune(
    runtime_budget=120  # How many seconds to run the search for
)
```

Once done, you can retrieve the best parameters obtained during tuning using:

```python
best_params = tuner.get_best_params()
print(f"Best parameters found: {best_params}")
```

You can then train your model on the full dataset using these optimal parameters:

```python
# Initialize and train the best model on the full dataset (X, y)
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X, y) # X and y are the complete dataset defined earlier
print("Best model trained on full data.")
```

More information on specific parameters and overrides not mentioned
in this walk-through can be found in the docstrings or in the `examples`
folder of the main repository.
