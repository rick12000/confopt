# ConfOpt

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

Now import the `ConformalSearcher` class and initialize it with:

- The model to tune.
- The raw X and y data.
- The parameter search space.
- An extra variable clarifying whether this is a regression or classification problem.

Hyperparameter tuning can be kicked off with the `search` method and a specification
of how long the tuning should run for (in seconds):

```python
from confopt.tuning import ConformalSearcher

searcher = ConformalSearcher(
    model=RandomForestRegressor(),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    search_space=parameter_search_space,
    prediction_type="regression",
)

searcher.search(
    runtime_budget=120  # How many seconds to run the search for
)
```

Once done, you can retrieve the best parameters obtained during tuning using:

```python
searcher.get_best_params()
```

Or obtain an already initialized model with:

```python
searcher.get_best_model()
```

More information on specific parameters and overrides not mentioned
in this walk-through can be found in the docstrings or in the `examples`
folder of the main repository.
