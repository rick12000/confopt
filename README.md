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

As an example, let's tune a Random Forest model on a regression task.

Start by setting up your training and validation data:

```python
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
split_idx = int(len(X) * 0.5)
X_train, y_train = X[:split_idx, :], y[:split_idx]
X_val, y_val = X[split_idx:, :], y[split_idx:]
```

Then import the Random Forest model to tune and define a search space for
its parameters:

```python
from sklearn.ensemble import RandomForestRegressor

parameter_search_space = {
    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400],
    "min_samples_split": [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    "min_samples_leaf": [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    "max_features": [None, 0.8, 0.9, 1],
}
```

Now create an instance of the `ConformalSearcher` class with the model to
tune, the training and validation data and the parameter search space. Then
use the `search` method to trigger a conformal hyperparameter search of your
model:

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
    n_random_searches=20,
    runtime_budget=90,
    confidence_level=0.5,
)
```

Once done, you can retrieve the best parameters obtained in tuning using:

```python
searcher.get_best_params()
```

Or obtain an already initialized model with:

```python
searcher.get_best_model()
```

More information on use cases can be found in the full
documentation or in the `examples` folder of the main repository.
