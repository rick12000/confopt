from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

from confopt.tuning import ConformalSearcher

# Set up toy data:
X, y = fetch_california_housing(return_X_y=True)
split_idx = int(len(X) * 0.5)
X_train, y_train = X[:split_idx, :], y[:split_idx]
X_val, y_val = X[split_idx:, :], y[split_idx:]

# Define parameter search space:
n_estimators_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400]
min_samples_split_list = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
min_samples_leaf_list = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
max_features_list = [None, 0.8, 0.9, 1]

parameter_dict = {
    "n_estimators": n_estimators_list,
    "min_samples_split": min_samples_split_list,
    "min_samples_leaf": min_samples_leaf_list,
    "max_features": max_features_list,
}

# Set up conformal searcher instance:
searcher = ConformalSearcher(
    model=RandomForestRegressor(),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    search_space=parameter_dict,
    prediction_type="regression",
)

# Carry out hyperparameter search:
searcher.search(
    conformal_search_estimator="qgbm",
    n_random_searches=20,
    runtime_budget=90,
    confidence_level=0.5,
)

# Extract results, in the form of either:

# 1. The best hyperparamter configuration found during search
best_params = searcher.get_best_params()

# 2. An initialized (but not trained) model object with the
#    best hyperparameter configuration found during search
model_init = searcher.get_best_model()

# 3. A trained model with the best hyperparameter configuration
#    found during search
model = searcher.get_best_fitted_model()
