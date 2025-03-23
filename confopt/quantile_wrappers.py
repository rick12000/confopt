from typing import List, Union, Optional
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from statsmodels.regression.quantile_regression import QuantReg


class BaseMultiFitQuantileEstimator:
    def __init__(self, model_class: type, model_params: dict):
        self.model_class = model_class
        self.model_params = model_params
        self.trained_estimators = []
        self.quantiles = None

    def fit(self, X: np.array, y: np.array, quantiles: List[float] = None):
        if quantiles is not None:
            self.quantiles = quantiles
        if self.quantiles is None or len(self.quantiles) == 0:
            raise ValueError(
                "Quantiles must be provided either in initialization or fit method"
            )
        self._validate_quantiles()
        self.trained_estimators = []
        for quantile in self.quantiles:
            params_with_quantile = {**self.model_params, "alpha": quantile}
            quantile_estimator = self.model_class(**params_with_quantile)
            quantile_estimator.fit(X, y)
            self.trained_estimators.append(quantile_estimator)
        return self

    def _validate_quantiles(self):
        if not all(0 <= q <= 1 for q in self.quantiles):
            raise ValueError("All quantiles must be between 0 and 1")

    def predict(self, X: np.array) -> np.array:
        if not self.trained_estimators:
            raise RuntimeError("Model must be fitted before prediction")
        y_pred = np.column_stack(
            [estimator.predict(X) for estimator in self.trained_estimators]
        )
        return y_pred


class BaseSingleFitQuantileEstimator:
    def __init__(self):
        self.fitted_model = None
        self.quantiles = None

    def fit(self, X: np.ndarray, y: np.ndarray, quantiles: List[float] = None):
        self.quantiles = quantiles
        if self.quantiles is None or len(self.quantiles) == 0:
            raise ValueError("Quantiles must be provided in fit method")
        # Validate quantiles
        if not all(0 <= q <= 1 for q in self.quantiles):
            raise ValueError("All quantiles must be between 0 and 1")
        # Call implementation-specific fit
        self._fit_implementation(X, y)
        return self

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError(
            "Subclasses should implement the _fit_implementation() method."
        )

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Subclasses should implement the _get_submodel_predictions() method."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.quantiles is None:
            raise ValueError("Model must be fitted with quantiles before prediction")
        candidate_distribution = self._get_candidate_local_distribution(X)
        percentiles = [q * 100 for q in self.quantiles]
        quantile_preds = np.percentile(candidate_distribution, percentiles, axis=1).T
        return quantile_preds


class QuantRegressionWrapper:
    def __init__(self, alpha: float = 0.5, max_iter: int = 1000, p_tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.p_tol = p_tol
        self.model = None
        self.result = None
        self.has_added_intercept = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.has_added_intercept = not np.any(np.all(X == 1, axis=0))
        if self.has_added_intercept:
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X
        self.model = QuantReg(y, X_with_intercept)
        self.result = self.model.fit(
            q=self.alpha, max_iter=self.max_iter, p_tol=self.p_tol
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.has_added_intercept:
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X
        return self.result.predict(X_with_intercept)


class QuantileLasso(BaseMultiFitQuantileEstimator):
    def __init__(
        self,
        alpha: float = 0.1,
        max_iter: int = 1000,
        p_tol: float = 1e-6,
        random_state: int = None,
    ):
        model_params = {"max_iter": max_iter, "p_tol": p_tol}
        super().__init__(model_class=QuantRegressionWrapper, model_params=model_params)
        self.reg_alpha = alpha
        self.random_state = random_state

    def __str__(self):
        return "QuantileLasso()"

    def __repr__(self):
        return "QuantileLasso()"


class QuantileGBM(BaseMultiFitQuantileEstimator):
    def __init__(
        self,
        learning_rate: float,
        n_estimators: int,
        min_samples_split: Union[float, int],
        min_samples_leaf: Union[float, int],
        max_depth: int,
        subsample: float = 1.0,
        max_features: Union[str, float, int] = None,
        random_state: int = None,
    ):
        model_params = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
            "subsample": subsample,
            "max_features": max_features,
            "random_state": random_state,
            "loss": "quantile",
        }
        model_params = {k: v for k, v in model_params.items() if v is not None}
        super().__init__(
            model_class=GradientBoostingRegressor, model_params=model_params
        )

    def __str__(self):
        return "QuantileGBM()"

    def __repr__(self):
        return "QuantileGBM()"


class QuantileLightGBM(BaseMultiFitQuantileEstimator):
    def __init__(
        self,
        learning_rate: float,
        n_estimators: int,
        max_depth: Optional[int] = None,
        min_child_samples: Optional[int] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
        min_child_weight: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs
    ):
        model_params = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "random_state": random_state,
            "objective": "quantile",
            "metric": "quantile",
            "verbose": -1,
            **kwargs,
        }
        model_params = {k: v for k, v in model_params.items() if v is not None}
        super().__init__(model_class=LGBMRegressor, model_params=model_params)

    def __str__(self):
        return "QuantileLightGBM()"

    def __repr__(self):
        return "QuantileLightGBM()"


class QuantileForest(BaseSingleFitQuantileEstimator):
    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 5,
        max_features: float = 0.8,
        min_samples_split: int = 2,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.rf_kwargs = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "bootstrap": bootstrap,
            "random_state": random_state,
        }

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        self.fitted_model = RandomForestRegressor(**self.rf_kwargs)
        self.fitted_model.fit(X, y)
        return self

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        sub_preds = np.column_stack(
            [estimator.predict(X) for estimator in self.fitted_model.estimators_]
        )
        return sub_preds


class QuantileKNN(BaseSingleFitQuantileEstimator):
    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.nn_model = None

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        self.nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm="ball_tree", leaf_size=40
        )
        self.nn_model.fit(X)
        return self

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        _, indices = self.nn_model.kneighbors(X)
        neighbor_preds = self.y_train[indices]
        return neighbor_preds
