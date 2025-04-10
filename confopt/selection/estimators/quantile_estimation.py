from typing import List, Union, Optional
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.base import clone
from abc import ABC, abstractmethod


class BaseMultiFitQuantileEstimator(ABC):
    def fit(self, X: np.array, y: np.array, quantiles: List[float]):
        self.trained_estimators = []
        for quantile in quantiles:
            quantile_estimator = self._fit_quantile_estimator(X, y, quantile)
            self.trained_estimators.append(quantile_estimator)
        return self

    @abstractmethod
    def _fit_quantile_estimator(self, X: np.array, y: np.array, quantile: float):
        pass

    def predict(self, X: np.array) -> np.array:
        if not self.trained_estimators:
            raise RuntimeError("Model must be fitted before prediction")

        y_pred = np.column_stack(
            [estimator.predict(X) for estimator in self.trained_estimators]
        )
        return y_pred


class BaseSingleFitQuantileEstimator(ABC):
    def fit(self, X: np.ndarray, y: np.ndarray, quantiles: List[float]):
        self.quantiles = quantiles
        self._fit_implementation(X, y)
        return self

    @abstractmethod
    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        candidate_distribution = self._get_candidate_local_distribution(X)
        quantile_preds = np.quantile(candidate_distribution, self.quantiles, axis=1).T
        return quantile_preds


class QuantRegWrapper:
    def __init__(self, results, has_intercept):
        self.results = results
        self.has_intercept = has_intercept

    def predict(self, X):
        if self.has_intercept:
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X

        return X_with_intercept @ self.results.params


class QuantileLasso(BaseMultiFitQuantileEstimator):
    def __init__(
        self,
        max_iter: int = 1000,
        p_tol: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.p_tol = p_tol
        self.random_state = random_state

    def _fit_quantile_estimator(self, X: np.array, y: np.array, quantile: float):
        has_added_intercept = not np.any(np.all(X == 1, axis=0))
        if has_added_intercept:
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X

        if self.random_state is not None:
            np.random.seed(self.random_state)

        model = QuantReg(y, X_with_intercept)
        result = model.fit(q=quantile, max_iter=self.max_iter, p_tol=self.p_tol)
        return QuantRegWrapper(result, has_added_intercept)


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
        super().__init__()
        self.base_estimator = GradientBoostingRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            loss="quantile",
        )

    def _fit_quantile_estimator(self, X: np.array, y: np.array, quantile: float):
        estimator = clone(self.base_estimator)
        estimator.set_params(alpha=quantile)
        estimator.fit(X, y)
        return estimator


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
    ):
        super().__init__()
        self.base_estimator = LGBMRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=random_state,
            objective="quantile",
            metric="quantile",
            verbose=-1,
        )

    def _fit_quantile_estimator(self, X: np.array, y: np.array, quantile: float):
        estimator = clone(self.base_estimator)
        estimator.set_params(alpha=quantile)
        estimator.fit(X, y)
        return estimator


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
        self.base_estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            random_state=random_state,
        )

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        self.fitted_model = self.base_estimator
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
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="ball_tree", leaf_size=40
        )

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        self.nn_model.fit(X)
        return self

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        _, indices = self.nn_model.kneighbors(X)
        neighbor_preds = self.y_train[indices]
        return neighbor_preds
