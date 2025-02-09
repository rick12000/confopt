import abc
from typing import List, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.regression.quantile_regression import QuantReg


class BiQuantileEstimator:
    """
    Base class for bi-quantile estimators.

    Estimators fit on X features to predict two symmetrical conditional
    quantiles of some target Y variable.
    """

    def __init__(
        self,
        quantiles: List[float],
        random_state: int,
    ):
        self.quantiles = quantiles
        self.random_state = random_state

    @abc.abstractmethod
    def fit(self, X: np.array, y: np.array):
        return

    def _predict(
        self,
        lo_quantile_estimator: BaseEstimator,
        hi_quantile_estimator: BaseEstimator,
        X: np.array,
    ) -> np.array:
        """
        Make quantile predictions using features in X.

        Parameters
        ----------
        lo_quantile_estimator :
            Trained lower quantile estimator.
        hi_quantile_estimator :
            Trained upper quantile estimator.
        X :
            Features used to return predictions.

        Returns
        -------
        y_pred :
            Quantile predictions, organized in a len(X) by
            2 array, where the first column contains lower
            quantile predictions, and the second contains
            higher quantile predictions.
        """
        lo_y_pred = lo_quantile_estimator.predict(X).reshape(len(X), 1)
        hi_y_pred = hi_quantile_estimator.predict(X).reshape(len(X), 1)
        y_pred = np.hstack([lo_y_pred, hi_y_pred])

        return y_pred


class QuantileGBM(BiQuantileEstimator):
    """
    Quantile gradient boosted machine estimator.
    """

    def __init__(
        self,
        quantiles: List[float],
        learning_rate: float,
        n_estimators: int,
        min_samples_split: Union[float, int],
        min_samples_leaf: Union[float, int],
        max_depth: int,
        random_state: int,
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        super().__init__(quantiles, random_state)

    def __str__(self):
        return "QuantileGBM()"

    def __repr__(self):
        return "QuantileGBM()"

    def fit(self, X: np.array, y: np.array):
        """
        Trains a bi-quantile GBM model on X and y data.

        Two separate quantile estimators are trained, one predicting
        an upper quantile and one predicting a symmetrical lower quantile.
        The estimators are aggregated in a tuple, for later joint
        use in prediction.

        Parameters
        ----------
        X :
            Feature variables.
        y :
            Target variable.
        """
        trained_estimators = ()
        for quantile in self.quantiles:
            quantile_estimator = GradientBoostingRegressor(
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state,
                loss="quantile",
                alpha=quantile,
            )
            quantile_estimator.fit(X, y)
            trained_estimators = trained_estimators + (quantile_estimator,)
        self.lo_quantile_estimator, self.hi_quantile_estimator = trained_estimators

    def predict(self, X: np.array) -> np.array:
        return self._predict(
            lo_quantile_estimator=self.lo_quantile_estimator,
            hi_quantile_estimator=self.hi_quantile_estimator,
            X=X,
        )


class QuantileKNN(BiQuantileEstimator):
    """
    K-Nearest Neighbors quantile estimator.
    """

    def __init__(self, quantiles: List[float], n_neighbors: int, random_state: int):
        self.n_neighbors = n_neighbors
        super().__init__(quantiles, random_state)

    def __str__(self):
        return "QuantileKNN()"

    def __repr__(self):
        return "QuantileKNN()"

    def fit(self, X: np.array, y: np.array):
        """
        Trains a bi-quantile KNN model on X and y data.
        """
        self.n_neighbors = min(self.n_neighbors, len(X) - 1)
        self.knn_estimator = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, algorithm="kd_tree"
        )
        self.knn_estimator.fit(X, y)

    def predict(self, X: np.array) -> np.array:
        """
        Predicts quantiles by estimating the empirical quantile of nearest neighbors.
        """
        lo_preds, hi_preds = [], []

        for x in X:
            neighbors = self.knn_estimator.kneighbors([x], return_distance=False)[0]
            neighbors_y = self.knn_estimator._y[neighbors]
            lo_quantile = np.quantile(neighbors_y, self.quantiles[0])
            hi_quantile = np.quantile(neighbors_y, self.quantiles[1])

            lo_preds.append(lo_quantile)
            hi_preds.append(hi_quantile)

        return np.column_stack([lo_preds, hi_preds])


class QuantileLasso:
    """
    Quantile Lasso regression using statsmodels (L1-penalized quantile regression).
    Inherits from BiQuantileEstimator (not shown here for brevity).
    """

    def __init__(
        self,
        quantiles: List[float],
        alpha: float = 0.1,  # Regularization strength (λ)
        max_iter: int = 1000,
        random_state: int = None,
    ):
        self.quantiles = quantiles
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Add intercept term (statsmodels does not auto-add it)
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        for q in self.quantiles:
            model = QuantReg(y, X_with_intercept)
            result = model.fit(
                q=q,
                alpha=self.alpha,
                max_iter=self.max_iter,
                p_tol=1e-6,  # Precision tolerance
                # statsmodels uses "alpha" as the L1 regularization strength
            )
            self.models[q] = result

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        predictions = np.zeros((len(X), len(self.quantiles)))
        for i, q in enumerate(self.quantiles):
            predictions[:, i] = self.models[q].predict(X_with_intercept)
        return predictions
