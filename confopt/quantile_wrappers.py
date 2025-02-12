from typing import List, Union

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.neighbors import KNeighborsRegressor
# from statsmodels.regression.quantile_regression import QuantReg


class QuantileGBM:
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
        self.quantiles = quantiles
        self.random_state = random_state

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
        self.trained_estimators = ()
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
            self.trained_estimators = self.trained_estimators + (quantile_estimator,)

    def predict(self, X: np.array) -> np.array:
        y_pred = np.array([])
        for estimator in self.trained_estimators:
            if len(y_pred) == 0:
                y_pred = estimator.predict(X).reshape(len(X), 1)
            else:
                y_pred = np.hstack([y_pred, estimator.predict(X).reshape(len(X), 1)])

        return y_pred


# class QuantileKNN(BiQuantileEstimator):
#     """
#     K-Nearest Neighbors quantile estimator.
#     """

#     def __init__(self, quantiles: List[float], n_neighbors: int, random_state: int):
#         self.n_neighbors = n_neighbors
#         super().__init__(quantiles, random_state)

#     def __str__(self):
#         return "QuantileKNN()"

#     def __repr__(self):
#         return "QuantileKNN()"

#     def fit(self, X: np.array, y: np.array):
#         """
#         Trains a bi-quantile KNN model on X and y data.
#         """
#         self.n_neighbors = min(self.n_neighbors, len(X) - 1)
#         self.knn_estimator = KNeighborsRegressor(
#             n_neighbors=self.n_neighbors, algorithm="kd_tree"
#         )
#         self.knn_estimator.fit(X, y)

#     def predict(self, X: np.array) -> np.array:
#         """
#         Predicts quantiles by estimating the empirical quantile of nearest neighbors.
#         """
#         lo_preds, hi_preds = [], []

#         for x in X:
#             neighbors = self.knn_estimator.kneighbors([x], return_distance=False)[0]
#             neighbors_y = self.knn_estimator._y[neighbors]
#             lo_quantile = np.quantile(neighbors_y, self.quantiles[0])
#             hi_quantile = np.quantile(neighbors_y, self.quantiles[1])

#             lo_preds.append(lo_quantile)
#             hi_preds.append(hi_quantile)

#         return np.column_stack([lo_preds, hi_preds])


# class QuantileLasso:
#     """
#     Quantile Lasso regression using statsmodels (L1-penalized quantile regression).
#     Inherits from BiQuantileEstimator (not shown here for brevity).
#     """

#     def __init__(
#         self,
#         quantiles: List[float],
#         alpha: float = 0.1,  # Regularization strength (λ)
#         max_iter: int = 1000,
#         random_state: int = None,
#     ):
#         self.quantiles = quantiles
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.random_state = random_state
#         self.models = {}

#     def fit(self, X: np.ndarray, y: np.ndarray):
#         # Add intercept term (statsmodels does not auto-add it)
#         X_with_intercept = np.column_stack([np.ones(len(X)), X])

#         for q in self.quantiles:
#             model = QuantReg(y, X_with_intercept)
#             result = model.fit(
#                 q=q,
#                 alpha=self.alpha,
#                 max_iter=self.max_iter,
#                 p_tol=1e-6,  # Precision tolerance
#                 # statsmodels uses "alpha" as the L1 regularization strength
#             )
#             self.models[q] = result

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         X_with_intercept = np.column_stack([np.ones(len(X)), X])
#         predictions = np.zeros((len(X), len(self.quantiles)))
#         for i, q in enumerate(self.quantiles):
#             predictions[:, i] = self.models[q].predict(X_with_intercept)
#         return predictions
