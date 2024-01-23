import abc
from typing import List, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor


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
