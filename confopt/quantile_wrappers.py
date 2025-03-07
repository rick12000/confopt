from typing import List, Union, Optional
from lightgbm import LGBMRegressor

import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    # HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import NearestNeighbors

# from sklearn.base import BaseEstimator

# from sklearn.neighbors import KNeighborsRegressor
# from statsmodels.regression.quantile_regression import QuantReg


class BaseQuantileEstimator:
    """
    Base class for quantile estimators using customizable models.
    """

    def __init__(
        self,
        quantiles: List[float],
        model_class: type,
        model_params: dict,
    ):
        """
        Initializes the BaseQuantileEstimator with the specified model and quantiles.

        Parameters
        ----------
        quantiles: List[float]
            List of quantiles to predict.
        model_class: type
            The class of the model to be used for quantile prediction.
        model_params: dict
            Dictionary of hyperparameters for the model.
        """
        self.quantiles = quantiles
        self.model_class = model_class
        self.model_params = model_params
        self.trained_estimators = []

    def fit(self, X: np.array, y: np.array):
        """
        Fits the model for each quantile.

        Parameters
        ----------
        X: np.array
            Feature variables.
        y: np.array
            Target variable.
        """
        self.trained_estimators = []
        for quantile in self.quantiles:
            params_with_quantile = {**self.model_params, "alpha": quantile}
            quantile_estimator = self.model_class(**params_with_quantile)
            quantile_estimator.fit(X, y)
            self.trained_estimators.append(quantile_estimator)

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the target variable for each quantile.

        Parameters
        ----------
        X: np.array
            Feature variables.

        Returns
        -------
        np.array
            A 2D numpy array with each column corresponding to a quantile's predictions.
        """
        y_pred = np.column_stack(
            [estimator.predict(X) for estimator in self.trained_estimators]
        )
        return y_pred


class QuantileGBM(BaseQuantileEstimator):
    """
    Quantile gradient boosted machine estimator.
    Inherits from BaseQuantileEstimator and uses GradientBoostingRegressor.
    """

    def __init__(
        self,
        quantiles: List[float],
        learning_rate: float,
        n_estimators: int,
        min_samples_split: Union[float, int],
        min_samples_leaf: Union[float, int],
        max_depth: int,
        subsample: float = 1.0,
        max_features: Union[str, float, int] = None,
        random_state: int = None,
    ):
        """
        Initializes the QuantileGBM with GBM-specific hyperparameters.

        Parameters
        ----------
        quantiles: List[float]
            List of quantiles to predict.
        learning_rate: float
            Learning rate for the GBM.
        n_estimators: int
            Number of boosting stages to perform.
        min_samples_split: Union[float, int]
            Minimum number of samples required to split an internal node.
        min_samples_leaf: Union[float, int]
            Minimum number of samples required to be at a leaf node.
        max_depth: int
            Maximum depth of the individual regression estimators.
        subsample: float
            The fraction of samples to be used for fitting the individual base learners.
        max_features: Union[str, float, int]
            The number of features to consider when looking for the best split.
        random_state: int
            Seed for random number generation.
        """
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
        # Remove None values
        model_params = {k: v for k, v in model_params.items() if v is not None}
        super().__init__(
            quantiles=quantiles,
            model_class=GradientBoostingRegressor,
            model_params=model_params,
        )

    def __str__(self):
        return "QuantileGBM()"

    def __repr__(self):
        return "QuantileGBM()"


class QuantileLightGBM(BaseQuantileEstimator):
    """
    Quantile LightGBM estimator.

    This estimator leverages LGBMRegressor for quantile regression by setting
    the objective to "quantile" and specifying the desired quantile via the
    'alpha' parameter.
    """

    def __init__(
        self,
        quantiles: List[float],
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
        **kwargs,
    ):
        """
        Initializes the QuantileLightGBM with LightGBM-specific hyperparameters.

        Parameters
        ----------
        quantiles : List[float]
            List of quantiles to predict. Each value should be between 0 and 1.
        learning_rate : float
            The learning rate for the boosting process.
        n_estimators : int
            The number of boosting iterations (equivalent to max_iter).
        max_depth : int, optional
            The maximum depth of the individual trees.
        min_child_samples : int, optional
            Minimum number of data needed in a leaf.
        subsample : float, optional
            Fraction of samples used for training trees.
        colsample_bytree : float, optional
            Fraction of features used for training each tree.
        reg_alpha : float, optional
            L1 regularization term.
        reg_lambda : float, optional
            L2 regularization term.
        min_child_weight : int, optional
            Minimum sum of instance weight needed in a child.
        random_state : int, optional
            Seed for random number generation.
        **kwargs :
            Additional keyword arguments to pass to LGBMRegressor.
        """
        # Set up parameters for LGBMRegressor. For quantile regression,
        # we specify objective="quantile".
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
        # Clean None values from parameters
        model_params = {k: v for k, v in model_params.items() if v is not None}
        super().__init__(
            quantiles=quantiles,
            model_class=LGBMRegressor,
            model_params=model_params,
        )

    def __str__(self):
        return "QuantileLightGBM()"

    def __repr__(self):
        return "QuantileLightGBM()"


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


class BaseSingleFitQuantileEstimator:
    """
    Base class for quantile estimators that are fit only once and then produce
    quantile predictions by aggregating a set of predictions (e.g., from sub-models
    or from nearest neighbors).

    Child classes should implement the fit() method and, if needed, override
    _get_submodel_predictions().
    """

    def __init__(self):
        """
        Parameters
        ----------
        quantiles : List[float]
            List of quantiles to predict (values between 0 and 1).
        """
        self.fitted_model = None  # For ensemble models (e.g., forest)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the underlying model. Subclasses should implement this.
        """
        raise NotImplementedError("Subclasses should implement the fit() method.")

    def _get_submodel_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Retrieves a collection of predictions for each sample.

        Default implementation assumes that self.fitted_model has an attribute
        'estimators_' (e.g. for ensembles like RandomForestRegressor). This method
        should be overridden for models that do not follow this pattern (e.g. KNN).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for prediction.

        Returns
        -------
        np.ndarray
            An array of shape (n_samples, n_predictions) where each row contains
            multiple predictions whose distribution will be used to compute quantiles.
        """
        if not hasattr(self.fitted_model, "estimators_"):
            raise ValueError(
                "The fitted model does not have an 'estimators_' attribute."
            )
        # Collect predictions from each sub-model (e.g. tree in a forest)
        sub_preds = np.column_stack(
            [estimator.predict(X) for estimator in self.fitted_model.estimators_]
        )
        return sub_preds

    def predict(self, X: np.ndarray, quantiles: List[float]) -> np.ndarray:
        """
        Computes quantile predictions for each sample by aggregating predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for prediction.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_samples, len(quantiles)), where each column
            corresponds to a quantile prediction.
        """
        submodel_preds = self._get_submodel_predictions(X)
        # Convert quantiles (0-1) to percentiles (0-100)
        percentiles = [q * 100 for q in quantiles]
        quantile_preds = np.percentile(submodel_preds, percentiles, axis=1).T
        return quantile_preds


class QuantileForest(BaseSingleFitQuantileEstimator):
    """
    Quantile estimator based on an ensemble (e.g., RandomForestRegressor).
    The quantile is computed as the percentile of predictions from the ensemble's
    individual sub-models (e.g., trees).
    """

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 5,
        max_features: float = 0.8,
        min_samples_split: int = 2,
        bootstrap: bool = True,
        **rf_kwargs,
    ):
        """
        Parameters
        ----------
        **rf_kwargs : dict
            Additional keyword arguments to pass to RandomForestRegressor.
        """
        super().__init__()
        self.rf_kwargs = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "bootstrap": bootstrap,
        }
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits a RandomForestRegressor on the training data.
        """
        self.fitted_model = RandomForestRegressor(**self.rf_kwargs)
        self.fitted_model.fit(X, y)


class QuantileKNN(BaseSingleFitQuantileEstimator):
    """
    Quantile KNN estimator: for each query sample, finds the m nearest neighbors
    in the training data and returns the desired quantile of their target values.
    """

    def __init__(self, n_neighbors: int = 5):
        """
        Parameters
        ----------
        n_neighbors : int, default=5
            The number of neighbors to use for the quantile estimation.
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.nn_model = None  # NearestNeighbors model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Stores the training data and fits a NearestNeighbors model.
        """
        self.X_train = X
        self.y_train = y

        # Use ball_tree algorithm which is generally faster for high dimensions
        # and specify a larger leaf size for better performance
        self.nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm="ball_tree", leaf_size=40
        )
        self.nn_model.fit(X)

    def _get_submodel_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        For each sample in X, finds the n_neighbors in the training data and
        returns their target values.

        Returns
        -------
        np.ndarray
            An array of shape (n_samples, n_neighbors) containing neighbor target values.
        """
        # Get indices of nearest neighbors for each sample
        _, indices = self.nn_model.kneighbors(X)
        # Retrieve the corresponding y values for the neighbors
        neighbor_preds = self.y_train[indices]  # shape: (n_samples, n_neighbors)
        return neighbor_preds


# from annoy import AnnoyIndex
# # Assuming BaseSingleFitQuantileEstimator is already defined as in the previous snippet

# class QuantileKNNApprox(BaseSingleFitQuantileEstimator):
#     """
#     Approximate Quantile KNN estimator using Annoy for fast nearest neighbor search.
#     For each query sample, the approximate m nearest neighbors are fetched from the training data,
#     and the target quantile is computed from their target values.
#     """
#     def __init__(self, quantiles: List[float], n_neighbors: int = 5, n_trees: int = 10, metric: str = 'euclidean'):
#         """
#         Parameters
#         ----------
#         quantiles : List[float]
#             List of quantiles to predict (values between 0 and 1).
#         n_neighbors : int, default=5
#             Number of neighbors to use for quantile estimation.
#         n_trees : int, default=10
#             Number of trees to build in the Annoy index (more trees gives higher accuracy at the expense of speed).
#         metric : str, default='euclidean'
#             Distance metric for Annoy. Common options include 'euclidean' and 'manhattan'.
#         """
#         super().__init__(quantiles)
#         self.n_neighbors = n_neighbors
#         self.n_trees = n_trees
#         self.metric = metric
#         self.X_train = None
#         self.y_train = None
#         self.annoy_index = None

#     def fit(self, X: np.ndarray, y: np.ndarray):
#         """
#         Fits the approximate nearest neighbor index (Annoy) on the training data.
#         """
#         self.X_train = X
#         self.y_train = y
#         n_features = X.shape[1]
#         self.annoy_index = AnnoyIndex(n_features, self.metric)
#         for i, row in enumerate(X):
#             self.annoy_index.add_item(i, row.tolist())
#         self.annoy_index.build(self.n_trees)
#         return self

#     def _get_submodel_predictions(self, X: np.ndarray) -> np.ndarray:
#         """
#         For each sample in X, uses the Annoy index to quickly retrieve the approximate
#         n_neighbors from the training data, then returns their target values.

#         Returns
#         -------
#         np.ndarray
#             Array of shape (n_samples, n_neighbors) with the neighbors' target values.
#         """
#         neighbor_vals = []
#         for x in X:
#             # Get the indices of the approximate nearest neighbors for this sample
#             indices = self.annoy_index.get_nns_by_vector(x.tolist(), self.n_neighbors)
#             neighbor_vals.append(self.y_train[indices])
#         return np.array(neighbor_vals)
