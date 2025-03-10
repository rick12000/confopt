from typing import List, Union, Optional
from lightgbm import LGBMRegressor

import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import NearestNeighbors
from statsmodels.regression.quantile_regression import QuantReg


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
        raise NotImplementedError(
            "Subclasses should implement the _get_submodel_predictions() method."
        )

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
        random_state: Optional[int] = None,
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
            "random_state": random_state,
        }
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits a RandomForestRegressor on the training data.
        """
        self.fitted_model = RandomForestRegressor(**self.rf_kwargs)
        self.fitted_model.fit(X, y)

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


class QuantRegressionWrapper:
    """
    Wrapper for statsmodels QuantReg to make it compatible with sklearn-style API.
    """

    def __init__(self, alpha: float = 0.5, max_iter: int = 1000, p_tol: float = 1e-6):
        """
        Initialize the QuantReg wrapper with parameters.

        Parameters
        ----------
        alpha : float
            The quantile to fit (between 0 and 1)
        max_iter : int
            Maximum number of iterations for optimization
        p_tol : float
            Convergence tolerance
        """
        self.alpha = alpha  # The quantile level
        self.max_iter = max_iter
        self.p_tol = p_tol
        self.model = None
        self.result = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit quantile regression model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        # Add intercept column to X if not present
        if not np.any(np.all(X == 1, axis=0)):
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X

        # Create and fit the model
        self.model = QuantReg(y, X_with_intercept)
        self.result = self.model.fit(
            q=self.alpha, max_iter=self.max_iter, p_tol=self.p_tol
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.result is None:
            raise ValueError("Model has not been fitted yet.")

        # Add intercept column to X if not present
        if not np.any(np.all(X == 1, axis=0)):
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X

        return self.result.predict(X_with_intercept)


class QuantileLasso(BaseQuantileEstimator):
    """
    Quantile Lasso regression using statsmodels (L1-penalized quantile regression).
    Inherits from BaseQuantileEstimator.

    This implementation fits a separate model for each quantile and uses them for prediction.
    """

    def __init__(
        self,
        quantiles: List[float],
        alpha: float = 0.1,  # Regularization strength (λ)
        max_iter: int = 1000,
        p_tol: float = 1e-6,  # Precision tolerance
        random_state: int = None,
    ):
        """
        Parameters
        ----------
        quantiles : List[float]
            List of quantiles to predict (values between 0 and 1).
        alpha : float, default=0.1
            L1 regularization parameter (lambda).
        max_iter : int, default=1000
            Maximum number of iterations.
        p_tol : float, default=1e-6
            Precision tolerance for convergence.
        random_state : int, optional
            Seed for random number generation.
        """
        # Create model parameters without quantiles
        model_params = {
            "max_iter": max_iter,
            "p_tol": p_tol,
            # alpha parameter is the quantile value in QuantReg,
            # so we'll pass it during fit
        }

        # Initialize with the QuantRegressionWrapper class as model_class
        super().__init__(
            quantiles=quantiles,
            model_class=QuantRegressionWrapper,
            model_params=model_params,
        )

        # Store the regularization parameter separately as it has a naming conflict
        # with the quantile parameter in QuantReg
        self.reg_alpha = alpha
        self.random_state = random_state

    def fit(self, X: np.array, y: np.array):
        """
        Fits a model for each quantile.

        Parameters
        ----------
        X : np.array
            Feature matrix.
        y : np.array
            Target vector.
        """
        self.trained_estimators = []
        for quantile in self.quantiles:
            # Each estimator gets the quantile value as its alpha parameter
            params_with_quantile = {**self.model_params, "alpha": quantile}
            quantile_estimator = self.model_class(**params_with_quantile)
            quantile_estimator.fit(X, y)
            self.trained_estimators.append(quantile_estimator)

        return self

    def __str__(self):
        return "QuantileLasso()"

    def __repr__(self):
        return "QuantileLasso()"
