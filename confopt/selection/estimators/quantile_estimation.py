from typing import List, Union, Optional
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.base import clone
from abc import ABC, abstractmethod
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    ConstantKernel as C,
)
from sklearn.cluster import KMeans


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


class GaussianProcessQuantileEstimator(BaseSingleFitQuantileEstimator):
    def __init__(
        self,
        kernel=None,
        alpha: float = 1e-10,
        n_samples: int = 1000,
        random_state: Optional[int] = None,
        n_inducing_points: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_optimized_sampling: bool = True,
    ):
        super().__init__()
        self.kernel = kernel
        self.alpha = alpha
        self.n_samples = n_samples
        self.random_state = random_state
        self.n_inducing_points = n_inducing_points
        self.batch_size = batch_size
        self.use_optimized_sampling = use_optimized_sampling
        self._kernel_cache = {}
        self._ppf_cache = {}

    def _get_kernel_object(self, kernel_name=None):
        """Convert a kernel name string to a scikit-learn kernel object."""
        if kernel_name is None:
            return C(1.0) * Matern(length_scale=3, nu=1.5)

        if isinstance(kernel_name, str):
            if kernel_name in self._kernel_cache:
                return self._kernel_cache[kernel_name]

            if kernel_name == "rbf":
                kernel_obj = C(1.0) * RBF(length_scale=1.0)
            elif kernel_name == "matern":
                kernel_obj = C(1.0) * Matern(length_scale=3, nu=1.5)
            elif kernel_name == "rational_quadratic":
                kernel_obj = C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
            elif kernel_name == "exp_sine_squared":
                kernel_obj = C(1.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0)
            else:
                raise ValueError(f"Unknown kernel name: {kernel_name}")

            self._kernel_cache[kernel_name] = kernel_obj
            return kernel_obj

        # If the kernel is already a kernel object, return it as is
        return kernel_name

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        if self.n_inducing_points is not None and self.n_inducing_points < len(X):
            try:
                kmeans = KMeans(
                    n_clusters=self.n_inducing_points, random_state=self.random_state
                )
                kmeans.fit(X)
                inducing_points = kmeans.cluster_centers_

                self.gp = GaussianProcessRegressor(
                    kernel=self._get_kernel_object(self.kernel),
                    alpha=self.alpha,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=self.random_state,
                )

                # Pre-compute kernel matrices for sparse approximation
                K_XZ = self._get_kernel_object(self.kernel)(X, inducing_points)
                K_ZZ = (
                    self._get_kernel_object(self.kernel)(inducing_points)
                    + np.eye(self.n_inducing_points) * 1e-10
                )
                K_ZZ_inv = np.linalg.inv(K_ZZ)

                # Compute inducing point weights
                self.inducing_points = inducing_points
                alpha = np.linalg.multi_dot([K_ZZ_inv, K_XZ.T, y])
                self.inducing_weights = alpha

                # We still fit the full GP model for cases when the sparse approach is not suitable
                self.gp.fit(X, y)
            except Exception:
                # Fall back to regular GP if sparse approximation fails
                self.gp = GaussianProcessRegressor(
                    kernel=self._get_kernel_object(self.kernel),
                    alpha=self.alpha,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=self.random_state,
                )
                self.gp.fit(X, y)
        else:
            self.gp = GaussianProcessRegressor(
                kernel=self._get_kernel_object(self.kernel),
                alpha=self.alpha,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self.random_state,
            )
            self.gp.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Override the base class predict method to use analytical Gaussian quantiles
        rather than sampling, ensuring monotonicity of quantiles.
        """
        # Process in batches for large data
        if self.batch_size is not None and len(X) > self.batch_size:
            results = []
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i : i + self.batch_size]
                batch_result = self._predict_batch(batch_X)
                results.append(batch_result)
            return np.vstack(results)
        else:
            return self._predict_batch(X)

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        # Get mean and std from the GP model
        y_mean, y_std = self.gp.predict(X, return_std=True)
        y_std = y_std.reshape(-1, 1)  # For proper broadcasting

        # Vectorize quantile computation for efficiency
        # Cache ppf values since they're the same for all predictions with same quantiles
        ppf_values = self._get_cached_ppf_values()

        # Use broadcasting for efficient computation: each row + each quantile
        quantile_preds = y_mean.reshape(-1, 1) + y_std * ppf_values.reshape(1, -1)

        return quantile_preds

    def _get_cached_ppf_values(self):
        # Cache the ppf values for reuse
        quantiles_key = tuple(self.quantiles)
        if quantiles_key not in self._ppf_cache:
            self._ppf_cache[quantiles_key] = np.array(
                [norm.ppf(q) for q in self.quantiles]
            )
        return self._ppf_cache[quantiles_key]

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        if not self.use_optimized_sampling:
            # For each test point, get mean and std from GP
            y_mean, y_std = self.gp.predict(X, return_std=True)

            # Set random seed for reproducibility
            rng = np.random.RandomState(self.random_state)

            # Generate samples from the GP posterior for each test point
            samples = np.array(
                [
                    rng.normal(y_mean[i], y_std[i], size=self.n_samples)
                    for i in range(len(X))
                ]
            )
            return samples

        # Optimized sampling with vectorization
        y_mean, y_std = self.gp.predict(X, return_std=True)
        y_std = y_std.reshape(-1, 1)  # Reshape for broadcasting

        # Generate all samples at once with broadcasting
        rng = np.random.RandomState(self.random_state)
        noise = rng.normal(0, 1, size=(len(X), self.n_samples))
        samples = y_mean.reshape(-1, 1) + y_std * noise

        return samples
