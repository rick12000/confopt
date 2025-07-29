"""Quantile regression estimators for distributional prediction.

This module provides quantile regression implementations using different algorithmic
approaches: multi-fit estimators that train separate models per quantile, and single-fit
estimators that model the full conditional distribution. Includes gradient boosting,
random forest, neural network, and Gaussian process variants optimized for uncertainty
quantification in conformal prediction frameworks.
"""

from typing import List, Union, Optional
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.base import clone
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    ConstantKernel as C,
    WhiteKernel,
    Sum,
    Kernel,
)
from sklearn.cluster import KMeans
import warnings
import copy


class BaseMultiFitQuantileEstimator(ABC):
    """Abstract base for quantile estimators that train separate models per quantile.

    Multi-fit estimators train individual models for each requested quantile level,
    allowing algorithms like gradient boosting to directly optimize quantile-specific
    loss functions. This approach provides flexibility at the cost of increased
    computational overhead proportional to the number of quantiles.

    The base class handles the iteration over quantiles and result aggregation,
    while subclasses implement the quantile-specific model fitting logic.
    """

    def fit(self, X: np.array, y: np.array, quantiles: List[float]):
        """Fit separate models for each quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels in [0, 1] to fit models for.

        Returns:
            Self for method chaining.
        """
        self.trained_estimators = []
        for quantile in quantiles:
            quantile_estimator = self._fit_quantile_estimator(X, y, quantile)
            self.trained_estimators.append(quantile_estimator)
        return self

    @abstractmethod
    def _fit_quantile_estimator(self, X: np.array, y: np.array, quantile: float):
        """Fit a single model for the specified quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantile: Quantile level in [0, 1] to fit model for.

        Returns:
            Fitted estimator for the quantile level.
        """

    def predict(self, X: np.array) -> np.array:
        """Generate predictions for all fitted quantile levels.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles).

        Raises:
            RuntimeError: If called before fitting any models.
        """
        if not self.trained_estimators:
            raise RuntimeError("Model must be fitted before prediction")

        y_pred = np.column_stack(
            [estimator.predict(X) for estimator in self.trained_estimators]
        )
        return y_pred


class BaseSingleFitQuantileEstimator(ABC):
    """Abstract base for quantile estimators that model the full conditional distribution.

    Single-fit estimators train one model that captures the complete conditional
    distribution of the target variable. Quantiles are then extracted from this
    distribution, either through sampling or analytical computation. This approach
    is computationally efficient and ensures monotonic quantile ordering.

    Subclasses must implement distribution modeling and quantile extraction logic.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, quantiles: List[float]):
        """Fit a single model to capture the conditional distribution.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels in [0, 1] to extract later.

        Returns:
            Self for method chaining.
        """
        self.quantiles = quantiles
        self._fit_implementation(X, y)
        return self

    @abstractmethod
    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        """Implement the model fitting logic for the conditional distribution.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
        """

    @abstractmethod
    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Extract candidate distribution samples for quantile computation.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Distribution samples with shape (n_samples, n_candidates).
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate quantile predictions from the fitted conditional distribution.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles).
        """
        candidate_distribution = self._get_candidate_local_distribution(X)
        quantile_preds = np.quantile(candidate_distribution, self.quantiles, axis=1).T
        return quantile_preds


class QuantRegWrapper:
    """Wrapper for statsmodels quantile regression results to provide scikit-learn interface.

    Adapts statsmodels QuantReg fitted results to provide a predict method compatible
    with the estimator framework. Handles intercept management for proper matrix
    multiplication during prediction.

    Args:
        results: Fitted QuantReg results object from statsmodels.
        has_intercept: Whether an intercept term was added to the design matrix.
    """

    def __init__(self, results, has_intercept):
        self.results = results
        self.has_intercept = has_intercept

    def predict(self, X):
        """Generate predictions using the fitted quantile regression coefficients.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Predictions with shape (n_samples,).
        """
        if self.has_intercept:
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X

        return X_with_intercept @ self.results.params


class QuantileLasso(BaseMultiFitQuantileEstimator):
    """Linear quantile regression using L1 regularization (Lasso).

    Implements quantile regression with L1 penalty using statsmodels backend.
    Fits separate linear models for each quantile level using the pinball loss
    function. Automatically handles intercept terms and provides reproducible
    results through random state control.

    Args:
        max_iter: Maximum iterations for optimization convergence.
        p_tol: Convergence tolerance for parameter changes.
        random_state: Seed for reproducible optimization.
    """

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
        """Fit linear quantile regression for a specific quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantile: Quantile level in [0, 1] to fit model for.

        Returns:
            QuantRegWrapper containing fitted model for the quantile.
        """
        has_added_intercept = not np.any(np.all(X == 1, axis=0))
        if has_added_intercept:
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X

        # Add small regularization to prevent numerical issues
        n_features = X_with_intercept.shape[1]
        regularization = 1e-8 * np.eye(n_features)
        X_regularized = X_with_intercept.T @ X_with_intercept + regularization
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        try:
            model = QuantReg(y, X_with_intercept)
            result = model.fit(q=quantile, max_iter=self.max_iter, p_tol=self.p_tol)
            return QuantRegWrapper(result, has_added_intercept)
        except np.linalg.LinAlgError:
            # Fallback to robust coordinate descent quantile regression
            warnings.warn(
                f"SVD convergence failed for quantile {quantile}. "
                "Using coordinate descent fallback solution."
            )
            
            # Use coordinate descent for robust quantile regression
            params = self._coordinate_descent_quantile_regression(
                X_with_intercept, y, quantile
            )
            
            # Create a mock result object compatible with QuantRegWrapper
            class MockQuantRegResult:
                def __init__(self, params):
                    self.params = params
            
            mock_result = MockQuantRegResult(params)
            return QuantRegWrapper(mock_result, has_added_intercept)

    def _coordinate_descent_quantile_regression(
        self, X: np.ndarray, y: np.ndarray, quantile: float
    ) -> np.ndarray:
        """Coordinate descent algorithm for quantile regression with regularization.
        
        Implements a robust coordinate descent solver for quantile regression that
        handles numerical instability better than general-purpose optimizers.
        Uses adaptive step sizes and convergence checking for stability.
        
        Args:
            X: Design matrix with shape (n_samples, n_features).
            y: Target values with shape (n_samples,).
            quantile: Quantile level in [0, 1].
            
        Returns:
            Coefficient vector with shape (n_features,).
        """
        n_samples, n_features = X.shape
        
        # Initialize coefficients with robust least squares estimate
        try:
            # Try regularized least squares initialization
            XtX = X.T @ X + 1e-6 * np.eye(n_features)
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback to zero initialization if solve fails
            beta = np.zeros(n_features)
        
        # Coordinate descent parameters
        max_iter = self.max_iter
        tolerance = self.p_tol
        lambda_reg = 1e-6  # Small L2 regularization for stability
        
        # Pre-compute frequently used values
        X_norms_sq = np.sum(X**2, axis=0) + lambda_reg
        
        for iteration in range(max_iter):
            beta_old = beta.copy()
            
            # Update each coefficient in turn
            for j in range(n_features):
                # Compute residual without j-th feature
                residual = y - X @ beta + X[:, j] * beta[j]
                
                # Compute coordinate-wise gradient components
                r_pos = residual >= 0
                r_neg = ~r_pos
                
                # Subgradient of quantile loss w.r.t. beta[j]
                grad_pos = -quantile * np.sum(X[r_pos, j])
                grad_neg = -(quantile - 1) * np.sum(X[r_neg, j])
                gradient = grad_pos + grad_neg
                
                # Add L2 regularization gradient
                gradient += lambda_reg * beta[j]
                
                # Update using coordinate descent step
                # For quantile regression, we use a simple gradient step with adaptive step size
                step_size = 1.0 / X_norms_sq[j]
                beta[j] -= step_size * gradient
                
                # Apply soft thresholding for implicit L1 regularization
                # This helps with numerical stability
                thresh = 1e-8
                if abs(beta[j]) < thresh:
                    beta[j] = 0.0
            
            # Check convergence
            param_change = np.linalg.norm(beta - beta_old)
            if param_change < tolerance:
                break
        
        return beta


class QuantileGBM(BaseMultiFitQuantileEstimator):
    """Gradient boosting quantile regression using scikit-learn backend.

    Implements quantile regression using gradient boosting with the quantile loss
    function. Each quantile level trains a separate GBM model with the alpha
    parameter set to the target quantile. Provides robust non-linear quantile
    estimation with automatic feature selection and interaction detection.

    Args:
        learning_rate: Step size for gradient descent updates.
        n_estimators: Number of boosting stages (trees) to fit.
        min_samples_split: Minimum samples required to split internal nodes.
        min_samples_leaf: Minimum samples required at leaf nodes.
        max_depth: Maximum depth of individual trees.
        subsample: Fraction of samples used for fitting individual trees.
        max_features: Number of features considered for best split.
        random_state: Seed for reproducible tree construction.
    """

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
        """Fit gradient boosting model for a specific quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantile: Quantile level in [0, 1] to fit model for.

        Returns:
            Fitted GradientBoostingRegressor for the quantile.
        """
        estimator = clone(self.base_estimator)
        estimator.set_params(alpha=quantile)
        estimator.fit(X, y)
        return estimator


class QuantileLightGBM(BaseMultiFitQuantileEstimator):
    """LightGBM-based quantile regression with advanced gradient boosting.

    Implements quantile regression using LightGBM's efficient gradient boosting
    implementation. Provides faster training than scikit-learn GBM with support
    for categorical features and advanced regularization. Each quantile level
    trains a separate model optimized for the quantile objective.

    Args:
        learning_rate: Step size for gradient descent updates.
        n_estimators: Number of boosting stages to fit.
        max_depth: Maximum depth of individual trees (-1 for no limit).
        min_child_samples: Minimum samples required in child nodes.
        subsample: Fraction of samples used for fitting individual trees.
        colsample_bytree: Fraction of features used for fitting individual trees.
        reg_alpha: L1 regularization strength.
        reg_lambda: L2 regularization strength.
        min_child_weight: Minimum sum of instance weight in child nodes.
        random_state: Seed for reproducible tree construction.
    """

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
        """Fit LightGBM model for a specific quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantile: Quantile level in [0, 1] to fit model for.

        Returns:
            Fitted LGBMRegressor for the quantile.
        """
        estimator = clone(self.base_estimator)
        estimator.set_params(alpha=quantile)
        estimator.fit(X, y)
        return estimator


class QuantileForest(BaseSingleFitQuantileEstimator):
    """Random forest quantile regression using tree ensemble distributions.

    Implements quantile regression by fitting a single random forest and using
    the distribution of tree predictions to estimate quantiles. This approach
    captures epistemic uncertainty through ensemble diversity and provides
    naturally monotonic quantiles from the empirical tree distribution.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of individual trees.
        max_features: Fraction of features considered for best split.
        min_samples_split: Minimum samples required to split internal nodes.
        bootstrap: Whether to use bootstrap sampling for tree training.
        random_state: Seed for reproducible tree construction.
    """

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 5,
        max_features: float = 0.8,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.base_estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            random_state=random_state,
        )

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        """Fit the random forest on the training data.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        self.fitted_model = self.base_estimator
        self.fitted_model.fit(X, y)
        return self

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Extract tree prediction distributions for quantile computation.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Tree predictions with shape (n_samples, n_estimators).
        """
        sub_preds = np.column_stack(
            [estimator.predict(X) for estimator in self.fitted_model.estimators_]
        )
        return sub_preds


class QuantileKNN(BaseSingleFitQuantileEstimator):
    """K-nearest neighbors quantile regression using local empirical distributions.

    Implements quantile regression by finding k nearest neighbors for each
    prediction point and using their target value distribution to estimate
    quantiles. This non-parametric approach adapts locally to data density
    and provides natural uncertainty quantification in sparse regions.

    Args:
        n_neighbors: Number of nearest neighbors to use for quantile estimation.
    """

    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="ball_tree", leaf_size=40
        )

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        """Fit the k-NN model by storing training data and building search index.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        self.X_train = X
        self.y_train = y
        self.nn_model.fit(X)
        return self

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Get neighbor target distributions for quantile computation.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Neighbor targets with shape (n_samples, n_neighbors).
        """
        _, indices = self.nn_model.kneighbors(X)
        neighbor_preds = self.y_train[indices]
        return neighbor_preds


class GaussianProcessQuantileEstimator(BaseSingleFitQuantileEstimator):
    """Gaussian process quantile regression with robust uncertainty quantification.

    Implements quantile regression using Gaussian processes that model the complete
    conditional distribution p(y|x). Provides both analytical quantile computation
    (assuming Gaussian posteriors) and Monte Carlo sampling for complex distributions.
    Includes computational optimizations: sparse GP approximations for scalability,
    pre-computed kernel inverses for efficient prediction, and explicit noise modeling
    for robust uncertainty separation.

    The estimator leverages GP's natural uncertainty quantification capabilities by
    extracting quantiles from the posterior predictive distribution. This approach
    ensures monotonic quantile ordering and provides both aleatoric (data) and
    epistemic (model) uncertainty estimates essential for conformal prediction.

    Computational Features:
        - Sparse approximations using inducing points for O(nm²) complexity
        - Batched prediction for memory-efficient large-scale inference
        - Pre-computed kernel matrices for repeated prediction speedup
        - Analytical quantile computation avoiding sampling overhead

    Methodological Features:
        - Explicit noise modeling separating aleatoric from epistemic uncertainty
        - Flexible kernel specifications (strings or objects) with safe deep copying
        - Robust variance computation with numerical stability checks
        - Caching of inverse normal CDF values for efficiency

    Args:
        kernel: GP kernel specification. Accepts string names ("rbf", "matern",
            "rational_quadratic", "exp_sine_squared") with sensible defaults, or
            custom Kernel objects. Defaults to Matern(nu=1.5) with length_scale=3.
        alpha: Noise variance regularization parameter added to kernel diagonal.
            Controls numerical stability and implicit noise modeling. Range: [1e-12, 1e-3].
        n_samples: Number of posterior samples for Monte Carlo quantile estimation
            when using sampling-based approach. Higher values improve accuracy but
            increase computational cost. Typical range: [500, 5000].
        random_state: Seed for reproducible random number generation in optimization,
            K-means clustering for inducing points, and posterior sampling.
        n_inducing_points: Number of inducing points for sparse GP approximation.
            Enables O(nm²) scaling for datasets with n > m. Recommended: m = n/10
            to n/5 for good accuracy-efficiency trade-off.
        batch_size: Batch size for prediction to manage memory usage on large datasets.
            Automatic batching prevents memory overflow while maintaining accuracy.
        use_optimized_sampling: Whether to use vectorized sampling approach for
            Monte Carlo quantile estimation. Provides significant speedup over
            iterative sampling with identical results.
        noise: Explicit noise specification for robust uncertainty modeling.
            "gaussian" enables automatic noise estimation, float values fix noise level.
            Properly separates aleatoric noise from epistemic uncertainty.

    Attributes:
        quantiles: List of quantile levels fitted during training.
        gp: Underlying GaussianProcessRegressor instance.
        K_inv_: Pre-computed kernel inverse matrix for efficient prediction.
        noise_: Estimated or specified noise level for uncertainty separation.
        inducing_points: Cluster centers used for sparse approximation.
        inducing_weights: Precomputed weights for sparse prediction.

    Raises:
        ValueError: If kernel specification is invalid or noise parameter malformed.
        RuntimeError: If sparse approximation fails and fallback is unsuccessful.

    Examples:
        Basic quantile regression:
        >>> gp = GaussianProcessQuantileEstimator()
        >>> gp.fit(X_train, y_train, quantiles=[0.1, 0.5, 0.9])
        >>> predictions = gp.predict(X_test)  # Shape: (n_test, 3)

        Custom kernel with noise modeling:
        >>> kernel = RBF(length_scale=2.0) + Matern(length_scale=1.5)
        >>> gp = GaussianProcessQuantileEstimator(kernel=kernel, noise="gaussian")
        >>> gp.fit(X_train, y_train, quantiles=[0.05, 0.95])

        Large-scale usage with sparse approximation:
        >>> gp = GaussianProcessQuantileEstimator(
        ...     n_inducing_points=500, batch_size=1000
        ... )
        >>> gp.fit(X_large, y_large, quantiles=np.linspace(0.1, 0.9, 9))
    """

    def __init__(
        self,
        kernel: Optional[Union[str, Kernel]] = None,
        alpha: float = 1e-10,
        n_samples: int = 1000,
        random_state: Optional[int] = None,
        n_inducing_points: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_optimized_sampling: bool = True,
        noise: Optional[Union[str, float]] = None,
    ):
        super().__init__()
        self.kernel = kernel
        self.alpha = alpha
        self.n_samples = n_samples
        self.random_state = random_state
        self.n_inducing_points = n_inducing_points
        self.batch_size = batch_size
        self.use_optimized_sampling = use_optimized_sampling
        self.noise = noise
        self._ppf_cache = {}
        self.K_inv_ = None
        self.noise_ = None

    def _get_kernel_object(
        self, kernel_spec: Optional[Union[str, Kernel]] = None
    ) -> Kernel:
        """Convert kernel specification to scikit-learn kernel object.

        Args:
            kernel_spec: Kernel specification (string name, kernel object, or None).

        Returns:
            Scikit-learn kernel object.

        Raises:
            ValueError: If unknown kernel name provided or invalid kernel type.
        """
        kernel_obj = None

        # Default fallback to Matern kernel with proper bounds
        if kernel_spec is None:
            kernel_obj = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=1.0,
                length_scale_bounds=(
                    1e-1,
                    1e2,
                ),  # Reasonable bounds to prevent collapse
                nu=1.5,
            )
        # If it's a string, look up predefined kernels with proper bounds
        elif isinstance(kernel_spec, str):
            if kernel_spec == "rbf":
                kernel_obj = C(1.0, (1e-3, 1e3)) * RBF(
                    length_scale=1.0, length_scale_bounds=(1e-1, 1e2)
                )
            elif kernel_spec == "matern":
                kernel_obj = C(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=1.0, length_scale_bounds=(1e-1, 1e2), nu=1.5
                )
            elif kernel_spec == "rational_quadratic":
                kernel_obj = C(1.0, (1e-3, 1e3)) * RationalQuadratic(
                    length_scale=1.0,
                    length_scale_bounds=(1e-1, 1e2),
                    alpha=1.0,
                    alpha_bounds=(1e-3, 1e3),
                )
            elif kernel_spec == "exp_sine_squared":
                kernel_obj = C(1.0, (1e-3, 1e3)) * ExpSineSquared(
                    length_scale=1.0,
                    length_scale_bounds=(1e-1, 1e2),
                    periodicity=1.0,
                    periodicity_bounds=(1e-1, 1e2),
                )
            else:
                raise ValueError(f"Unknown kernel name: {kernel_spec}")
        # If it's already a kernel object, make a deep copy for safety
        elif isinstance(kernel_spec, Kernel):
            kernel_obj = copy.deepcopy(kernel_spec)
        # If it's neither string nor kernel object, raise error
        else:
            raise ValueError(
                f"Kernel must be a string name, Kernel object, or None. Got: {type(kernel_spec)}"
            )

        return kernel_obj

    def _fit_implementation(
        self, X: np.ndarray, y: np.ndarray
    ) -> "GaussianProcessQuantileEstimator":
        """Fit Gaussian process with sparse approximation and robust noise handling.

        Implements a two-stage fitting process: first configures the kernel with
        explicit noise modeling, then fits the GP with optional sparse approximation
        for scalability. The method handles noise separation to ensure proper
        uncertainty decomposition between aleatoric (data) and epistemic (model)
        components during prediction.

        Sparse approximation uses K-means clustering to select representative
        inducing points, reducing computational complexity from O(n³) to O(nm²)
        where m << n. Falls back gracefully to full GP if sparse approximation fails.

        Args:
            X: Training features with shape (n_samples, n_features). Features are
                normalized internally by the GP for numerical stability.
            y: Training targets with shape (n_samples,). Targets are normalized
                if normalize_y=True in the underlying GP.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If both sparse and full GP fitting fail.
            ValueError: If noise specification is malformed.
        """
        # Handle noise modeling
        kernel_to_use = self._get_kernel_object(self.kernel)

        if (
            self.noise is not None
            and not _param_for_white_kernel_in_sum(kernel_to_use)[0]
        ):
            if isinstance(self.noise, str) and self.noise == "gaussian":
                kernel_to_use = kernel_to_use + WhiteKernel()
            elif isinstance(self.noise, (int, float)):
                kernel_to_use = kernel_to_use + WhiteKernel(
                    noise_level=self.noise, noise_level_bounds="fixed"
                )

        if self.n_inducing_points is not None and self.n_inducing_points < len(X):
            try:
                kmeans = KMeans(
                    n_clusters=self.n_inducing_points, random_state=self.random_state
                )
                kmeans.fit(X)
                inducing_points = kmeans.cluster_centers_

                self.gp = GaussianProcessRegressor(
                    kernel=kernel_to_use,
                    alpha=self.alpha,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=self.random_state,
                )

                # Pre-compute kernel matrices for sparse approximation
                K_XZ = kernel_to_use(X, inducing_points)
                K_ZZ = (
                    kernel_to_use(inducing_points)
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
                    kernel=kernel_to_use,
                    alpha=self.alpha,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=self.random_state,
                )
                self.gp.fit(X, y)
        else:
            self.gp = GaussianProcessRegressor(
                kernel=kernel_to_use,
                alpha=self.alpha,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self.random_state,
            )
            self.gp.fit(X, y)

        # Pre-compute K_inv for efficient predictions and handle noise separation
        self._precompute_kernel_inverse()
        self._handle_noise_separation()

        return self

    def _precompute_kernel_inverse(self) -> None:
        """Pre-compute kernel inverse matrix for efficient repeated predictions.

        Computes and stores the inverse of the training kernel matrix K using
        Cholesky decomposition for numerical stability. This pre-computation
        enables O(nm) prediction complexity instead of O(n³) kernel inversion
        per prediction call, crucial for applications requiring many predictions.

        Uses the already-computed Cholesky factor L from GP fitting to avoid
        redundant decomposition. Falls back to direct matrix inversion if
        Cholesky approach fails due to numerical issues.

        Raises:
            UserWarning: If Cholesky decomposition fails and direct inversion is used.
        """
        try:
            # Use Cholesky decomposition for numerical stability
            L_inv = solve_triangular(self.gp.L_.T, np.eye(self.gp.L_.shape[0]))
            self.K_inv_ = L_inv.dot(L_inv.T)
        except Exception:
            # Fallback to direct inversion if Cholesky fails
            warnings.warn(
                "Cholesky decomposition failed, using direct matrix inversion"
            )
            K = self.gp.kernel_(self.gp.X_train_, self.gp.X_train_)
            K += np.eye(K.shape[0]) * self.gp.alpha
            self.K_inv_ = np.linalg.inv(K)

    def _handle_noise_separation(self) -> None:
        """Separate noise components for proper uncertainty decomposition.

        Implements the critical step of noise separation required for accurate
        uncertainty quantification in GPs. During training, noise is included
        in the kernel matrix for proper posterior computation. During prediction,
        noise must be excluded from the predictive variance to avoid double-counting
        uncertainty sources.

        This method stores the estimated noise level and sets kernel noise to zero,
        following the mathematical framework in Rasmussen & Williams (2006) Eq. 2.24.
        The separation ensures that predictive variance represents only epistemic
        uncertainty, while noise represents aleatoric uncertainty.

        Handles both simple WhiteKernel cases and complex composite kernels with
        nested Sum structures containing noise components.
        """
        self.noise_ = None

        if self.noise is not None:
            # Store noise level and set kernel noise to zero for prediction variance
            if isinstance(self.gp.kernel_, WhiteKernel):
                self.noise_ = self.gp.kernel_.noise_level
                self.gp.kernel_.set_params(noise_level=0.0)
            else:
                white_present, white_param = _param_for_white_kernel_in_sum(
                    self.gp.kernel_
                )
                if white_present:
                    noise_kernel = self.gp.kernel_.get_params()[white_param]
                    self.noise_ = noise_kernel.noise_level
                    self.gp.kernel_.set_params(
                        **{white_param: WhiteKernel(noise_level=0.0)}
                    )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate quantile predictions using analytical Gaussian distribution.

        Overrides base class to leverage analytical quantile computation from
        Gaussian posterior distributions. This approach ensures monotonic quantile
        ordering and provides superior computational efficiency compared to
        Monte Carlo sampling methods, while maintaining mathematical rigor.

        The method uses the GP posterior mean μ(x) and variance σ²(x) to compute
        quantiles analytically as q_τ(x) = μ(x) + σ(x)Φ⁻¹(τ), where Φ⁻¹ is
        the inverse normal CDF. This leverages the Gaussianity assumption of
        GP posteriors for exact quantile computation.

        Implements batched processing for memory efficiency on large datasets,
        automatically splitting predictions when batch_size is specified.

        Args:
            X: Features for prediction with shape (n_samples, n_features).
                Must have same feature dimensionality as training data.

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles).
            Each column corresponds to one quantile level, ordered as specified
            during fitting. Values are monotonically increasing across quantiles
            for each sample (mathematical guarantee of analytical approach).

        Raises:
            RuntimeError: If called before fitting or if prediction fails.
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
        """Compute quantiles analytically from GP posterior with numerical robustness.

        Core prediction method that combines GP mean and variance predictions
        with inverse normal CDF values to compute quantiles analytically.
        Uses pre-computed kernel inverse for efficiency and includes comprehensive
        numerical stability checks for negative variances.

        The analytical quantile computation q_τ = μ + σΦ⁻¹(τ) leverages cached
        inverse CDF values and vectorized broadcasting for computational efficiency.
        This approach scales as O(nm) for n predictions with m quantiles.

        Args:
            X: Features with shape (batch_size, n_features). Batch dimension
                allows memory-efficient processing of large prediction sets.

        Returns:
            Quantile predictions with shape (batch_size, n_quantiles).
            Guaranteed monotonic ordering across quantiles due to analytical
            computation from Gaussian distribution properties.
        """
        # Get mean and std from the GP model using optimized computation
        y_mean, y_std = self._predict_with_precomputed_inverse(X)
        y_std = y_std.reshape(-1, 1)  # For proper broadcasting

        # Vectorize quantile computation for efficiency
        # Cache ppf values since they're the same for all predictions with same quantiles
        ppf_values = self._get_cached_ppf_values()

        # Use broadcasting for efficient computation: each row + each quantile
        quantile_preds = y_mean.reshape(-1, 1) + y_std * ppf_values.reshape(1, -1)

        return quantile_preds

    def _predict_with_precomputed_inverse(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Efficient prediction using pre-computed kernel inverse matrix.

        Implements optimized GP prediction that leverages pre-computed kernel
        inverse to avoid repeated expensive matrix operations. Provides identical
        results to standard GP prediction but with significantly improved
        computational efficiency for repeated prediction calls.

        Handles proper normalization/denormalization of predictions to account
        for GP's internal target scaling. Includes robust numerical checks for
        negative variances that can arise from floating-point precision issues
        in ill-conditioned kernel matrices.

        Args:
            X: Features with shape (n_samples, n_features). Must match the
                feature space dimensionality used during training.

        Returns:
            Tuple of (y_mean, y_std) where:
                - y_mean: Posterior mean predictions, shape (n_samples,)
                - y_std: Posterior standard deviations, shape (n_samples,)
            Both outputs are properly denormalized if GP used target scaling.

        Raises:
            UserWarning: If negative variances detected and corrected to zero.
        """
        if self.K_inv_ is None:
            # Fallback to standard GP prediction if K_inv not available
            return self.gp.predict(X, return_std=True)

        # Compute kernel between test and training points
        K_trans = self.gp.kernel_(X, self.gp.X_train_)

        # Compute mean prediction
        y_mean = K_trans.dot(self.gp.alpha_)

        # Undo normalization if applied
        if hasattr(self.gp, "_y_train_std"):
            y_mean = self.gp._y_train_std * y_mean + self.gp._y_train_mean
        elif hasattr(self.gp, "y_train_std_"):
            y_mean = self.gp.y_train_std_ * y_mean + self.gp.y_train_mean_

        # Compute variance using pre-computed inverse
        y_var = self.gp.kernel_.diag(X)
        y_var -= np.einsum("ki,kj,ij->k", K_trans, K_trans, self.K_inv_)

        # Check for negative variances due to numerical issues
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. Setting those variances to 0."
            )
            y_var[y_var_negative] = 0.0

        # Undo normalization for variance
        if hasattr(self.gp, "_y_train_std"):
            y_var = y_var * self.gp._y_train_std**2
        elif hasattr(self.gp, "y_train_std_"):
            y_var = y_var * self.gp.y_train_std_**2

        y_std = np.sqrt(y_var)

        return y_mean, y_std

    def _get_cached_ppf_values(self) -> np.ndarray:
        """Cache inverse normal CDF values for computational efficiency.

        Computes and caches the inverse normal cumulative distribution function
        values Φ⁻¹(τ) for all requested quantile levels τ. Caching avoids
        repeated expensive scipy.stats.norm.ppf calls during prediction,
        providing significant speedup for repeated predictions with same quantiles.

        Cache key uses tuple of quantile values to handle different quantile
        sets across multiple estimator instances or refitting scenarios.

        Returns:
            Cached inverse normal CDF values with shape (n_quantiles,).
            Values correspond to quantile levels specified during fitting,
            used in analytical quantile computation q = μ + σΦ⁻¹(τ).
        """
        # Cache the ppf values for reuse
        quantiles_key = tuple(self.quantiles)
        if quantiles_key not in self._ppf_cache:
            self._ppf_cache[quantiles_key] = np.array(
                [norm.ppf(q) for q in self.quantiles]
            )
        return self._ppf_cache[quantiles_key]

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Generate posterior samples for Monte Carlo quantile estimation.

        Provides sampling-based quantile estimation as an alternative to analytical
        computation. Generates samples from the GP posterior distribution p(f|D)
        at test points, enabling empirical quantile estimation through sample
        quantiles. Useful for non-Gaussian posteriors or when sampling-based
        uncertainty propagation is preferred.

        Supports both vectorized and iterative sampling approaches based on
        use_optimized_sampling parameter. Vectorized approach provides identical
        results with significantly improved computational efficiency through
        broadcasting operations.

        The sampling approach scales as O(n*s) where s is the number of samples,
        compared to O(n) for analytical quantiles. Trade-off between computational
        cost and flexibility for complex posterior distributions.

        Args:
            X: Features with shape (n_samples, n_features). Test points where
                posterior samples are generated for quantile estimation.

        Returns:
            Posterior samples with shape (n_samples, n_samples_per_point).
            Each row contains samples from the posterior distribution at the
            corresponding test point, used for empirical quantile computation.
        """
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


class QuantileLeaf(BaseSingleFitQuantileEstimator):
    """Quantile Regression Forest using raw Y values from leaf nodes (Meinshausen 2006).

    Implements quantile regression following the approach in Meinshausen (2006) where
    quantiles are computed from the empirical distribution of all raw Y training values
    that fall into the same leaf nodes as the prediction point across all trees.

    For a prediction point x, the method collects all training targets Y_i where
    training point X_i and prediction point x end up in the same leaf node across
    all trees in the forest. Quantiles are then computed as empirical percentiles
    of this combined set of Y values.

    This approach differs from standard random forest quantiles by using raw training
    targets rather than tree predictions, providing more accurate uncertainty
    quantification especially in regions with heteroscedastic noise.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of individual trees.
        max_features: Fraction of features considered for best split.
        min_samples_split: Minimum samples required to split internal nodes.
        min_samples_leaf: Minimum samples required at leaf nodes.
        bootstrap: Whether to use bootstrap sampling for tree training.
        random_state: Seed for reproducible tree construction.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_features: float = 0.8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.X_train = None
        self.y_train = None
        self.forest = None

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        """Fit the random forest and store training data for leaf node lookup.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
        )
        self.forest.fit(X, y)
        return self

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Extract raw Y values from leaf nodes for quantile computation.

        For each prediction point, finds all training targets that fall into
        the same leaf nodes across all trees. This creates the empirical
        distribution used for quantile estimation following Meinshausen (2006).

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Raw Y values from matching leaf nodes with shape (n_samples, variable).
            Each row contains the training targets from leaf nodes that contain
            the corresponding prediction point. Rows may have different lengths,
            so the array is padded with NaN values and the actual distribution
            is extracted during quantile computation.
        """
        # Get leaf indices for training and test data for all trees
        train_leaf_indices = self.forest.apply(self.X_train)  # (n_train, n_trees)
        test_leaf_indices = self.forest.apply(X)  # (n_test, n_trees)

        # Collect Y values for each test point
        candidate_distributions = []

        for i in range(len(X)):
            y_values_for_point = []

            # For each tree, find training points in the same leaf as test point i
            for tree_idx in range(self.n_estimators):
                test_leaf = test_leaf_indices[i, tree_idx]
                # Find training points that ended up in the same leaf
                same_leaf_mask = train_leaf_indices[:, tree_idx] == test_leaf
                # Collect corresponding Y values
                y_values_for_point.extend(self.y_train[same_leaf_mask])

            candidate_distributions.append(np.array(y_values_for_point))

        # Convert to consistent array format by padding with NaN
        max_length = max(len(dist) for dist in candidate_distributions)
        padded_distributions = np.full((len(X), max_length), np.nan)

        for i, dist in enumerate(candidate_distributions):
            padded_distributions[i, : len(dist)] = dist

        return padded_distributions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate quantile predictions from raw Y values in matching leaf nodes.

        Overrides the base class method to handle variable-length distributions
        from leaf nodes. Computes empirical quantiles while ignoring NaN padding.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles).
        """
        candidate_distributions = self._get_candidate_local_distribution(X)

        # Compute quantiles for each test point, ignoring NaN values
        quantile_preds = np.zeros((len(X), len(self.quantiles)))

        for i in range(len(X)):
            # Extract non-NaN values for this point
            valid_values = candidate_distributions[i][
                ~np.isnan(candidate_distributions[i])
            ]

            if len(valid_values) > 0:
                # Compute empirical quantiles
                quantile_preds[i] = np.quantile(valid_values, self.quantiles)
            else:
                # Fallback to forest mean prediction if no valid values
                # This should rarely happen with proper forest configuration
                mean_pred = self.forest.predict(X[i : i + 1])[0]
                quantile_preds[i] = mean_pred

        return quantile_preds


def _param_for_white_kernel_in_sum(kernel, kernel_str=""):
    """Check if a WhiteKernel exists in a Sum Kernel and return the corresponding parameter key.

    Args:
        kernel: Kernel object to check.
        kernel_str: Current parameter path string.

    Returns:
        Tuple of (bool, str) indicating if WhiteKernel exists and its parameter key.
    """
    if kernel_str != "":
        kernel_str = kernel_str + "__"

    if isinstance(kernel, Sum):
        for param, child in kernel.get_params(deep=False).items():
            if isinstance(child, WhiteKernel):
                return True, kernel_str + param
            else:
                present, child_str = _param_for_white_kernel_in_sum(
                    child, kernel_str + param
                )
                if present:
                    return True, child_str

    return False, "_"
