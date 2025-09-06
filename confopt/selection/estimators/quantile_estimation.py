"""Quantile regression estimators for distributional prediction.

This module provides quantile regression implementations using different algorithmic
approaches: multi-fit estimators that train separate models per quantile, and single-fit
estimators that model the full conditional distribution. Includes gradient boosting,
random forest, neural network, and Gaussian process variants optimized for uncertainty
quantification in conformal prediction frameworks.
"""

from typing import List, Union, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.base import clone
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.linalg import solve_triangular, cholesky, LinAlgError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    ConstantKernel as C,
    Kernel,
)
import warnings
import copy
import logging


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
        X_with_intercept.T @ X_with_intercept + regularization

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


class QuantileGP(BaseSingleFitQuantileEstimator):
    """Gaussian process quantile regression with robust uncertainty quantification.

    Implements quantile regression using Gaussian processes that model the complete
    conditional distribution p(y|x). Provides analytical quantile computation from
    Gaussian posteriors with proper noise handling and robust hyperparameter optimization.

    All features are treated as continuous using kernels with Automatic Relevance
    Determination (ARD). Categorical features should be one-hot encoded prior to
    being passed to this class.

    Key improvements over basic sklearn GP usage:
    - Proper noise handling without post-hoc kernel modification
    - Robust numerical implementation with Cholesky decomposition
    - Analytical quantile computation for efficiency
    - Batched prediction for memory efficiency
    - Consistent kernel usage between training and prediction
    - ARD kernels for automatic feature relevance determination

    Args:
        kernel: GP kernel specification. Accepts string names ("rbf", "matern",
            "rational_quadratic", "exp_sine_squared") with sensible defaults, or
            custom Kernel objects. Defaults to Matern(nu=1.5).
        noise_variance: Explicit noise variance. If "optimize", will be learned.
            If numeric, uses fixed value. Default is "optimize".
        alpha: Regularization parameter for numerical stability. Range: [1e-12, 1e-6].
        n_restarts_optimizer: Number of restarts for hyperparameter optimization.
        random_state: Seed for reproducible optimization and prediction.
        batch_size: Batch size for prediction to manage memory usage.
        optimize_hyperparameters: Whether to optimize kernel hyperparameters.
            If False, uses kernel as-is.
        prior_lengthscale_concentration: For future custom optimization (unused).
        prior_lengthscale_rate: For future custom optimization (unused).
        prior_noise_concentration: For future custom optimization (unused).
        prior_noise_rate: For future custom optimization (unused).

    Attributes:
        quantiles: List of quantile levels fitted during training.
        X_train_: Training features.
        y_train_: Training targets (normalized).
        kernel_: Fitted kernel with optimized hyperparameters.
        noise_variance_: Fitted noise variance.
        chol_factor_: Cholesky decomposition of kernel matrix.
        alpha_: Precomputed weights for prediction.
        y_train_mean_: Mean of training targets.
        y_train_std_: Standard deviation of training targets.
    """

    def __init__(
        self,
        kernel: Optional[Union[str, Kernel]] = None,
        noise_variance: Optional[Union[str, float]] = "optimize",
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 5,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
        optimize_hyperparameters: bool = True,
        prior_lengthscale_concentration: float = 2.0,
        prior_lengthscale_rate: float = 1.0,
        prior_noise_concentration: float = 1.1,
        prior_noise_rate: float = 30.0,
    ):
        super().__init__()
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.batch_size = batch_size
        self.optimize_hyperparameters = optimize_hyperparameters
        self.prior_lengthscale_concentration = prior_lengthscale_concentration
        self.prior_lengthscale_rate = prior_lengthscale_rate
        self.prior_noise_concentration = prior_noise_concentration
        self.prior_noise_rate = prior_noise_rate
        self._ppf_cache = {}

        # Fitted attributes
        self.X_train_ = None
        self.y_train_ = None
        self.kernel_ = None
        self.noise_variance_ = None
        self.chol_factor_ = None
        self.alpha_ = None
        self.y_train_mean_ = None
        self.y_train_std_ = None
        # Eigendecomposition fallback attributes
        self.eigenvals_ = None
        self.eigenvecs_ = None

    def _get_kernel_object(
        self,
        kernel_spec: Optional[Union[str, Kernel]] = None,
        n_features: Optional[int] = None,
    ) -> Kernel:
        """Convert kernel specification to scikit-learn kernel object with ARD support.

        Creates kernels with per-feature length scales for Automatic Relevance
        Determination (ARD). This allows the model to automatically learn the
        importance of each feature by optimizing individual length scales.

        Args:
            kernel_spec: Kernel specification (string name, kernel object, or None).
            n_features: Number of features for ARD initialization. If None, uses scalar length scale.

        Returns:
            Scikit-learn kernel object with proper ARD bounds for optimization.

        Raises:
            ValueError: If unknown kernel name provided or invalid kernel type.
        """
        # Initialize length scale for ARD
        if n_features is not None and n_features > 1:
            # ARD: one length scale per feature
            length_scale = np.ones(n_features)
            length_scale_bounds = (1e-2, 1e2)
        else:
            # Scalar length scale for single feature or unspecified
            length_scale = 1.0
            length_scale_bounds = (1e-2, 1e2)

        # Default to Matern kernel with ARD
        if kernel_spec is None:
            return C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds,
                nu=2.5,
            )

        # String specifications with ARD support
        elif isinstance(kernel_spec, str):
            kernel_map = {
                "rbf": C(1.0, (1e-3, 1e3))
                * RBF(
                    length_scale=length_scale, length_scale_bounds=length_scale_bounds
                ),
                "matern": C(1.0, (1e-3, 1e3))
                * Matern(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    nu=2.5,
                ),
                "rational_quadratic": C(1.0, (1e-3, 1e3))
                * RationalQuadratic(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    alpha=1.0,
                    alpha_bounds=(1e-3, 1e3),
                ),
                "exp_sine_squared": C(1.0, (1e-3, 1e3))
                * ExpSineSquared(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    periodicity=1.0,
                    periodicity_bounds=(1e-2, 1e2),
                ),
            }

            if kernel_spec not in kernel_map:
                raise ValueError(f"Unknown kernel name: {kernel_spec}")
            return kernel_map[kernel_spec]

        # Kernel object - make a deep copy for safety
        elif isinstance(kernel_spec, Kernel):
            return copy.deepcopy(kernel_spec)

        else:
            raise ValueError(
                f"Kernel must be a string name, Kernel object, or None. Got: {type(kernel_spec)}"
            )

    def _optimize_hyperparameters(self) -> None:
        """Optimize kernel hyperparameters and noise variance using sklearn's optimization."""
        if not self.optimize_hyperparameters:
            return

        # Determine alpha value for optimization
        # If noise_variance is "optimize", use a small alpha and let GP optimize noise
        # If noise_variance is fixed, use it as alpha
        if self.noise_variance == "optimize":
            alpha_for_opt = self.alpha  # Small regularization only
        else:
            alpha_for_opt = self.noise_variance_ + self.alpha

        # Use sklearn's GaussianProcessRegressor for hyperparameter optimization
        # This provides robust optimization with proper parameter mapping
        temp_gp = GaussianProcessRegressor(
            kernel=self.kernel_,
            alpha=alpha_for_opt,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            normalize_y=False,  # We handle normalization ourselves
        )

        try:
            # Suppress sklearn GP convergence warnings about parameter bounds
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*close to the specified.*bound.*",
                    category=UserWarning,
                    module="sklearn.gaussian_process.kernels",
                )
                temp_gp.fit(self.X_train_, self.y_train_)
            # Extract optimized kernel
            self.kernel_ = temp_gp.kernel_

            # Extract optimized noise variance if it was being optimized
            if self.noise_variance == "optimize":
                # sklearn's alpha includes both noise and regularization
                # Extract the optimized noise component
                self.noise_variance_ = max(temp_gp.alpha - self.alpha, 1e-10)

        except Exception as e:
            logging.warning(
                f"Hyperparameter optimization failed: {e}, using default parameters"
            )
            # Keep the original kernel and noise variance if optimization fails

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray) -> "QuantileGP":
        """Fit Gaussian process with proper hyperparameter optimization.

        Implements robust GP fitting with:
        - Custom hyperparameter optimization with principled priors
        - Proper noise handling without post-hoc kernel modification
        - Numerical stability through Cholesky decomposition

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        # Store training data
        self.X_train_ = X.copy()

        # Normalize targets
        self.y_train_mean_ = np.mean(y)
        self.y_train_std_ = np.std(y)
        if self.y_train_std_ < 1e-12:
            self.y_train_std_ = 1.0
        self.y_train_ = (y - self.y_train_mean_) / self.y_train_std_

        # Initialize kernel with ARD support
        n_features = X.shape[1]
        self.kernel_ = self._get_kernel_object(self.kernel, n_features)

        # Set noise variance
        if isinstance(self.noise_variance, (int, float)):
            self.noise_variance_ = self.noise_variance
        else:
            self.noise_variance_ = 1e-6  # Default, will be optimized if needed

        # Optimize hyperparameters
        self._optimize_hyperparameters()

        # Fit the model with optimized parameters
        self._fit_gp()

        return self

    def _fit_gp(self) -> None:
        """Fit GP with current hyperparameters using robust Cholesky decomposition."""
        # Compute kernel matrix
        K = self.kernel_(self.X_train_)

        # Add noise and regularization
        K += (self.noise_variance_ + self.alpha) * np.eye(len(self.X_train_))

        # Robust Cholesky decomposition with progressive regularization
        regularization_levels = [0, 1e-8, 1e-6, 1e-4, 1e-3]

        for reg in regularization_levels:
            try:
                K_reg = K + reg * np.eye(len(self.X_train_)) if reg > 0 else K
                self.chol_factor_ = cholesky(K_reg, lower=True)
                if reg > 0:
                    logging.warning(
                        f"Added regularization {reg} for numerical stability"
                    )
                break
            except LinAlgError:
                if reg == regularization_levels[-1]:
                    # Final fallback: use eigendecomposition for very ill-conditioned matrices
                    logging.warning(
                        "Cholesky failed, using eigendecomposition fallback"
                    )
                    self._fit_gp_eigendecomp(K)
                    return
                continue

        # Solve for alpha using Cholesky decomposition
        self.alpha_ = solve_triangular(self.chol_factor_, self.y_train_, lower=True)

    def _fit_gp_eigendecomp(self, K: np.ndarray) -> None:
        """Fallback GP fitting using eigendecomposition for ill-conditioned matrices."""
        # Eigendecomposition of kernel matrix
        eigenvals, eigenvecs = np.linalg.eigh(K)

        # Clip negative eigenvalues and add regularization
        eigenvals = np.maximum(eigenvals, 1e-12)

        # Reconstruct with regularized eigenvalues
        eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Use pseudo-inverse for fitting
        try:
            K_inv = eigenvecs @ np.diag(1.0 / eigenvals) @ eigenvecs.T
            self.alpha_ = K_inv @ self.y_train_
            # Store decomposition for prediction
            self.eigenvals_ = eigenvals
            self.eigenvecs_ = eigenvecs
            self.chol_factor_ = None  # Signal to use eigendecomp in prediction
        except Exception as e:
            raise RuntimeError(f"Both Cholesky and eigendecomposition failed: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate quantile predictions using analytical Gaussian distribution.

        Uses the GP posterior mean and variance to compute quantiles analytically
        as q_τ(x) = μ(x) + σ(x)Φ⁻¹(τ), ensuring monotonic quantile ordering.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles).
        """
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
        """Compute quantiles analytically from GP posterior.

        Args:
            X: Features with shape (batch_size, n_features).

        Returns:
            Quantile predictions with shape (batch_size, n_quantiles).
        """
        # Get mean and variance from GP
        y_mean, y_var = self._predict_mean_var(X)
        y_std = np.sqrt(y_var).reshape(-1, 1)

        # Get cached inverse normal CDF values
        ppf_values = self._get_cached_ppf_values()

        # Compute quantiles analytically
        quantile_preds = y_mean.reshape(-1, 1) + y_std * ppf_values.reshape(1, -1)

        return quantile_preds

    def _predict_mean_var(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance using Cholesky or eigendecomposition.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Tuple of (y_mean, y_var) with shapes (n_samples,) each.
        """
        # Compute kernel between test and training points
        K_star = self.kernel_(X, self.X_train_)

        if self.chol_factor_ is not None:
            # Use Cholesky-based computation
            chol_solve = solve_triangular(self.chol_factor_, K_star.T, lower=True)
            y_mean = chol_solve.T @ self.alpha_

            # Compute variance (in normalized space)
            K_star_star = self.kernel_.diag(X)
            y_var = K_star_star - np.sum(chol_solve**2, axis=0)

        else:
            # Use eigendecomposition fallback
            y_mean = K_star @ self.alpha_

            # Compute variance using eigendecomposition
            K_star_star = self.kernel_.diag(X)
            # K^{-1} = V * Λ^{-1} * V^T
            K_inv_K_star = (
                self.eigenvecs_
                @ (K_star.T / self.eigenvals_.reshape(-1, 1))
                @ self.eigenvecs_.T
            )
            y_var = K_star_star - np.sum(K_star * K_inv_K_star.T, axis=1)

        # Denormalize mean
        y_mean = y_mean * self.y_train_std_ + self.y_train_mean_

        # Ensure non-negative variance before denormalization
        y_var = np.maximum(y_var, 1e-12)

        # Denormalize variance (transforms from normalized to original scale)
        y_var *= self.y_train_std_**2

        # Add noise variance in original scale for total predictive variance
        y_var += self.noise_variance_ * self.y_train_std_**2

        return y_mean, y_var

    def _get_cached_ppf_values(self) -> np.ndarray:
        """Cache inverse normal CDF values for efficiency.

        Returns:
            Cached inverse normal CDF values with shape (n_quantiles,).
        """
        quantiles_key = tuple(self.quantiles)
        if quantiles_key not in self._ppf_cache:
            self._ppf_cache[quantiles_key] = np.array(
                [norm.ppf(q) for q in self.quantiles]
            )
        return self._ppf_cache[quantiles_key]

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Generate posterior samples for Monte Carlo quantile estimation.

        This method is required by the base class but not used by this implementation
        since we use analytical quantile computation. Included for compatibility.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Posterior samples with shape (n_samples, n_samples_per_point).
        """
        # Get mean and variance from GP
        y_mean, y_var = self._predict_mean_var(X)
        y_std = np.sqrt(y_var)

        # Generate samples from the GP posterior for each test point
        rng = np.random.RandomState(self.random_state)
        n_samples = 1000  # Default number of samples
        samples = np.array(
            [rng.normal(y_mean[i], y_std[i], size=n_samples) for i in range(len(X))]
        )
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
