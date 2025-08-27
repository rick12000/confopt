"""Feature scaling utilities that respect categorical column types.

This module provides scaling functionality that only applies normalization
to continuous features while leaving categorical features unchanged.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from confopt.utils.configurations.encoded_data import EncodedData, ColumnType


class CategoricalAwareScaler:
    """Feature scaler that only normalizes continuous columns.

    This scaler applies StandardScaler only to continuous features,
    leaving categorical (one-hot and level-encoded) features unchanged.
    """

    def __init__(self):
        self.continuous_scaler = None
        self.continuous_indices = None
        self.n_features = None

    def fit(self, X):
        """Fit the scaler on continuous features only.

        Args:
            X: Input data - can be EncodedData or numpy array
        """
        if isinstance(X, EncodedData):
            # Get continuous column indices
            self.continuous_indices = np.where(
                X.get_column_mask(ColumnType.CONTINUOUS)
            )[0]
            self.n_features = len(X.columns)

            if len(self.continuous_indices) > 0:
                # Fit scaler only on continuous columns
                continuous_data = X.to_numpy()[:, self.continuous_indices]
                self.continuous_scaler = StandardScaler()
                self.continuous_scaler.fit(continuous_data)
            else:
                self.continuous_scaler = None
        else:
            # Fallback: treat all features as continuous
            self.continuous_indices = np.arange(X.shape[1])
            self.n_features = X.shape[1]
            self.continuous_scaler = StandardScaler()
            self.continuous_scaler.fit(X)

        return self

    def transform(self, X):
        """Transform data by scaling only continuous features.

        Args:
            X: Input data - can be EncodedData or numpy array

        Returns:
            Scaled data as numpy array
        """
        if isinstance(X, EncodedData):
            X_array = X.to_numpy()
        else:
            X_array = X.copy() if hasattr(X, "copy") else np.array(X)

        if self.continuous_scaler is not None and len(self.continuous_indices) > 0:
            # Scale only continuous columns
            X_array[:, self.continuous_indices] = self.continuous_scaler.transform(
                X_array[:, self.continuous_indices]
            )

        return X_array

    def fit_transform(self, X):
        """Fit and transform in one step.

        Args:
            X: Input data - can be EncodedData or numpy array

        Returns:
            Scaled data as numpy array
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Inverse transform scaled data.

        Args:
            X: Scaled data as numpy array

        Returns:
            Data in original scale as numpy array
        """
        X_inv = X.copy() if hasattr(X, "copy") else np.array(X)

        if self.continuous_scaler is not None and len(self.continuous_indices) > 0:
            # Inverse scale only continuous columns
            X_inv[
                :, self.continuous_indices
            ] = self.continuous_scaler.inverse_transform(
                X_inv[:, self.continuous_indices]
            )

        return X_inv
