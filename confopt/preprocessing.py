import random
from typing import Tuple, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


def train_val_split(
    X: np.array,
    y: np.array,
    train_split: float,
    normalize: bool = True,
    ordinal: bool = False,
    random_state: int = None,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Split X and y data into training and validation sets.

    Splits can be carried out randomly or sequentially, with or without normalization.

    Parameters
    ----------
    X :
        Feature variables.
    y :
        Target variable.
    train_split :
        Percentage of training data to carve out of the overall
        data. Values must be contained in the [0, 1] interval.
    normalize :
        Whether X features in both the training and validation
        splits should be normalized according to the training split.
    ordinal :
        Whether the split should occur ordinally (only set to True
        if the X and y data was passed according to some sequential
        order, eg. sorted by date), else split will be random.
    random_state :
        Random seed.

    Returns
    -------
    X_train :
        X features training split.
    y_train :
        y target training split.
    X_val :
        X features validation split.
    y_val :
        y target validation split.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    if ordinal:
        train_idx = list(range(len(X) - round(len(X) * (1 - train_split))))
        val_idx = list(range(len(X) - round(len(X) * (1 - train_split)), len(X)))
    else:
        train_idx = list(
            np.random.choice(len(X), round(len(X) * train_split), replace=False)
        )
        val_idx = list(np.setdiff1d(np.arange(len(X)), train_idx))

    X_val = X[val_idx, :]
    X_train = X[train_idx, :]

    y_val = y[val_idx]
    y_train = y[train_idx]

    if normalize:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val


def remove_iqr_outliers(
    X: np.array, y: np.array, scope: str, iqr_factor: Optional[float] = 1.5
) -> Tuple[np.array, np.array]:
    """
    Remove data outliers via interquartile range filtering.

    Interquartile range is applied to target variable only.

    Parameters
    ----------
    X :
        Feature variables.
    y :
        Target variable.
    scope :
        Determines which outliers are removed. Takes:
            - 'top_only': Only upper threshold outliers are removed.
            - 'bottom_only': Only lower threshold outliers are removed.
            - 'top_and_bottom': All outliers are removed.
    iqr_factor :
        Factor by which to multiply the interquartile range when
        determining outlier thresholds.

    Returns
    -------
    X_retained :
        Outlier filtered X features variables.
    y_retained :
        Outlier filtered y target variable.
    """
    q1 = np.quantile(y, 0.25)
    q3 = np.quantile(y, 0.75)
    iqr = abs(q3 - q1)

    bottom_outlier_idxs = list(np.where(y < (q1 - iqr_factor * iqr))[0])
    top_outlier_idxs = list(np.where(y > (q3 + iqr_factor * iqr))[0])

    if scope == "top_only":
        outlier_idxs = top_outlier_idxs.copy()
    elif scope == "bottom_only":
        outlier_idxs = bottom_outlier_idxs.copy()
    elif scope == "top_and_bottom":
        outlier_idxs = top_outlier_idxs + bottom_outlier_idxs
    else:
        raise ValueError(
            "'scope' can only take one of 'top_only', 'bottom_only' or 'top_and_bottom', "
            f"but {scope} was passed."
        )

    retained_idxs = list(set(list(range(0, len(X)))) - set(outlier_idxs))
    X_retained = X[retained_idxs, :]
    y_retained = y[retained_idxs]

    return X_retained, y_retained
