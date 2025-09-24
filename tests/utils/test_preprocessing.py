import numpy as np
import pytest

from confopt.utils.preprocessing import train_val_split

DEFAULT_SEED = 1234


@pytest.mark.parametrize("train_split", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("ordinal", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_train_val_split(train_split, ordinal, normalize):
    dummy_array = np.zeros((100, 3))
    X = dummy_array[:, :2]
    y = dummy_array[:, 2]
    X_train, y_train, X_val, y_val = train_val_split(
        X=X,
        y=y,
        train_split=train_split,
        ordinal=ordinal,
        normalize=normalize,
        random_state=DEFAULT_SEED,
    )

    assert len(X_val) == len(y_val)
    assert len(X_train) == len(y_train)

    assert len(X_val) + len(X_train) == len(X)

    assert abs(len(X_train) - round(len(X) * train_split)) <= 1
    assert abs(len(X_val) - round(len(X) * (1 - train_split))) <= 1


@pytest.mark.parametrize("train_split", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("ordinal", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_train_val_split__reproducibility(train_split, ordinal, normalize):
    dummy_array = np.zeros((100, 3))
    X = dummy_array[:, :2]
    y = dummy_array[:, 2]
    (
        X_train_first_call,
        y_train_first_call,
        X_val_first_call,
        y_val_first_call,
    ) = train_val_split(
        X=X,
        y=y,
        train_split=train_split,
        ordinal=ordinal,
        normalize=normalize,
        random_state=DEFAULT_SEED,
    )
    (
        X_train_second_call,
        y_train_second_call,
        X_val_second_call,
        y_val_second_call,
    ) = train_val_split(
        X=X,
        y=y,
        train_split=train_split,
        ordinal=ordinal,
        normalize=normalize,
        random_state=DEFAULT_SEED,
    )
    assert np.array_equal(X_train_first_call, X_train_second_call)
    assert np.array_equal(y_train_first_call, y_train_second_call)
    assert np.array_equal(X_val_first_call, X_val_second_call)
    assert np.array_equal(y_val_first_call, y_val_second_call)
