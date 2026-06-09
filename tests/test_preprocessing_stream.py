import numpy as np

from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder
)


def test_standard_scaler_partial_fit():

    X1 = np.array([[1, 2],
                   [3, 4]])

    X2 = np.array([[5, 6]])

    scaler = StandardScaler()

    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    assert scaler.mean_.shape == (2,)


def test_standard_scaler_transform_shape():

    X = np.array([[1, 2],
                  [3, 4]])

    scaler = StandardScaler()

    Xt = scaler.fit_transform(X)

    assert Xt.shape == X.shape


def test_minmax_partial_fit():

    X1 = np.array([[1, 2]])
    X2 = np.array([[5, 10]])

    scaler = MinMaxScaler()

    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    assert scaler.max_[1] == 10


def test_imputer_partial_fit():

    X1 = np.array([[1, np.nan]])
    X2 = np.array([[3, 5]])

    imp = Imputer()

    imp.partial_fit(X1)
    imp.partial_fit(X2)

    assert imp.fill_.shape == (2,)


def test_onehot_dynamic_categories():

    X1 = np.array([["A"]])
    X2 = np.array([["B"]])

    enc = OneHotEncoder()

    enc.partial_fit(X1)
    enc.partial_fit(X2)

    assert len(enc.categories_[0]) == 2