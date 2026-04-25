import numpy as np
from numcompute.preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder

def test_standard_scaler():
    X = np.array([[1,2],[3,4]])
    Xs = StandardScaler().fit(X).transform(X)
    assert np.allclose(np.mean(Xs, axis=0), 0)

def test_minmax_scaler():
    X = np.array([[1,2],[3,4]])
    Xm = MinMaxScaler().fit(X).transform(X)
    assert Xm.min() == 0 and Xm.max() == 1

def test_imputer():
    X = np.array([[1,np.nan],[3,4]])
    Xi = Imputer().fit(X).transform(X)
    assert not np.isnan(Xi).any()

def test_onehot():
    X = np.array([[1],[2],[1]])
    enc = OneHotEncoder().fit(X).transform(X)
    assert enc.shape[1] == 2