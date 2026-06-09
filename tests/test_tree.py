import numpy as np

from numcompute.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)


def test_classifier_fit():

    X = np.array([
        [0],
        [1],
        [2],
        [3]
    ])

    y = np.array([
        0,
        0,
        1,
        1
    ])

    model = DecisionTreeClassifier()

    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)


def test_classifier_partial_fit():

    X = np.array([
        [0],
        [1],
        [2]
    ])

    y = np.array([
        0,
        0,
        1
    ])

    model = DecisionTreeClassifier()

    model.partial_fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)


def test_predict_proba_shape():

    X = np.array([
        [0],
        [1],
        [2]
    ])

    y = np.array([
        0,
        0,
        1
    ])

    model = DecisionTreeClassifier()

    model.fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape[0] == len(X)


def test_regressor_fit():

    X = np.array([
        [1],
        [2],
        [3]
    ])

    y = np.array([
        10,
        20,
        30
    ])

    model = DecisionTreeRegressor()

    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)


def test_regressor_partial_fit():

    X = np.array([
        [1],
        [2]
    ])

    y = np.array([
        10,
        20
    ])

    model = DecisionTreeRegressor()

    model.partial_fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)