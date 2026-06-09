import numpy as np

from numcompute.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


def test_rf_classifier_fit():

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

    model = RandomForestClassifier(
        n_estimators=3
    )

    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)


def test_rf_classifier_partial_fit():

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

    model = RandomForestClassifier(
        n_estimators=3
    )

    model.partial_fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)


def test_rf_predict_proba():

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

    model = RandomForestClassifier(
        n_estimators=3
    )

    model.fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape[0] == len(X)


def test_rf_regressor_fit():

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

    model = RandomForestRegressor(
        n_estimators=3
    )

    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)


def test_rf_regressor_partial_fit():

    X = np.array([
        [1],
        [2]
    ])

    y = np.array([
        10,
        20
    ])

    model = RandomForestRegressor(
        n_estimators=3
    )

    model.partial_fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)