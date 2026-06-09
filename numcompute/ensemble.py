"""
ensemble.py

Tree-based ensemble methods for NumCompute.

Provides:

- RandomForestClassifier
- RandomForestRegressor

Supports:
- fit()
- partial_fit()
- predict()

Assignment 2:
- Ensemble Learning
- Streaming Learning
"""

import numpy as np

from numcompute.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)


# ============================================================
# Base Random Forest
# ============================================================

class _BaseRandomForest:
    """
    Base Random Forest implementation.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees.

    max_depth : int, default=5
        Maximum depth of each tree.

    min_samples_split : int, default=2
        Minimum samples required to split.

    max_features : {"sqrt", "log2", None}, default="sqrt"
        Number of features sampled for each tree.

    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        max_features="sqrt",
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.trees_ = []
        self.feature_indices_ = []

        # streaming support
        self._X_seen = None
        self._y_seen = None

    # --------------------------------------------------------
    # Feature Sampling
    # --------------------------------------------------------

    def _n_features_to_sample(self, n_features):

        if self.max_features is None:
            return n_features

        if self.max_features == "sqrt":
            return max(
                1,
                int(np.sqrt(n_features))
            )

        if self.max_features == "log2":
            return max(
                1,
                int(np.log2(n_features))
            )

        raise ValueError(
            "max_features must be "
            "'sqrt', 'log2', or None"
        )

    # --------------------------------------------------------
    # Streaming Support
    # --------------------------------------------------------

    def partial_fit(self, X, y):
        """
        Incrementally update forest.

        Parameters
        ----------
        X : np.ndarray of shape
            (chunk_size, n_features)

        y : np.ndarray of shape
            (chunk_size,)

        Returns
        -------
        self
        """

        X = np.asarray(X)
        y = np.asarray(y)

        if self._X_seen is None:

            self._X_seen = X.copy()
            self._y_seen = y.copy()

        else:

            self._X_seen = np.vstack(
                [self._X_seen, X]
            )

            self._y_seen = np.concatenate(
                [self._y_seen, y]
            )

        self.fit(
            self._X_seen,
            self._y_seen
        )

        return self


# ============================================================
# Random Forest Classifier
# ============================================================

class RandomForestClassifier(
    _BaseRandomForest
):
    """
    Random Forest Classifier.

    Ensemble of DecisionTreeClassifier models.
    """

    def fit(self, X, y):
        """
        Train random forest.

        Parameters
        ----------
        X : np.ndarray of shape
            (n_samples, n_features)

        y : np.ndarray of shape
            (n_samples,)

        Returns
        -------
        RandomForestClassifier
        """

        rng = np.random.default_rng(
            self.random_state
        )

        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        self.trees_ = []
        self.feature_indices_ = []

        n_selected_features = (
            self._n_features_to_sample(
                n_features
            )
        )

        for _ in range(self.n_estimators):

            sample_idx = rng.choice(
                n_samples,
                size=n_samples,
                replace=True
            )

            feature_idx = rng.choice(
                n_features,
                size=n_selected_features,
                replace=False
            )

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )

            tree.fit(
                X[sample_idx][:, feature_idx],
                y[sample_idx]
            )

            self.trees_.append(tree)
            self.feature_indices_.append(
                feature_idx
            )

        return self

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """

        X = np.asarray(X)

        predictions = np.array([
            tree.predict(
                X[:, features]
            )
            for tree, features in zip(
                self.trees_,
                self.feature_indices_
            )
        ])

        predictions = predictions.T

        result = []

        for row in predictions:

            classes, counts = np.unique(
                row,
                return_counts=True
            )

            winners = classes[
                counts == counts.max()
            ]

            result.append(
                np.min(winners)
            )

        return np.array(result)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """

        X = np.asarray(X)

        classes = np.unique(
            np.concatenate([
                tree.classes_
                for tree in self.trees_
            ])
        )
        n_samples = len(X)
        all_probs = np.zeros(
            (
                len(self.trees_),
                n_samples,
                len(classes)
            )
        )
        for tree_idx, (tree, features) in enumerate(
            zip(
                self.trees_,
                self.feature_indices_
            )
        ):
            tree_probs = tree.predict_proba(
                X[:, features]
            )
            for local_idx, cls in enumerate(
            tree.classes_
        ):
                global_idx = np.where(
                    classes == cls
                )[0][0]

                all_probs[
                    tree_idx,
                    :,
                    global_idx
                ] = tree_probs[:, local_idx]

        return np.mean(
            all_probs,
            axis=0
        )



# ============================================================
# Random Forest Regressor
# ============================================================

class RandomForestRegressor(
    _BaseRandomForest
):
    """
    Random Forest Regressor.

    Ensemble of DecisionTreeRegressor models.
    """

    def fit(self, X, y):
        """
        Train random forest regressor.
        """

        rng = np.random.default_rng(
            self.random_state
        )

        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        self.trees_ = []
        self.feature_indices_ = []

        n_selected_features = (
            self._n_features_to_sample(
                n_features
            )
        )

        for _ in range(self.n_estimators):

            sample_idx = rng.choice(
                n_samples,
                size=n_samples,
                replace=True
            )

            feature_idx = rng.choice(
                n_features,
                size=n_selected_features,
                replace=False
            )

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )

            tree.fit(
                X[sample_idx][:, feature_idx],
                y[sample_idx]
            )

            self.trees_.append(tree)
            self.feature_indices_.append(
                feature_idx
            )

        return self

    def predict(self, X):
        """
        Predict continuous targets.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """

        X = np.asarray(X)

        predictions = np.array([
            tree.predict(
                X[:, features]
            )
            for tree, features in zip(
                self.trees_,
                self.feature_indices_
            )
        ])

        return np.mean(
            predictions,
            axis=0
        )